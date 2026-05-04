import html
import re
from typing import Dict, List

from backend.preprocessing import sentence_tokenize, word_tokenize
from backend.retrieval import query_references


MODEL_NAME = "distilbert-base-cased-distilled-squad"
EXPLAINER_MODEL_NAME = "google/flan-t5-small"
MIN_CONFIDENCE = 0.15
MAX_ANSWER_WORDS = 48
MIN_EXPANDED_WORDS = 35
MIN_ANSWER_SENTENCES = 5
ABSTRACT_QUESTION_PREFIXES = (
    "what is",
    "what are",
    "define",
    "meaning of",
    "explain",
    "who is",
    "what do you mean by",
)


class QAModelService:
    def __init__(self, model_name: str = MODEL_NAME, explainer_model_name: str = EXPLAINER_MODEL_NAME):
        self.model_name = model_name
        self.explainer_model_name = explainer_model_name
        self._pipeline = None
        self._explainer_pipeline = None

    def load(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for question answering. Install dependencies first."
                ) from exc
            self._pipeline = pipeline("question-answering", model=self.model_name)
        return self._pipeline

    def load_explainer(self):
        if self._explainer_pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for grounded explanation generation. Install dependencies first."
                ) from exc
            self._explainer_pipeline = pipeline(
                "text2text-generation",
                model=self.explainer_model_name,
            )
        return self._explainer_pipeline

    def answer(self, question: str, context: str) -> Dict[str, object]:
        qa_pipeline = self.load()
        prediction = qa_pipeline(question=question, context=context)
        answer = clean_answer(prediction.get("answer", ""))
        snippet = expand_answer_context(
            context=context,
            answer=answer,
            start=prediction.get("start"),
            end=prediction.get("end"),
        )
        if not snippet:
            snippet = best_source_snippet(context, answer, question, max_sentences=3)
        explanation = generate_grounded_explanation(
            service=self,
            question=question,
            answer_span=answer,
            evidence=snippet or context,
        )
        return {
            "answer": explanation,
            "answer_span": answer,
            "confidence": round(float(prediction.get("score", 0.0)), 4),
            "start": prediction.get("start"),
            "end": prediction.get("end"),
            "model": f"{self.model_name} + {self.explainer_model_name}",
            "source_snippet": snippet,
        }


def highlight_answer(text: str, answer: str) -> str:
    safe_text = html.escape(text)
    if not answer:
        return safe_text
    start = text.lower().find(answer.lower())
    if start == -1:
        return safe_text
    end = start + len(answer)
    return (
        f"{html.escape(text[:start])}"
        f"<mark>{html.escape(text[start:end])}</mark>"
        f"{html.escape(text[end:])}"
    )


def fallback_answer(
    question: str,
    retrieval_results: List[Dict[str, object]],
    service: QAModelService | None = None,
) -> Dict[str, object]:
    if not retrieval_results:
        return {
            "answer": "",
            "confidence": 0.0,
            "model": "retrieval-fallback",
            "source_snippet": "",
        }
    snippet = build_fallback_passage(question, retrieval_results)
    best = max(retrieval_results, key=lambda item: item.get("score", 0.0))
    explanation = (
        generate_grounded_explanation(service, question, "", snippet)
        if service is not None
        else explain_from_evidence(question, snippet)
    )
    return {
        "answer": explanation,
        "confidence": best.get("score", 0.0),
        "model": "retrieval-fallback + grounded-explainer" if service is not None else "retrieval-fallback",
        "source_snippet": snippet,
    }


def clean_answer(answer: str, max_words: int = MAX_ANSWER_WORDS) -> str:
    answer = re.sub(r"\s+", " ", (answer or "")).strip(" \n\t-:;,.")
    if not answer:
        return ""
    tokens = answer.split()
    if len(tokens) > max_words:
        answer = " ".join(tokens[:max_words]).rstrip(",;:")
    return answer


def answer_is_usable(answer_bundle: Dict[str, object]) -> bool:
    answer = clean_answer(answer_bundle.get("answer_span") or answer_bundle.get("answer", ""))
    expanded = clean_answer(answer_bundle.get("answer", ""), max_words=120)
    confidence = float(answer_bundle.get("confidence", 0.0))
    if not answer:
        return False
    if not expanded:
        return False
    if len(word_tokenize(answer)) > MAX_ANSWER_WORDS:
        return False
    if confidence < MIN_CONFIDENCE:
        return False
    if answer.lower() in {"[cls]", "[sep]", "unknown", "none"}:
        return False
    return True


def is_abstract_question(question: str) -> bool:
    normalized = " ".join((question or "").lower().split())
    return normalized.startswith(ABSTRACT_QUESTION_PREFIXES)


def best_source_snippet(context: str, answer: str, question: str, max_sentences: int = 2) -> str:
    sentences = sentence_tokenize(context)
    if not sentences:
        return ""

    answer_lower = answer.lower().strip()
    if answer_lower:
        containing = [sentence for sentence in sentences if answer_lower in sentence.lower()]
        if containing:
            return " ".join(containing[:max_sentences]).strip()

    question_tokens = {
        token.lower()
        for token in word_tokenize(question)
        if len(token) > 2
    }
    ranked = []
    for sentence in sentences:
        sentence_tokens = {
            token.lower()
            for token in word_tokenize(sentence)
            if len(token) > 2
        }
        overlap = len(question_tokens & sentence_tokens)
        ranked.append((overlap, sentence))
    ranked.sort(key=lambda row: row[0], reverse=True)
    return " ".join(sentence for score, sentence in ranked[:max_sentences] if score > 0).strip()


def build_fallback_passage(
    question: str,
    retrieval_results: List[Dict[str, object]],
    min_sentences: int = MIN_ANSWER_SENTENCES,
    max_sentences: int = 6,
) -> str:
    sentence_pool: List[Dict[str, object]] = []
    references = query_references(question)

    for item in retrieval_results:
        sentence = (item.get("sentence") or "").strip()
        if sentence:
            sentence_pool.append(
                {
                    "sentence": sentence,
                    "score": item.get("score", 0.0),
                    "lexical_score": item.get("lexical_score", 0.0),
                    "reference_score": item.get("reference_score", 0.0),
                }
            )
            continue

        for sentence in sentence_tokenize(item.get("text", "")):
            cleaned = clean_answer(sentence, max_words=80)
            if cleaned:
                sentence_pool.append(
                    {
                        "sentence": cleaned,
                        "score": item.get("score", 0.0),
                        "lexical_score": 0.0,
                        "reference_score": item.get("reference_score", 0.0),
                    }
                )

    if not sentence_pool:
        return ""

    question_tokens = {
        token.lower()
        for token in word_tokenize(question)
        if len(token) > 2
    }
    ranked = []
    used = set()
    for item in sentence_pool:
        signature = item["sentence"].lower()
        if signature in used:
            continue
        used.add(signature)
        sentence_tokens = {
            token.lower()
            for token in word_tokenize(item["sentence"])
            if len(token) > 2
        }
        overlap = len(question_tokens & sentence_tokens)
        reference_hit = 1 if any(ref in item["sentence"].lower() for ref in references) else 0
        ranked.append(
            (
                reference_hit,
                item.get("reference_score", 0.0),
                overlap,
                item.get("lexical_score", 0.0),
                item.get("score", 0.0),
                item["sentence"],
            )
        )

    ranked.sort(reverse=True)
    selected = [sentence for _, _, _, _, _, sentence in ranked[:max_sentences]]

    if len(selected) < min_sentences:
        for _, _, _, _, _, sentence in ranked[max_sentences:]:
            selected.append(sentence)
            if len(selected) >= min_sentences:
                break

    return " ".join(selected[:max_sentences]).strip()


def expand_answer_context(
    context: str,
    answer: str,
    start: int | None,
    end: int | None,
    min_sentences: int = 3,
    max_sentences: int = 5,
) -> str:
    sentences = sentence_tokenize(context)
    if not sentences:
        return ""

    anchor_index = -1
    if answer:
        for index, sentence in enumerate(sentences):
            if answer.lower() in sentence.lower():
                anchor_index = index
                break

    if anchor_index == -1 and start is not None and end is not None:
        cursor = 0
        for index, sentence in enumerate(sentences):
            sentence_end = cursor + len(sentence)
            if cursor <= start <= sentence_end or cursor <= end <= sentence_end:
                anchor_index = index
                break
            cursor = sentence_end + 1

    if anchor_index == -1:
        return ""

    left = anchor_index
    right = anchor_index + 1
    selected = [sentences[anchor_index]]

    while len(selected) < max_sentences:
        expanded = False
        if left > 0:
            left -= 1
            selected.insert(0, sentences[left])
            expanded = True
        if len(selected) >= min_sentences and len(" ".join(selected).split()) >= MIN_EXPANDED_WORDS:
            break
        if right < len(sentences):
            selected.append(sentences[right])
            right += 1
            expanded = True
        if not expanded:
            break
        if len(selected) >= min_sentences and len(" ".join(selected).split()) >= MIN_EXPANDED_WORDS:
            break

    return clean_answer(" ".join(selected), max_words=160)


def format_multiline_answer(text: str, min_sentences: int = MIN_ANSWER_SENTENCES) -> str:
    sentences = [clean_answer(sentence, max_words=80) for sentence in sentence_tokenize(text) if clean_answer(sentence, max_words=80)]
    if not sentences:
        cleaned = clean_answer(text, max_words=160)
        return cleaned

    if len(sentences) < min_sentences:
        expanded: List[str] = []
        for sentence in sentences:
            parts = [clean_answer(part, max_words=40) for part in re.split(r"[;:]", sentence) if clean_answer(part, max_words=40)]
            if len(parts) > 1:
                expanded.extend(parts)
            else:
                expanded.append(sentence)
        sentences = expanded

    return "\n".join(sentences[:max(min_sentences, min(len(sentences), 6))])


def explain_from_evidence(question: str, text: str) -> str:
    sentences = [clean_answer(sentence, max_words=80) for sentence in sentence_tokenize(text) if clean_answer(sentence, max_words=80)]
    if not sentences:
        return clean_answer(text, max_words=160)

    references = query_references(question)
    topic = references[0].title() if references else ""

    explanation: List[str] = []
    if topic:
        explanation.append(f"{topic} is addressed in the uploaded dataset.")
    else:
        explanation.append("Based on the uploaded dataset, this is the relevant explanation.")

    explanation.append(sentences[0])

    for sentence in sentences[1:4]:
        if sentence.lower() not in {line.lower() for line in explanation}:
            explanation.append(sentence)

    if len(explanation) < 4 and len(sentences) == 1:
        fragments = [clean_answer(part, max_words=28) for part in re.split(r",|;|:", sentences[0]) if clean_answer(part, max_words=28)]
        for fragment in fragments[1:]:
            if fragment.lower() not in {line.lower() for line in explanation}:
                explanation.append(fragment)
            if len(explanation) >= 4:
                break

    if topic:
        explanation.append(f"This answer is based only on the retrieved dataset content for {topic}.")
    else:
        explanation.append("This answer is based only on the retrieved dataset content.")

    if len(explanation) < 4:
        explanation.append("The wording above is a grounded explanation of the retrieved evidence.")

    return "\n".join(explanation[:5])


def generate_grounded_explanation(
    service: QAModelService | None,
    question: str,
    answer_span: str,
    evidence: str,
) -> str:
    fallback = explain_from_evidence(question, evidence or answer_span)
    if service is None:
        return fallback

    cleaned_evidence = clean_answer(evidence, max_words=180)
    cleaned_span = clean_answer(answer_span, max_words=40)
    if not cleaned_evidence:
        return fallback

    prompt = (
        "Use only the evidence below.\n"
        "Answer the question in 4 to 6 short lines.\n"
        "Rewrite in clear natural language.\n"
        "Do not add outside facts.\n"
        "If the answer is not supported, say the uploaded dataset does not contain it.\n\n"
        f"Question: {question}\n"
        f"Answer span: {cleaned_span or 'Not found'}\n"
        f"Evidence: {cleaned_evidence}\n"
    )

    try:
        explainer = service.load_explainer()
        generated = explainer(
            prompt,
            max_new_tokens=140,
            do_sample=False,
        )
    except Exception:
        return fallback

    if not generated:
        return fallback

    text = clean_generated_explanation(generated[0].get("generated_text", ""))
    if not text:
        return fallback
    return format_explained_lines(text)


def clean_generated_explanation(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return ""
    return text


def format_explained_lines(text: str) -> str:
    sentences = [clean_answer(sentence, max_words=50) for sentence in sentence_tokenize(text) if clean_answer(sentence, max_words=50)]
    if not sentences:
        fragments = [clean_answer(part, max_words=40) for part in re.split(r"[;:]", text) if clean_answer(part, max_words=40)]
        sentences = fragments
    if not sentences:
        return clean_answer(text, max_words=160)
    if len(sentences) < 4:
        return explain_from_evidence("", " ".join(sentences))
    return "\n".join(sentences[:6])
