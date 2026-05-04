import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from backend.preprocessing import sentence_tokenize, word_tokenize


MAX_CONTEXT_WORDS = 560
MIN_SENTENCE_WORDS = 4
MAX_SENTENCE_WORDS = 60
REFERENCE_RE = re.compile(r"\b(article|section)\s+(\d+[a-zA-Z\-]*)\b", re.IGNORECASE)
AMENDMENT_RE = re.compile(r"\b((?:\d+(?:st|nd|rd|th)?)|(?:[a-z]+(?:-[a-z]+)*))\s+amendment\b", re.IGNORECASE)
HEADING_RE = re.compile(r"(?:^|[\s(])(\d+[A-Z]?)\.\s+([A-Z][^.]{3,120})\.", re.IGNORECASE)


class VectorStore:
    def __init__(self, index_path: Path, metadata_path: Path):
        self.index_path = index_path
        self.metadata_path = metadata_path

    @staticmethod
    def _load_faiss():
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss-cpu is required for vector search. Install dependencies first.") from exc
        return faiss

    def save(self, embeddings: np.ndarray, metadata: List[Dict[str, object]]) -> None:
        faiss = self._load_faiss()
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        faiss.write_index(index, str(self.index_path))
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, object]]:
        faiss = self._load_faiss()
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("Vector index is missing for this session.")

        index = faiss.read_index(str(self.index_path))
        metadata = self.load_metadata()
        scores, indices = index.search(query_vector.astype("float32"), top_k)
        matches: List[Dict[str, object]] = []
        for score, index_id in zip(scores[0], indices[0]):
            if index_id < 0 or index_id >= len(metadata):
                continue
            item = metadata[index_id].copy()
            item["score"] = round(float(score), 4)
            matches.append(item)
        return matches

    def load_metadata(self) -> List[Dict[str, object]]:
        if not self.metadata_path.exists():
            raise FileNotFoundError("Vector metadata is missing for this session.")
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))


def _lexical_overlap(question: str, text: str) -> float:
    question_tokens = {
        token.lower()
        for token in word_tokenize(question)
        if len(token) > 2
    }
    text_tokens = {
        token.lower()
        for token in word_tokenize(text)
        if len(token) > 2
    }
    if not question_tokens:
        return 0.0
    return len(question_tokens & text_tokens) / len(question_tokens)


def _extract_references(text: str) -> List[str]:
    references = [f"{label.lower()} {value.lower()}" for label, value in REFERENCE_RE.findall(text or "")]
    references.extend(f"{value.lower()} amendment" for value in AMENDMENT_RE.findall(text or ""))
    return references


def _reference_match_score(question: str, text: str) -> float:
    question_refs = set(_extract_references(question))
    if not question_refs:
        return 0.0
    text_lower = (text or "").lower()
    hits = sum(1 for ref in question_refs if ref in text_lower)
    return hits / len(question_refs)


def query_references(question: str) -> List[str]:
    seen = set()
    ordered = []
    for ref in _extract_references(question):
        if ref not in seen:
            seen.add(ref)
            ordered.append(ref)
    return ordered


def parse_specific_reference(question: str) -> Dict[str, str]:
    match = REFERENCE_RE.search(question or "")
    if match:
        return {"kind": match.group(1).lower(), "value": match.group(2).lower()}

    amendment = AMENDMENT_RE.search(question or "")
    if amendment:
        return {"kind": "amendment", "value": amendment.group(1).lower()}

    return {}


def _heading_match_score(reference: Dict[str, str], text: str) -> float:
    if not reference:
        return 0.0

    text_lower = (text or "").lower()
    kind = reference.get("kind")
    value = reference.get("value", "")

    if kind in {"article", "section"}:
        heading_patterns = [
            rf"(?:^|[\s(]){re.escape(value)}\.\s+[A-Za-z]",
            rf"\b{kind}\s+{re.escape(value)}\.\s+[A-Za-z]",
            rf"\b{kind}\s+{re.escape(value)}\b\s*[:.\-]",
        ]
        score = 0.0
        for pattern in heading_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 2.0
        if re.search(rf"\b{re.escape(value)}\.\s+[A-Za-z]", text[:220], re.IGNORECASE):
            score += 2.0
        return score

    if kind == "amendment":
        normalized = value.replace("-", " ")
        compact = value.replace("-", "")
        score = 0.0
        if re.search(rf"\({re.escape(normalized)} amendment\)", text_lower):
            score += 2.5
        if f"{normalized} amendment" in text_lower or f"{compact} amendment" in text_lower:
            score += 1.5
        return score

    return 0.0


def exact_reference_matches(
    question: str,
    chunks: List[Dict[str, object]],
    limit: int = 5,
) -> List[Dict[str, object]]:
    references = query_references(question)
    specific_reference = parse_specific_reference(question)
    if not references:
        return []

    matches: List[Dict[str, object]] = []
    for item in chunks:
        text = item.get("text", "")
        text_lower = text.lower()
        score = 0.0
        heading_score = _heading_match_score(specific_reference, text)
        for ref in references:
            if ref in text_lower:
                score += 1.0
                if text_lower.startswith(ref):
                    score += 1.5
                elif ref in text_lower[:160]:
                    score += 0.75
        score += heading_score
        if score <= 0:
            continue
        updated = item.copy()
        updated["semantic_score"] = round(score, 4)
        updated["lexical_score"] = round(_lexical_overlap(question, text), 4)
        updated["heading_score"] = round(heading_score, 4)
        updated["reference_score"] = round(score / len(references), 4)
        updated["score"] = round((score * 0.7) + (_lexical_overlap(question, text) * 0.3), 4)
        matches.append(updated)

    matches.sort(
        key=lambda row: (
            row.get("heading_score", 0.0),
            row.get("reference_score", 0.0),
            row.get("score", 0.0),
            row.get("word_count", 0),
        ),
        reverse=True,
    )
    return matches[:limit]


def rerank_results(question: str, results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    reranked: List[Dict[str, object]] = []
    for item in results:
        lexical = _lexical_overlap(question, item["text"])
        reference_score = _reference_match_score(question, item["text"])
        sentence_hits = sum(
            1
            for sentence in sentence_tokenize(item["text"])
            if _lexical_overlap(question, sentence) > 0
        )
        combined = (item.get("score", 0.0) * 0.55) + (lexical * 0.2) + (reference_score * 0.25)
        updated = item.copy()
        updated["semantic_score"] = round(item.get("score", 0.0), 4)
        updated["lexical_score"] = round(lexical, 4)
        updated["reference_score"] = round(reference_score, 4)
        updated["score"] = round(combined, 4)
        updated["supporting_sentences"] = sentence_hits
        reranked.append(updated)
    reranked.sort(
        key=lambda row: (row["score"], row.get("reference_score", 0.0), row["supporting_sentences"], row["word_count"]),
        reverse=True,
    )
    return reranked


def clean_sentence_for_context(sentence: str) -> str:
    sentence = re.sub(r"\s+", " ", sentence).strip()
    sentence = re.sub(r"\b\d+\b", lambda match: match.group(0), sentence)
    return sentence


def sentence_candidates(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for item in results:
        for sentence in sentence_tokenize(item["text"]):
            cleaned = clean_sentence_for_context(sentence)
            word_count = len(word_tokenize(cleaned))
            if word_count < MIN_SENTENCE_WORDS or word_count > MAX_SENTENCE_WORDS:
                continue
            candidates.append(
                {
                    "sentence": cleaned,
                    "source_file": item["source_file"],
                    "chunk_id": item["chunk_id"],
                    "chunk_score": item.get("score", 0.0),
                    "semantic_score": item.get("semantic_score", item.get("score", 0.0)),
                    "lexical_score": item.get("lexical_score", 0.0),
                    "reference_score": item.get("reference_score", 0.0),
                }
            )
    return candidates


def rerank_sentences(
    question: str,
    question_vector: np.ndarray,
    candidates: List[Dict[str, object]],
    sentence_vectors: np.ndarray,
) -> List[Dict[str, object]]:
    reranked: List[Dict[str, object]] = []
    if len(candidates) != len(sentence_vectors):
        return reranked

    for candidate, vector in zip(candidates, sentence_vectors):
        embedding_score = float(np.dot(question_vector[0], vector))
        lexical_score = _lexical_overlap(question, candidate["sentence"])
        reference_score = _reference_match_score(question, candidate["sentence"])
        combined = (
            (embedding_score * 0.45)
            + (lexical_score * 0.2)
            + (candidate["chunk_score"] * 0.15)
            + (reference_score * 0.2)
        )
        updated = candidate.copy()
        updated["embedding_score"] = round(embedding_score, 4)
        updated["lexical_score"] = round(lexical_score, 4)
        updated["reference_score"] = round(reference_score, 4)
        updated["score"] = round(combined, 4)
        reranked.append(updated)

    reranked.sort(
        key=lambda row: (row["score"], row.get("reference_score", 0.0), row["lexical_score"], row["embedding_score"]),
        reverse=True,
    )
    return reranked


def build_sentence_context(
    sentences: List[Dict[str, object]],
    max_sentences: int = 5,
    max_words: int = MAX_CONTEXT_WORDS,
) -> Tuple[str, List[Dict[str, object]]]:
    selected: List[Dict[str, object]] = []
    used = set()
    total_words = 0

    for item in sentences:
        signature = item["sentence"].lower()
        if signature in used:
            continue
        sentence_word_count = len(word_tokenize(item["sentence"]))
        if selected and total_words + sentence_word_count > max_words:
            continue
        used.add(signature)
        selected.append(item)
        total_words += sentence_word_count
        if len(selected) >= max_sentences:
            break

    context = " ".join(item["sentence"] for item in selected).strip()
    return context, selected


def build_context(
    results: List[Dict[str, object]],
    max_chunks: int = 5,
    max_words: int = MAX_CONTEXT_WORDS,
    min_chunks: int = 3,
) -> Tuple[str, List[Dict[str, object]]]:
    selected: List[Dict[str, object]] = []
    total_words = 0

    for item in results[:max(1, max_chunks)]:
        chunk_words = len(word_tokenize(item["text"]))
        if selected and len(selected) >= min_chunks and total_words + chunk_words > max_words:
            continue
        selected.append(item)
        total_words += chunk_words
        if len(selected) >= max_chunks:
            break

    if not selected:
        return "", []

    context = "\n\n".join(item["text"] for item in selected).strip()
    return context, selected
