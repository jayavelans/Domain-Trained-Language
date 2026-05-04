import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will",
    "with", "this", "these", "those", "or", "if", "but", "not", "into", "about",
    "than", "then", "them", "they", "their", "there", "here", "what", "which", "who",
    "whom", "when", "where", "why", "how", "you", "your", "we", "our", "i", "me", "my",
}
DETERMINERS = {"a", "an", "the", "this", "that", "these", "those"}
PREPOSITIONS = {"of", "in", "on", "for", "to", "by", "with", "from", "under", "over", "into"}
CONJUNCTIONS = {"and", "or", "but", "if", "while", "because"}
PRONOUNS = {"he", "she", "it", "they", "him", "her", "them", "his", "their", "its", "we", "you", "i"}
COMMON_VERBS = {
    "be", "am", "is", "are", "was", "were", "has", "have", "had", "do", "does",
    "did", "say", "says", "said", "go", "goes", "went", "make", "made", "use",
    "used", "provide", "provided", "include", "included",
}
TITLE_HINTS = {"mr", "mrs", "ms", "dr", "judge", "justice", "president", "minister", "professor"}
ORG_SUFFIXES = {"inc", "ltd", "llc", "corp", "company", "court", "university", "committee", "agency"}
LEGAL_TERMS = {"law", "court", "judge", "justice", "act", "section", "article", "case", "appeal", "order"}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[^\w\s.!?,'\"()/:;-]", " ", text)
    return normalize_whitespace(text)


def sentence_tokenize(text: str) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    return [sentence.strip() for sentence in SENTENCE_RE.split(text) if sentence.strip()]


def word_tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text)


def normalize_tokens(tokens: List[str]) -> List[str]:
    return [token.lower() for token in tokens]


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [token for token in tokens if token.lower() not in STOPWORDS]


def pos_tag(tokens: List[str]) -> List[Tuple[str, str]]:
    tagged = []
    for token in tokens:
        lower = token.lower()
        if lower in DETERMINERS:
            tag = "DT"
        elif lower in PREPOSITIONS:
            tag = "IN"
        elif lower in CONJUNCTIONS:
            tag = "CC"
        elif lower in PRONOUNS:
            tag = "PRP"
        elif lower.isdigit():
            tag = "CD"
        elif lower in COMMON_VERBS or lower.endswith(("ing", "ed")):
            tag = "VB"
        elif lower.endswith(("al", "ive", "ous", "ful", "able", "ible")):
            tag = "JJ"
        elif token[:1].isupper():
            tag = "NNP"
        else:
            tag = "NN"
        tagged.append((token, tag))
    return tagged


def named_entities(tokens: List[str]) -> List[Dict[str, str]]:
    entities: List[Dict[str, str]] = []
    for index, token in enumerate(tokens):
        lower = token.lower().strip(".")
        if lower in TITLE_HINTS and index + 1 < len(tokens):
            entities.append({"text": f"{token} {tokens[index + 1]}", "label": "PERSON"})
        elif token[:1].isupper() and len(token) > 2:
            entities.append({"text": token, "label": "PROPER_NOUN"})
        elif lower in ORG_SUFFIXES:
            entities.append({"text": token, "label": "ORG_HINT"})
        elif lower in LEGAL_TERMS:
            entities.append({"text": token, "label": "DOMAIN_TERM"})
    deduped = []
    seen = set()
    for entity in entities:
        key = (entity["text"], entity["label"])
        if key not in seen:
            seen.add(key)
            deduped.append(entity)
    return deduped


def coreference_summary(sentences: List[str]) -> List[Dict[str, str]]:
    if not sentences:
        return []
    mentions: List[Dict[str, str]] = []
    recent_nouns: List[str] = []
    for sentence in sentences[:12]:
        tokens = word_tokenize(sentence)
        tags = pos_tag(tokens)
        nouns = [word for word, tag in tags if tag in {"NN", "NNP"} and word.lower() not in PRONOUNS]
        pronouns = [word for word, tag in tags if tag == "PRP"]
        for pronoun in pronouns:
            mentions.append(
                {
                    "sentence": sentence,
                    "pronoun": pronoun,
                    "resolved_to": recent_nouns[-1] if recent_nouns else "Unknown",
                }
            )
        recent_nouns.extend(nouns[:2])
    return mentions[:10]


def chunk_text(
    text: str,
    source_name: str,
    chunk_size: int = 260,
    overlap: int = 40,
) -> List[Dict[str, object]]:
    sentences = sentence_tokenize(text)
    chunks: List[Dict[str, object]] = []
    current_sentences: List[str] = []
    current_words = 0
    chunk_id = 0

    for sentence in sentences:
        sentence_words = word_tokenize(sentence)
        if not sentence_words:
            continue
        if current_sentences and current_words + len(sentence_words) > chunk_size:
            chunk_text_value = " ".join(current_sentences).strip()
            chunk_tokens = word_tokenize(chunk_text_value)
            chunks.append(
                {
                    "chunk_id": f"{source_name}-{chunk_id}",
                    "source_file": source_name,
                    "text": chunk_text_value,
                    "word_count": len(chunk_tokens),
                    "sentences": current_sentences.copy(),
                }
            )
            chunk_id += 1
            overlap_sentences: List[str] = []
            overlap_count = 0
            for old_sentence in reversed(current_sentences):
                overlap_sentences.insert(0, old_sentence)
                overlap_count += len(word_tokenize(old_sentence))
                if overlap_count >= overlap:
                    break
            current_sentences = overlap_sentences
            current_words = sum(len(word_tokenize(item)) for item in current_sentences)

        current_sentences.append(sentence)
        current_words += len(sentence_words)

    if current_sentences:
        chunk_text_value = " ".join(current_sentences).strip()
        chunk_tokens = word_tokenize(chunk_text_value)
        chunks.append(
            {
                "chunk_id": f"{source_name}-{chunk_id}",
                "source_file": source_name,
                "text": chunk_text_value,
                "word_count": len(chunk_tokens),
                "sentences": current_sentences.copy(),
            }
        )

    return chunks


def extract_text_from_pdf(path: Path) -> str:
    try:
        from PyPDF2 import PdfReader
    except ImportError as exc:
        raise ImportError("PyPDF2 is required for PDF uploads. Install dependencies first.") from exc

    with path.open("rb") as file:
        reader = PdfReader(file)
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = " ".join(page for page in pages if page)
    return clean_text(text)


def extract_text_from_csv(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = []
        text_candidates = {"text", "content", "body", "sentence", "article", "description"}
        for row in reader:
            preferred = [row.get(column, "") for column in reader.fieldnames or [] if column.lower() in text_candidates]
            if preferred:
                rows.extend(item.strip() for item in preferred if item and item.strip())
            else:
                rows.extend(str(value).strip() for value in row.values() if value and str(value).strip())
    return clean_text(" ".join(rows))


def extract_text_from_txt(path: Path) -> str:
    return clean_text(path.read_text(encoding="utf-8", errors="ignore"))


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix == ".csv":
        return extract_text_from_csv(path)
    if suffix == ".txt":
        return extract_text_from_txt(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def preprocess_document(text: str) -> Dict[str, object]:
    cleaned = clean_text(text)
    sentences = sentence_tokenize(cleaned)
    tokens = word_tokenize(cleaned)
    normalized_tokens = normalize_tokens(tokens)
    filtered_tokens = remove_stopwords(normalized_tokens)
    tagged = pos_tag(tokens[:150])
    entities = named_entities(tokens[:150])
    coref = coreference_summary(sentences)
    return {
        "raw_text": text,
        "cleaned_text": cleaned,
        "sentences": sentences,
        "tokens": tokens,
        "normalized_tokens": normalized_tokens,
        "filtered_tokens": filtered_tokens,
        "pos_tags": tagged,
        "named_entities": entities,
        "coreference": coref,
        "stats": {
            "characters": len(text),
            "sentences": len(sentences),
            "tokens": len(tokens),
            "filtered_tokens": len(filtered_tokens),
        },
    }
