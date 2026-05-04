import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.embeddings import EmbeddingService, chunk_embeddings
from backend.preprocessing import chunk_text, extract_text, preprocess_document
from backend.qa_model import QAModelService, answer_is_usable, fallback_answer, highlight_answer, is_abstract_question
from backend.retrieval import (
    VectorStore,
    build_context,
    build_sentence_context,
    exact_reference_matches,
    parse_specific_reference,
    query_references,
    rerank_results,
    rerank_sentences,
    sentence_candidates,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "uploaded"
VECTOR_DIR = BASE_DIR / "vector_db"
FRONTEND_DIR = BASE_DIR / "frontend"
ALLOWED_EXTENSIONS = {".pdf", ".csv", ".txt"}

app = FastAPI(title="Dynamic Domain-Trained QA System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_service = EmbeddingService()
qa_service = QAModelService()
SESSION_STATE: Dict[str, Dict[str, object]] = {}
SMALLTALK_INPUTS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
}
CONCEPT_EXPLANATIONS = {
    "article": (
        "An article is a numbered provision in a legal or constitutional document.\n"
        "Each article usually covers one subject, power, right, duty, or procedure.\n"
        "In the Constitution of India, articles are the main building blocks of the text.\n"
        "For example, one article may define a power, while another may describe a limitation or rule.\n"
        "If you want, ask for a specific one like 'What is Article 162?' or 'Explain Article 19.'"
    ),
    "section": (
        "A section is a numbered part of a statute or legal act.\n"
        "It is used to organize the law into clear individual provisions.\n"
        "Each section normally states a rule, offence, definition, power, or procedure.\n"
        "Different legal documents use different numbering styles, such as articles, sections, clauses, or schedules.\n"
        "Ask for a specific section if you want a direct answer from the uploaded dataset."
    ),
    "amendment": (
        "An amendment is a formal change made to an existing law or constitutional provision.\n"
        "It may insert new words, remove text, replace a provision, or change how a rule works.\n"
        "In constitutional documents, amendments are often identified by number, such as the Forty-Second Amendment.\n"
        "The effect of an amendment depends on the exact provision it changes and the wording used in the text.\n"
        "Ask for a specific amendment number if you want a dataset-based explanation."
    ),
}


class AskRequest(BaseModel):
    session_id: str
    question: str
    top_k: int = 5


def _ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)


def _session_payload(session_id: str) -> Dict[str, object]:
    payload = SESSION_STATE.get(session_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Session not found. Upload a dataset first.")
    return payload


def _save_upload(file: UploadFile, target_dir: Path) -> Path:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    target_path = target_dir / f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    with target_path.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    return target_path


def _prepare_session(files: List[UploadFile]) -> Dict[str, object]:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    _ensure_directories()
    session_id = uuid.uuid4().hex
    session_dir = DATA_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    source_files = []
    combined_chunks = []
    preprocessing_summary = []

    for uploaded_file in files:
        saved_path = _save_upload(uploaded_file, session_dir)
        try:
            raw_text = extract_text(saved_path)
        except (ImportError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not raw_text.strip():
            continue

        processed = preprocess_document(raw_text)
        chunks = chunk_text(processed["cleaned_text"], saved_path.name)
        combined_chunks.extend(chunks)
        source_files.append(
            {
                "filename": saved_path.name,
                "path": str(saved_path),
                "stats": processed["stats"],
            }
        )
        preprocessing_summary.append(
            {
                "filename": saved_path.name,
                "sentence_sample": processed["sentences"][:3],
                "token_sample": processed["tokens"][:20],
                "pos_sample": processed["pos_tags"][:20],
                "ner_sample": processed["named_entities"][:12],
                "coreference_sample": processed["coreference"][:5],
                "stats": processed["stats"],
            }
        )

    if not combined_chunks:
        raise HTTPException(status_code=400, detail="No readable text was extracted from the uploaded files.")

    try:
        embeddings = chunk_embeddings(combined_chunks, embedding_service)
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    index_path = VECTOR_DIR / f"{session_id}.index"
    metadata_path = VECTOR_DIR / f"{session_id}.json"
    store = VectorStore(index_path=index_path, metadata_path=metadata_path)
    try:
        store.save(embeddings, combined_chunks)
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    SESSION_STATE[session_id] = {
        "session_id": session_id,
        "source_files": source_files,
        "preprocessing": preprocessing_summary,
        "index_path": index_path,
        "metadata_path": metadata_path,
        "chat_history": [],
        "chunk_count": len(combined_chunks),
    }
    return SESSION_STATE[session_id]


def _is_smalltalk(question: str) -> bool:
    normalized = " ".join((question or "").lower().split())
    return normalized in SMALLTALK_INPUTS


def _smalltalk_answer() -> Dict[str, object]:
    return {
        "answer": (
            "Hello.\n"
            "Your dataset is loaded and ready.\n"
            "Ask me about articles, sections, amendments, legal terms, or topics from the uploaded files.\n"
            "I will answer from the dataset instead of using outside knowledge.\n"
            "Try a question like 'What is Article 420?' or 'Explain the Thirty-First Amendment.'"
        ),
        "confidence": 1.0,
        "model": "assistant-routing",
        "source_snippet": "",
    }


def _concept_answer(question: str) -> Dict[str, object] | None:
    normalized = " ".join((question or "").lower().split())
    if "article" in normalized and not query_references(question):
        return {
            "answer": CONCEPT_EXPLANATIONS["article"],
            "confidence": 0.95,
            "model": "assistant-explainer",
            "source_snippet": "",
        }
    if "section" in normalized and not query_references(question):
        return {
            "answer": CONCEPT_EXPLANATIONS["section"],
            "confidence": 0.95,
            "model": "assistant-explainer",
            "source_snippet": "",
        }
    if "amendment" in normalized and not query_references(question):
        return {
            "answer": CONCEPT_EXPLANATIONS["amendment"],
            "confidence": 0.95,
            "model": "assistant-explainer",
            "source_snippet": "",
        }
    return None


def _not_found_answer(reference: Dict[str, str]) -> Dict[str, object]:
    label = reference.get("kind", "reference").capitalize()
    value = reference.get("value", "")
    return {
        "answer": (
            f"{label} {value} was not found in the uploaded dataset.\n"
            "I searched the indexed content for an exact match before answering.\n"
            "The current dataset appears to contain different provisions or document parts.\n"
            "Please ask for another article, section, or amendment that exists in this uploaded file.\n"
            "If you expected this reference to exist, upload the document that contains it."
        ),
        "confidence": 0.0,
        "model": "dataset-exact-check",
        "source_snippet": "",
    }


def _merge_ranked_results(primary: List[Dict[str, object]], secondary: List[Dict[str, object]], limit: int) -> List[Dict[str, object]]:
    merged: List[Dict[str, object]] = []
    seen = set()
    for item in primary + secondary:
        chunk_id = item.get("chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        merged.append(item)
        if len(merged) >= limit:
            break
    return merged


def _answer_question(payload: AskRequest) -> Dict[str, object]:
    session = _session_payload(payload.session_id)
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if _is_smalltalk(payload.question):
        answer = _smalltalk_answer()
    else:
        answer = _concept_answer(payload.question)
    if _is_smalltalk(payload.question) or answer is not None:
        answer = answer or _smalltalk_answer()
        record = {
            "role": "user",
            "message": payload.question,
            "timestamp": datetime.utcnow().isoformat(),
        }
        session["chat_history"].append(record)
        session["chat_history"].append(
            {
                "role": "assistant",
                "message": answer["answer"],
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        return {
            "session_id": payload.session_id,
            "question": payload.question,
            "answer": answer["answer"],
            "confidence": answer["confidence"],
            "model": answer["model"],
            "source_snippet": "",
            "evidence_chunk": None,
            "highlighted_evidence": "",
            "retrieved_chunks": [],
            "chat_history": session["chat_history"],
        }

    store = VectorStore(session["index_path"], session["metadata_path"])
    try:
        metadata = store.load_metadata()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    specific_reference = parse_specific_reference(payload.question)
    reference_hits = exact_reference_matches(payload.question, metadata, limit=5)
    if specific_reference and not reference_hits:
        answer = _not_found_answer(specific_reference)
        record = {
            "role": "user",
            "message": payload.question,
            "timestamp": datetime.utcnow().isoformat(),
        }
        session["chat_history"].append(record)
        session["chat_history"].append(
            {
                "role": "assistant",
                "message": answer["answer"],
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        return {
            "session_id": payload.session_id,
            "question": payload.question,
            "answer": answer["answer"],
            "confidence": answer["confidence"],
            "model": answer["model"],
            "source_snippet": "",
            "evidence_chunk": None,
            "highlighted_evidence": "",
            "retrieved_chunks": [],
            "chat_history": session["chat_history"],
        }

    try:
        question_vector = embedding_service.encode([payload.question])
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    try:
        retrieved = store.search(question_vector, top_k=max(3, min(payload.top_k, 5)))
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    retrieved = rerank_results(payload.question, retrieved)
    if specific_reference and reference_hits:
        selected = reference_hits[:max(1, min(payload.top_k, 5))]
    else:
        selected = _merge_ranked_results(reference_hits, retrieved, limit=max(3, min(payload.top_k, 5)))
    chunk_context, selected_chunks = build_context(
        selected,
        max_chunks=max(3, min(payload.top_k, 5)),
        max_words=560,
        min_chunks=1 if query_references(payload.question) else 3,
    )
    candidates = sentence_candidates(selected)

    ranked_sentences = []
    if candidates:
        try:
            sentence_vectors = embedding_service.encode([item["sentence"] for item in candidates])
        except ImportError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        ranked_sentences = rerank_sentences(payload.question, question_vector, candidates, sentence_vectors)

    _, filtered_sentences = build_sentence_context(
        ranked_sentences,
        max_sentences=5,
        max_words=420,
    )

    if not selected_chunks or not chunk_context:
        answer = fallback_answer(payload.question, ranked_sentences or selected, qa_service)
        evidence = []
        highlighted = ""
    else:
        try:
            answer = qa_service.answer(payload.question, chunk_context)
            if not answer_is_usable(answer):
                answer = fallback_answer(payload.question, filtered_sentences or selected_chunks, qa_service)
        except Exception:
            answer = fallback_answer(payload.question, filtered_sentences or selected_chunks, qa_service)

        if is_abstract_question(payload.question) and filtered_sentences:
            abstract_best = answer.get("source_snippet") or filtered_sentences[0]["sentence"]
            if len(answer.get("answer", "").split()) < 10 and not query_references(payload.question):
                answer["answer"] = abstract_best
                answer["source_snippet"] = abstract_best

        evidence = filtered_sentences or sentence_candidates(selected_chunks)[:5]
        best_snippet = answer.get("source_snippet") or (evidence[0]["sentence"] if evidence else "")
        highlighted = highlight_answer(best_snippet, answer.get("answer_span") or answer["answer"])

    record = {
        "role": "user",
        "message": payload.question,
        "timestamp": datetime.utcnow().isoformat(),
    }
    session["chat_history"].append(record)
    session["chat_history"].append(
        {
            "role": "assistant",
            "message": answer["answer"],
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    return {
        "session_id": payload.session_id,
        "question": payload.question,
        "answer": answer["answer"],
        "confidence": answer.get("confidence", 0.0),
        "model": answer.get("model"),
        "source_snippet": answer.get("source_snippet"),
        "evidence_chunk": evidence[0] if evidence else None,
        "highlighted_evidence": highlighted,
        "retrieved_chunks": selected_chunks,
        "chat_history": session["chat_history"],
    }


@app.on_event("startup")
def startup() -> None:
    _ensure_directories()


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/upload")
async def upload_dataset(files: List[UploadFile] = File(...)) -> Dict[str, object]:
    try:
        session = _prepare_session(files)
    finally:
        for file in files:
            await file.close()
    return {
        "session_id": session["session_id"],
        "message": "Dataset processed successfully.",
        "source_files": session["source_files"],
        "preprocessing": session["preprocessing"],
        "chunk_count": session["chunk_count"],
    }


@app.post("/ask")
def ask_question(payload: AskRequest) -> Dict[str, object]:
    return _answer_question(payload)


@app.get("/session/{session_id}")
def get_session(session_id: str) -> Dict[str, object]:
    session = _session_payload(session_id)
    return {
        "session_id": session_id,
        "source_files": session["source_files"],
        "preprocessing": session["preprocessing"],
        "chunk_count": session["chunk_count"],
        "chat_history": session["chat_history"],
    }


app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
