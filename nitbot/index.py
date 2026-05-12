import json
import re
from pathlib import Path

import faiss
import numpy as np
from .model import embeddings, generate_chat_response
from .sharedict import session_state


BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_DIR = BASE_DIR / "faiss_index"
FAISS_INDEX_FILE = FAISS_DIR / "index.faiss"
FAISS_CHUNKS_FILE = FAISS_DIR / "chunks.json"
TOP_K = 10
MAX_HISTORY_TURNS = 12


def log(message: str):
    print(f"[faiss_bot] {message}", flush=True)


def contains_explicit_professor_reference(text: str) -> bool:
    patterns = [
        r"\bProf\.?\s+[A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){0,5}\b",
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def is_professor_or_institute_query(text: str) -> bool:
    lowered = text.lower()
    keywords = [
        "prof",
        "professor",
        "faculty",
        "department",
        "nitw",
        "nit warangal",
        "course",
        "research",
        "office",
        "room",
        "email",
        "phone",
        "contact",
        "publication",
        "advisor",
        "appointment",
        "student research",
        "lab",
        "internship",
    ]
    return any(keyword in lowered for keyword in keywords) or contains_explicit_professor_reference(text)


def should_use_last_professor(text: str) -> bool:
    lowered = text.lower().strip()
    follow_up_signals = [
        "detail",
        "details",
        "personal",
        "profile",
        "more",
        "more info",
        "information",
        "contact",
        "email",
        "phone",
        "room",
        "office",
        "about him",
        "about her",
        "about them",
        "about it",
        "his",
        "her",
        "their",
        "its",
    ]
    short_follow_up = len(lowered.split()) <= 8
    return short_follow_up or any(signal in lowered for signal in follow_up_signals)


def build_retrieval_query(user_input: str) -> str:
    last_professor = session_state.get("last_professor")
    if not last_professor:
        return user_input

    if contains_explicit_professor_reference(user_input):
        return user_input

    if should_use_last_professor(user_input):
        return f"{user_input} about {last_professor}"

    return user_input


def load_faiss_index():
    if not FAISS_INDEX_FILE.exists() or not FAISS_CHUNKS_FILE.exists():
        return None, []

    index = faiss.read_index(str(FAISS_INDEX_FILE))
    chunks = json.loads(FAISS_CHUNKS_FILE.read_text(encoding="utf-8"))
    return index, chunks


def initialize_document_search():
    index, chunks = load_faiss_index()
    if index is not None and chunks:
        session_state["document_index"] = index
        session_state["document_chunks"] = chunks
        log(f"Loaded FAISS index with {len(chunks)} chunks")
        return

    session_state["document_index"] = None
    session_state["document_chunks"] = []
    log("FAISS index not found. Run build_professor_faiss.py first.")


def similarity_search(query: str, k: int = TOP_K):
    index = session_state.get("document_index")
    chunks = session_state.get("document_chunks", [])

    if index is None or not chunks:
        return []

    query_vector = np.array([embeddings.embed_query(query)], dtype="float32")
    search_k = min(k, len(chunks))
    distances, indices = index.search(query_vector, search_k)

    results = []
    for rank, idx in enumerate(indices[0], start=1):
        if 0 <= idx < len(chunks):
            chunk = chunks[idx]
            results.append(
                {
                    "rank": rank,
                    "distance": float(distances[0][rank - 1]),
                    "professor": chunk.get("professor", "Unknown Professor"),
                    "part": chunk.get("part", 1),
                    "text": chunk["text"],
                }
            )
    return results


def format_context_chunks(results):
    if not results:
        return []

    formatted = []
    for result in results:
        formatted.append(
            "\n".join(
                [
                    f"Result {result['rank']}",
                    f"Professor: {result['professor']}",
                    f"Chunk part: {result['part']}",
                    f"Distance: {result['distance']:.4f}",
                    result["text"],
                ]
            )
        )
    return formatted


def chat_input(user_input):
    if "document_index" not in session_state:
        initialize_document_search()

    history = session_state.setdefault("chat_history", [])
    retrieval_query = build_retrieval_query(user_input)
    should_use_retrieval = (
        session_state.get("document_index") is not None
        and (
            is_professor_or_institute_query(user_input)
            or (
                session_state.get("last_professor")
                and should_use_last_professor(user_input)
            )
        )
    )
    retrieval_results = similarity_search(retrieval_query) if should_use_retrieval else []
    context_chunks = format_context_chunks(retrieval_results)
    response = generate_chat_response(user_input, context_chunks, history)

    if retrieval_results:
        session_state["last_professor"] = retrieval_results[0]["professor"]

    history.append({"user": user_input, "assistant": response})
    if len(history) > MAX_HISTORY_TURNS:
        del history[:-MAX_HISTORY_TURNS]

    return response
