from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.logic_rag import LogicRAG
from src.utils import utils as logic_utils

app = FastAPI(
    title="LogicRAG Service",
    version="0.1.0",
    description="HTTP wrapper around chensyCN/LogicRAG for dependency-graph retrieval.",
)

_CACHE_DIR = Path("/tmp/logicrag_corpora")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_LOCK = threading.Lock()


@dataclass
class _CachedRAG:
    rag: LogicRAG
    context_to_doc: dict[str, str]
    lock: threading.Lock


_RAG_CACHE: dict[str, _CachedRAG] = {}


class CorpusDoc(BaseModel):
    doc_id: str
    title: str
    text: str


class LogicRAGAnswerRequest(BaseModel):
    question: str = Field(min_length=1)
    corpus: list[CorpusDoc]
    top_k: int = Field(default=5, ge=1, le=20)
    max_rounds: int = Field(default=4, ge=1, le=10)
    filter_repeats: bool = True


class LogicRAGAnswerResponse(BaseModel):
    answer: str
    contexts: list[str]
    retrieved_doc_ids: list[str]
    rounds: int
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


def _corpus_hash(corpus: list[CorpusDoc]) -> str:
    rows = [{"doc_id": row.doc_id, "title": row.title, "text": row.text} for row in corpus]
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _serialize_logicrag_doc(title: str, text: str) -> str:
    return f"Title: {title}. Content: {text}"


def _lookup_doc_id(context: str, context_to_doc: dict[str, str]) -> str | None:
    direct = context_to_doc.get(context)
    if direct:
        return direct

    # Fallback for minor whitespace/punctuation mismatches.
    normalized = " ".join(context.split())
    for key, value in context_to_doc.items():
        if normalized == " ".join(key.split()):
            return value
    return None


def _get_or_create_rag(request: LogicRAGAnswerRequest) -> _CachedRAG:
    if not request.corpus:
        raise ValueError("Corpus cannot be empty for LogicRAG")

    hash_value = _corpus_hash(request.corpus)
    cache_key = f"{hash_value}:{int(request.filter_repeats)}"

    with _LOCK:
        cached = _RAG_CACHE.get(cache_key)
        if cached is not None:
            return cached

        corpus_path = _CACHE_DIR / f"{hash_value}.json"
        if not corpus_path.exists():
            payload = [{"title": row.title, "text": row.text} for row in request.corpus]
            corpus_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        rag = LogicRAG(str(corpus_path), filter_repeats=request.filter_repeats)
        mapping = {
            _serialize_logicrag_doc(row.title, row.text): row.doc_id
            for row in request.corpus
        }
        entry = _CachedRAG(
            rag=rag,
            context_to_doc=mapping,
            lock=threading.Lock(),
        )
        _RAG_CACHE[cache_key] = entry
        return entry


@app.get("/api/logicrag/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/logicrag/answer", response_model=LogicRAGAnswerResponse)
def answer(request: LogicRAGAnswerRequest) -> LogicRAGAnswerResponse:
    try:
        cache_entry = _get_or_create_rag(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with cache_entry.lock:
        prompt_before = int(logic_utils.TOKEN_COST.get("prompt", 0))
        completion_before = int(logic_utils.TOKEN_COST.get("completion", 0))
        cache_entry.rag.set_top_k(request.top_k)
        cache_entry.rag.set_max_rounds(request.max_rounds)

        started = time.perf_counter()
        try:
            answer_text, contexts, rounds = cache_entry.rag.answer_question(request.question)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LogicRAG inference failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        prompt_after = int(logic_utils.TOKEN_COST.get("prompt", 0))
        completion_after = int(logic_utils.TOKEN_COST.get("completion", 0))
        prompt_tokens = max(0, prompt_after - prompt_before)
        completion_tokens = max(0, completion_after - completion_before)

    retrieved_doc_ids: list[str] = []
    for context in contexts:
        doc_id = _lookup_doc_id(context, cache_entry.context_to_doc)
        if doc_id and doc_id not in retrieved_doc_ids:
            retrieved_doc_ids.append(doc_id)

    return LogicRAGAnswerResponse(
        answer=answer_text.strip(),
        contexts=[str(item) for item in contexts if str(item).strip()],
        retrieved_doc_ids=retrieved_doc_ids,
        rounds=int(rounds),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
    )
