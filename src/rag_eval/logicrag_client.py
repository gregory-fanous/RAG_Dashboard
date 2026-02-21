from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from .models import Document


@dataclass(frozen=True)
class LogicRAGResult:
    answer: str
    contexts: list[str]
    retrieved_doc_ids: list[str]
    rounds: int
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class LogicRAGClient:
    def __init__(self, base_url: str, timeout_sec: float = 180.0) -> None:
        if not base_url:
            raise ValueError("LogicRAG base_url is required")
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = max(1.0, float(timeout_sec))

    def answer(
        self,
        *,
        question: str,
        documents: list[Document],
        top_k: int,
        max_rounds: int,
        filter_repeats: bool = True,
    ) -> LogicRAGResult:
        payload = {
            "question": question,
            "top_k": int(max(1, top_k)),
            "max_rounds": int(max(1, max_rounds)),
            "filter_repeats": bool(filter_repeats),
            "corpus": [
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "text": doc.text,
                }
                for doc in documents
            ],
        }

        endpoint = f"{self.base_url}/api/logicrag/answer"
        raw = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=raw,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LogicRAG HTTP error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LogicRAG connection failure: {exc.reason}") from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("LogicRAG returned invalid JSON") from exc

        return LogicRAGResult(
            answer=str(parsed.get("answer", "")).strip(),
            contexts=[str(item) for item in parsed.get("contexts", []) if str(item).strip()],
            retrieved_doc_ids=[str(item) for item in parsed.get("retrieved_doc_ids", []) if str(item).strip()],
            rounds=int(parsed.get("rounds", 0) or 0),
            prompt_tokens=int(parsed.get("prompt_tokens", 0) or 0),
            completion_tokens=int(parsed.get("completion_tokens", 0) or 0),
            latency_ms=float(parsed.get("latency_ms", 0.0) or 0.0),
        )
