from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimeSettings:
    execution_mode: str
    allow_synthetic_mode: bool
    openai_api_key: str
    openai_base_url: str
    openai_chat_model: str
    openai_embedding_model: str
    openai_verifier_model: str
    request_timeout_sec: float
    max_generation_tokens: int
    max_context_chars: int
    embedding_batch_size: int
    top_candidate_multiplier: int
    max_rerank_candidates: int
    max_verifier_claims: int
    logicrag_service_url: str = ""
    logicrag_timeout_sec: float = 180.0
    logicrag_max_rounds: int = 4
    logicrag_filter_repeats: bool = True

    @classmethod
    def from_env(cls, load_dotenv_file: bool = True) -> "RuntimeSettings":
        if load_dotenv_file:
            try:
                from dotenv import load_dotenv

                root = Path(__file__).resolve().parents[2]
                load_dotenv(root / ".env", override=False)
            except Exception:
                pass

        return cls(
            execution_mode=os.getenv("RAG_EVAL_EXECUTION_MODE", "real").strip().lower(),
            allow_synthetic_mode=_env_bool("RAG_ALLOW_SYNTHETIC_MODE", False),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini").strip(),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large").strip(),
            openai_verifier_model=os.getenv("OPENAI_VERIFIER_MODEL", "gpt-4.1-mini").strip(),
            request_timeout_sec=float(os.getenv("RAG_REQUEST_TIMEOUT_SEC", "120")),
            max_generation_tokens=int(os.getenv("RAG_MAX_GENERATION_TOKENS", "700")),
            max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "1400")),
            embedding_batch_size=int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "48")),
            top_candidate_multiplier=int(os.getenv("RAG_TOP_CANDIDATE_MULTIPLIER", "5")),
            max_rerank_candidates=int(os.getenv("RAG_MAX_RERANK_CANDIDATES", "20")),
            max_verifier_claims=int(os.getenv("RAG_MAX_VERIFIER_CLAIMS", "8")),
            logicrag_service_url=os.getenv("LOGICRAG_SERVICE_URL", "").strip(),
            logicrag_timeout_sec=float(os.getenv("LOGICRAG_TIMEOUT_SEC", "180")),
            logicrag_max_rounds=int(os.getenv("LOGICRAG_MAX_ROUNDS", "4")),
            logicrag_filter_repeats=_env_bool("LOGICRAG_FILTER_REPEATS", True),
        )

    def validate_for_real_mode(self) -> None:
        missing: list[str] = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if self.openai_api_key.upper().startswith("YOUR_") or "PLACEHOLDER" in self.openai_api_key.upper():
            missing.append("OPENAI_API_KEY(real value)")
        if not self.openai_chat_model:
            missing.append("OPENAI_CHAT_MODEL")
        if not self.openai_embedding_model:
            missing.append("OPENAI_EMBEDDING_MODEL")
        if not self.openai_verifier_model:
            missing.append("OPENAI_VERIFIER_MODEL")

        if missing:
            raise ValueError(
                "Missing required environment fields for real execution mode: "
                + ", ".join(missing)
                + ". Populate .env before running."
            )

    def validate_for_synthetic_mode(self) -> None:
        if not self.allow_synthetic_mode:
            raise ValueError(
                "Synthetic execution mode is disabled. Set RAG_ALLOW_SYNTHETIC_MODE=true "
                "only for CI/testing."
            )
