from __future__ import annotations

import hashlib
import math
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .logicrag_client import LogicRAGClient
from .models import CaseRun, ClaimVerification, EvaluationCase, EvaluationDataset, StrategyConfig
from .providers import JsonParseError, LLMCallResult, OpenAIProvider, extract_json
from .runtime import RuntimeSettings

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
ENTITY_PATTERN = re.compile(r"\b[A-Z][a-zA-Z0-9]{2,}\b")
CITATION_PATTERN = re.compile(r"\[[^\]]+\]")


@dataclass
class _Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    tags: list[str]
    effective_date: datetime | None
    entities: set[str]
    embedding: list[float] = field(default_factory=list)


@dataclass
class _StrategyIndex:
    chunks: list[_Chunk]
    entity_to_indices: dict[str, set[int]]


@dataclass
class _ZvecIndex:
    collection: Any
    chunk_id_to_index: dict[str, int]


@dataclass
class _UsageCounter:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add_call(self, call: LLMCallResult) -> None:
        self.prompt_tokens += call.prompt_tokens
        self.completion_tokens += call.completion_tokens

    def add_embedding_tokens(self, token_count: int) -> None:
        self.prompt_tokens += max(0, int(token_count))

    def add_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += max(0, int(prompt_tokens))
        self.completion_tokens += max(0, int(completion_tokens))


@dataclass
class RealRAGSystem:
    dataset: EvaluationDataset
    provider: OpenAIProvider
    settings: RuntimeSettings
    logicrag_client: LogicRAGClient | None = None
    seed: int = 42

    _index_cache: dict[str, _StrategyIndex] = field(default_factory=dict, init=False)
    _embedding_cache: dict[str, list[float]] = field(default_factory=dict, init=False)
    _zvec_index_cache: dict[str, _ZvecIndex] = field(default_factory=dict, init=False)

    def run_case(self, case: EvaluationCase, strategy: StrategyConfig) -> CaseRun:
        usage = _UsageCounter()
        retrieval_start = time.perf_counter()

        if strategy.logicrag:
            if self.logicrag_client is None:
                raise ValueError(
                    "LogicRAG was requested but LOGICRAG_SERVICE_URL is not configured."
                )
            logicrag_query = case.query
            if strategy.query_rewrite:
                rewritten = self._rewrite_query(query=case.query, strategy=strategy, usage=usage)
                if rewritten:
                    logicrag_query = rewritten
            if strategy.hyde:
                hypothetical = self._hyde_document(query=logicrag_query, strategy=strategy, usage=usage)
                if hypothetical:
                    logicrag_query = f"{logicrag_query}\n\nHypothesis:\n{hypothetical}"
            logicrag_result = self.logicrag_client.answer(
                question=logicrag_query,
                documents=self.dataset.documents,
                top_k=max(strategy.retrieval_k, 1),
                max_rounds=max(1, self.settings.logicrag_max_rounds),
                filter_repeats=self.settings.logicrag_filter_repeats,
            )
            usage.add_tokens(
                prompt_tokens=logicrag_result.prompt_tokens,
                completion_tokens=logicrag_result.completion_tokens,
            )
            top_chunks = self._logicrag_to_chunks(
                contexts=logicrag_result.contexts,
                retrieved_doc_ids=logicrag_result.retrieved_doc_ids,
            )
            if strategy.temporal_recency_filter:
                top_chunks = self._sort_chunks_by_recency(top_chunks)
            answer = logicrag_result.answer
            retrieval_latency_ms = max(
                float(logicrag_result.latency_ms),
                (time.perf_counter() - retrieval_start) * 1000.0,
            )
        else:
            index = self._index_for_strategy(strategy=strategy, usage=usage)

            query_for_retrieval = case.query
            if strategy.query_rewrite:
                rewritten = self._rewrite_query(query=case.query, strategy=strategy, usage=usage)
                if rewritten:
                    query_for_retrieval = rewritten

            if strategy.hyde:
                hypothetical = self._hyde_document(query=query_for_retrieval, strategy=strategy, usage=usage)
                if hypothetical:
                    query_for_retrieval = f"{query_for_retrieval}\n\nHypothesis:\n{hypothetical}"

            initial_scored = self._retrieve(query=query_for_retrieval, index=index, strategy=strategy)

            scored = dict(initial_scored)
            if strategy.recursive_retrieval:
                followup_query = self._followup_query(
                    query=case.query,
                    scored_chunks=initial_scored,
                    index=index,
                    usage=usage,
                )
                if followup_query:
                    for idx, score in self._retrieve(query=followup_query, index=index, strategy=strategy):
                        scored[idx] = max(scored.get(idx, -1.0), score)

            if strategy.graph_augmentation:
                scored = self._apply_graph_augmentation(scored=scored, index=index)

            if strategy.temporal_recency_filter:
                scored = self._apply_temporal_boost(scored=scored, index=index)

            ranked_indices = sorted(scored.keys(), key=lambda idx: scored[idx], reverse=True)
            if strategy.cross_encoder_rerank:
                ranked_indices = self._cross_encoder_rerank(
                    query=case.query,
                    ranked_indices=ranked_indices,
                    index=index,
                    strategy=strategy,
                    usage=usage,
                )

            top_chunks = [index.chunks[idx] for idx in ranked_indices[: max(strategy.retrieval_k, 1)]]
            retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

            generation_start = time.perf_counter()
            answer_call = self._generate_answer(
                query=case.query,
                contexts=top_chunks,
                strategy=strategy,
            )
            usage.add_call(answer_call)
            answer = answer_call.text
            generation_latency_ms = (time.perf_counter() - generation_start) * 1000.0

        contexts = [chunk.text for chunk in top_chunks]
        retrieved_doc_ids: list[str] = []
        for chunk in top_chunks:
            if chunk.doc_id not in retrieved_doc_ids:
                retrieved_doc_ids.append(chunk.doc_id)
            if len(retrieved_doc_ids) >= strategy.retrieval_k:
                break
        citation_ids = [chunk.chunk_id for chunk in top_chunks]

        verification_start = time.perf_counter()
        if strategy.self_rag:
            critique = self._self_rag_reflection(
                query=case.query,
                answer=answer,
                contexts=top_chunks,
                strategy=strategy,
                usage=usage,
            )
            if critique:
                revised_call = self._generate_answer(
                    query=case.query,
                    contexts=top_chunks,
                    strategy=strategy,
                    critique=critique,
                )
                usage.add_call(revised_call)
                answer = revised_call.text

        claim_verifications: list[ClaimVerification] = []
        if strategy.claim_verification:
            claim_verifications = self._verify_claims(
                query=case.query,
                answer=answer,
                contexts=top_chunks,
                strategy=strategy,
                usage=usage,
            )
            answer = self._filter_to_supported_claims(
                answer=answer,
                verifications=claim_verifications,
                citation_ids=citation_ids,
                enforce_citations=strategy.citation_enforcement,
            )
        elif strategy.citation_enforcement:
            answer = self._ensure_citations(answer=answer, citation_ids=citation_ids)

        verification_latency_ms = (time.perf_counter() - verification_start) * 1000.0
        if strategy.logicrag:
            generation_latency_ms = verification_latency_ms
        else:
            generation_latency_ms += verification_latency_ms

        return CaseRun(
            query_id=case.query_id,
            retrieved_doc_ids=retrieved_doc_ids,
            contexts=contexts,
            answer=answer,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            claim_verifications=claim_verifications,
            citation_ids=citation_ids,
            execution_mode="real",
        )

    def _rng_for(self, query_id: str, strategy_name: str) -> random.Random:
        payload = f"{self.seed}:{query_id}:{strategy_name}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return random.Random(int(digest[:16], 16))

    def _index_for_strategy(self, strategy: StrategyConfig, usage: _UsageCounter) -> _StrategyIndex:
        backend = self._normalized_backend(strategy.retrieval_backend)
        cache_key = f"{strategy.chunking_strategy}:{strategy.temporal_recency_filter}:{backend}"
        cached = self._index_cache.get(cache_key)
        if cached is not None:
            return cached

        chunks: list[_Chunk] = []
        for doc in self.dataset.documents:
            doc_chunks = self._chunk_document(doc_id=doc.doc_id, title=doc.title, text=doc.text, tags=doc.tags, strategy=strategy)
            chunks.extend(doc_chunks)

        texts = [chunk.text for chunk in chunks]
        embeddings, embed_tokens = self._embed_texts(texts=texts)
        usage.add_embedding_tokens(embed_tokens)
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        entity_to_indices: dict[str, set[int]] = {}
        for idx, chunk in enumerate(chunks):
            for entity in chunk.entities:
                entity_to_indices.setdefault(entity, set()).add(idx)

        built = _StrategyIndex(chunks=chunks, entity_to_indices=entity_to_indices)
        if backend == "zvec":
            self._build_zvec_index(cache_key=cache_key, index=built)
        self._index_cache[cache_key] = built
        return built

    def _chunk_document(self, doc_id: str, title: str, text: str, tags: list[str], strategy: StrategyConfig) -> list[_Chunk]:
        chunks: list[str]
        if strategy.chunking_strategy == "fixed":
            chunks = self._fixed_chunks(text)
        elif strategy.chunking_strategy == "sentence_window":
            chunks = self._sentence_window_chunks(text)
        elif strategy.chunking_strategy == "semantic":
            chunks = self._semantic_chunks(text)
        elif strategy.chunking_strategy == "adaptive":
            chunks = self._adaptive_chunks(text)
        else:
            chunks = self._fixed_chunks(text)

        effective_date = self._extract_effective_date(tags)
        out: list[_Chunk] = []
        for idx, chunk_text in enumerate(chunks):
            clean = chunk_text.strip()
            if not clean:
                continue
            chunk_id = f"{doc_id}#c{idx}"
            entities = {item.lower() for item in ENTITY_PATTERN.findall(clean)}
            out.append(
                _Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    doc_title=title,
                    text=clean[: self.settings.max_context_chars],
                    tags=tags,
                    effective_date=effective_date,
                    entities=entities,
                )
            )
        return out

    def _logicrag_to_chunks(self, contexts: list[str], retrieved_doc_ids: list[str]) -> list[_Chunk]:
        doc_lookup = {doc.doc_id: doc for doc in self.dataset.documents}
        chunks: list[_Chunk] = []

        for idx, context in enumerate(contexts[: max(1, len(contexts))]):
            clean = str(context).strip()
            if not clean:
                continue

            doc_id = (
                retrieved_doc_ids[idx]
                if idx < len(retrieved_doc_ids) and retrieved_doc_ids[idx]
                else f"logicrag_doc_{idx:04d}"
            )
            doc = doc_lookup.get(doc_id)
            title = doc.title if doc else doc_id
            tags = doc.tags if doc else []
            entities = {item.lower() for item in ENTITY_PATTERN.findall(clean)}

            chunks.append(
                _Chunk(
                    chunk_id=f"{doc_id}#logic{idx}",
                    doc_id=doc_id,
                    doc_title=title,
                    text=clean[: self.settings.max_context_chars],
                    tags=tags,
                    effective_date=self._extract_effective_date(tags),
                    entities=entities,
                )
            )

        return chunks

    def _sort_chunks_by_recency(self, chunks: list[_Chunk]) -> list[_Chunk]:
        def _score(chunk: _Chunk) -> float:
            value = chunk.effective_date
            if value is None:
                return float("-inf")
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.timestamp()

        return sorted(chunks, key=_score, reverse=True)

    def _fixed_chunks(self, text: str, size: int = 220, overlap: int = 40) -> list[str]:
        words = text.split()
        if not words:
            return [""]

        chunks: list[str] = []
        step = max(1, size - overlap)
        for start in range(0, len(words), step):
            segment = words[start : start + size]
            if segment:
                chunks.append(" ".join(segment))
        return chunks

    def _sentence_window_chunks(self, text: str, window: int = 3, stride: int = 2) -> list[str]:
        sentences = [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]
        if not sentences:
            return self._fixed_chunks(text)

        chunks: list[str] = []
        for i in range(0, len(sentences), stride):
            segment = sentences[i : i + window]
            if segment:
                chunks.append(" ".join(segment))
        return chunks

    def _semantic_chunks(self, text: str, max_words: int = 190) -> list[str]:
        sentences = [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]
        if not sentences:
            return self._fixed_chunks(text)

        chunks: list[str] = []
        current: list[str] = []
        current_words = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())
            starts_new_section = sentence.startswith("#") or sentence.startswith("##")
            if current and (current_words + sentence_words > max_words or starts_new_section):
                chunks.append(" ".join(current))
                current = []
                current_words = 0

            current.append(sentence)
            current_words += sentence_words

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _adaptive_chunks(self, text: str) -> list[str]:
        word_count = len(text.split())
        if word_count > 1300:
            return self._semantic_chunks(text, max_words=220)
        if word_count > 450:
            return self._sentence_window_chunks(text, window=4, stride=2)
        return self._fixed_chunks(text, size=170, overlap=30)

    def _embed_texts(self, texts: list[str]) -> tuple[list[list[float]], int]:
        embeddings: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for idx, text in enumerate(texts):
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            cached = self._embedding_cache.get(key)
            if cached is None:
                uncached_indices.append(idx)
                uncached_texts.append(text)
            else:
                embeddings[idx] = cached

        token_count = 0
        if uncached_texts:
            step = max(1, self.settings.embedding_batch_size)
            for start in range(0, len(uncached_texts), step):
                batch = uncached_texts[start : start + step]
                vectors, tokens = self.provider.embeddings(
                    texts=batch,
                    model=self.settings.openai_embedding_model,
                )
                token_count += tokens
                for text, vector in zip(batch, vectors):
                    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    self._embedding_cache[key] = vector

            for idx in uncached_indices:
                key = hashlib.sha256(texts[idx].encode("utf-8")).hexdigest()
                embeddings[idx] = self._embedding_cache[key]

        finalized = [vector if vector is not None else [] for vector in embeddings]
        return finalized, token_count

    def _retrieve(self, query: str, index: _StrategyIndex, strategy: StrategyConfig) -> list[tuple[int, float]]:
        backend = self._normalized_backend(strategy.retrieval_backend)
        if backend == "zvec":
            return self._retrieve_with_zvec(query=query, index=index, strategy=strategy)
        return self._retrieve_in_memory(query=query, index=index)

    def _retrieve_in_memory(self, query: str, index: _StrategyIndex) -> list[tuple[int, float]]:
        vectors, _ = self._embed_texts([query])
        query_vector = vectors[0]
        scored: list[tuple[int, float]] = []
        for idx, chunk in enumerate(index.chunks):
            score = self._cosine(query_vector, chunk.embedding)
            scored.append((idx, score))

        limit = min(len(scored), max(1, self.settings.top_candidate_multiplier * 8))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def _retrieve_with_zvec(self, query: str, index: _StrategyIndex, strategy: StrategyConfig) -> list[tuple[int, float]]:
        cache_key = (
            f"{strategy.chunking_strategy}:"
            f"{strategy.temporal_recency_filter}:"
            f"{self._normalized_backend(strategy.retrieval_backend)}"
        )
        zvec_index = self._zvec_index_cache.get(cache_key)
        if zvec_index is None:
            self._build_zvec_index(cache_key=cache_key, index=index)
            zvec_index = self._zvec_index_cache.get(cache_key)
        if zvec_index is None:
            raise ValueError("zvec index was not initialized")

        vectors, _ = self._embed_texts([query])
        query_vector = vectors[0]
        limit = min(len(index.chunks), max(1, self.settings.top_candidate_multiplier * 8))

        rows = self._zvec_search(collection=zvec_index.collection, vector=query_vector, limit=limit)
        scored: list[tuple[int, float]] = []
        for chunk_id, score in rows:
            idx = zvec_index.chunk_id_to_index.get(chunk_id)
            if idx is None:
                continue
            scored.append((idx, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def _normalized_backend(self, backend: str) -> str:
        normalized = str(backend or "in_memory").strip().lower()
        if normalized in {"", "default"}:
            return "in_memory"
        if normalized not in {"in_memory", "zvec"}:
            return "in_memory"
        return normalized

    def _build_zvec_index(self, cache_key: str, index: _StrategyIndex) -> None:
        if cache_key in self._zvec_index_cache:
            return

        try:
            import zvec  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional dependency
            raise ValueError(
                "retrieval_backend='zvec' requires the optional 'zvec' package. "
                "Install it in this environment before running."
            ) from exc

        if not index.chunks:
            self._zvec_index_cache[cache_key] = _ZvecIndex(collection=None, chunk_id_to_index={})
            return

        dimension = len(index.chunks[0].embedding)
        if dimension <= 0:
            raise ValueError("Cannot build zvec index because embeddings are empty.")

        collection = self._zvec_create_collection(zvec_module=zvec, name=cache_key, dimension=dimension)

        ids = [chunk.chunk_id for chunk in index.chunks]
        vectors = [chunk.embedding for chunk in index.chunks]
        self._zvec_insert(collection=collection, ids=ids, vectors=vectors)
        self._zvec_index_cache[cache_key] = _ZvecIndex(
            collection=collection,
            chunk_id_to_index={chunk.chunk_id: idx for idx, chunk in enumerate(index.chunks)},
        )

    def _zvec_create_collection(self, zvec_module: Any, name: str, dimension: int) -> Any:
        safe_name = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]

        if hasattr(zvec_module, "Collection"):
            collection_cls = getattr(zvec_module, "Collection")
            try:
                return collection_cls(name=safe_name, dimension=dimension)
            except TypeError:
                return collection_cls(safe_name, dimension)

        if hasattr(zvec_module, "Client"):
            client = zvec_module.Client()
            for method_name in ("create_collection", "get_or_create_collection"):
                method = getattr(client, method_name, None)
                if callable(method):
                    try:
                        return method(name=safe_name, dimension=dimension)
                    except TypeError:
                        return method(safe_name, dimension)

        raise ValueError(
            "Unsupported zvec Python API shape in this environment. "
            "Expected Collection(...) or Client().create_collection(...)."
        )

    def _zvec_insert(self, collection: Any, ids: list[str], vectors: list[list[float]]) -> None:
        if collection is None:
            return

        for method_name in ("upsert", "insert", "add"):
            method = getattr(collection, method_name, None)
            if not callable(method):
                continue
            try:
                method(ids=ids, vectors=vectors)
                return
            except TypeError:
                try:
                    rows = [{"id": row_id, "vector": vector} for row_id, vector in zip(ids, vectors)]
                    method(rows)
                    return
                except TypeError:
                    continue

        raise ValueError("Unable to insert vectors into zvec collection (unsupported API signature).")

    def _zvec_search(self, collection: Any, vector: list[float], limit: int) -> list[tuple[str, float]]:
        if collection is None:
            return []

        for method_name in ("search", "query"):
            method = getattr(collection, method_name, None)
            if not callable(method):
                continue

            result = None
            try:
                result = method(vector=vector, top_k=limit)
            except TypeError:
                try:
                    result = method(query_vector=vector, k=limit)
                except TypeError:
                    try:
                        result = method([vector], top_k=limit)
                    except TypeError:
                        continue

            rows = self._zvec_extract_rows(result)
            if rows:
                return rows

        raise ValueError("Unable to execute zvec search (unsupported API signature).")

    def _zvec_extract_rows(self, payload: Any) -> list[tuple[str, float]]:
        if payload is None:
            return []

        # dict-like results: {"ids": [...], "scores": [...]}
        if isinstance(payload, dict):
            ids = payload.get("ids")
            scores = payload.get("scores") or payload.get("distances")
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
            if isinstance(scores, list) and scores and isinstance(scores[0], list):
                scores = scores[0]
            if isinstance(ids, list) and isinstance(scores, list):
                out: list[tuple[str, float]] = []
                for row_id, score in zip(ids, scores):
                    if row_id is None:
                        continue
                    out.append((str(row_id), float(score)))
                return out

            hits = payload.get("hits") or payload.get("results")
            if isinstance(hits, list):
                return self._zvec_extract_rows(hits)

        # list-like results: [{"id": "...", "score": 0.9}, ...]
        if isinstance(payload, list):
            out: list[tuple[str, float]] = []
            for item in payload:
                if isinstance(item, dict):
                    row_id = item.get("id") or item.get("key")
                    score = item.get("score")
                    if score is None and item.get("distance") is not None:
                        score = -float(item.get("distance"))
                    if row_id is None or score is None:
                        continue
                    out.append((str(row_id), float(score)))
                    continue

                row_id = getattr(item, "id", None) or getattr(item, "key", None)
                score = getattr(item, "score", None)
                if score is None:
                    distance = getattr(item, "distance", None)
                    score = -float(distance) if distance is not None else None
                if row_id is None or score is None:
                    continue
                out.append((str(row_id), float(score)))
            return out

        return []

    def _followup_query(
        self,
        query: str,
        scored_chunks: list[tuple[int, float]],
        index: _StrategyIndex,
        usage: _UsageCounter,
    ) -> str:
        if not scored_chunks:
            return ""
        contexts = [index.chunks[idx].text for idx, _ in scored_chunks[:3]]
        prompt = (
            "You are a retrieval query optimizer. Given the user query and high-signal evidence, "
            "produce one concise follow-up retrieval query that expands evidence coverage without changing intent."
        )
        user = (
            f"Query:\n{query}\n\nTop evidence:\n" + "\n\n".join(contexts) + "\n\nReturn only the follow-up query."
        )
        try:
            call = self.provider.chat(
                system_prompt=prompt,
                user_prompt=user,
                model=self.settings.openai_chat_model,
                temperature=0.0,
                max_tokens=80,
            )
            usage.add_call(call)
            return call.text.strip()
        except Exception:
            return ""

    def _apply_graph_augmentation(self, scored: dict[int, float], index: _StrategyIndex) -> dict[int, float]:
        boosted = dict(scored)
        top_indices = sorted(scored.keys(), key=lambda idx: scored[idx], reverse=True)[:4]
        for idx in top_indices:
            chunk = index.chunks[idx]
            for entity in chunk.entities:
                for neighbor_idx in index.entity_to_indices.get(entity, set()):
                    if neighbor_idx == idx:
                        continue
                    boosted_score = scored[idx] * 0.72
                    boosted[neighbor_idx] = max(boosted.get(neighbor_idx, -1.0), boosted_score)
        return boosted

    def _apply_temporal_boost(self, scored: dict[int, float], index: _StrategyIndex) -> dict[int, float]:
        dated = [index.chunks[idx].effective_date for idx in scored if index.chunks[idx].effective_date is not None]
        if not dated:
            return scored

        newest = max(dated)
        boosted: dict[int, float] = {}
        for idx, score in scored.items():
            when = index.chunks[idx].effective_date
            if when is None:
                boosted[idx] = score
                continue
            age_days = max(0.0, (newest - when).total_seconds() / 86400.0)
            recency_boost = max(0.0, 1.0 - min(age_days, 3650.0) / 3650.0)
            boosted[idx] = score + (0.08 * recency_boost)
        return boosted

    def _cross_encoder_rerank(
        self,
        query: str,
        ranked_indices: list[int],
        index: _StrategyIndex,
        strategy: StrategyConfig,
        usage: _UsageCounter,
    ) -> list[int]:
        if not ranked_indices:
            return ranked_indices

        candidates = ranked_indices[: self.settings.max_rerank_candidates]
        chunks = [index.chunks[idx] for idx in candidates]
        lines = [f"- {chunk.chunk_id}: {chunk.text[:260]}" for chunk in chunks]

        system = "You are a retrieval reranker. Score each chunk relevance to the query from 0 to 1."
        user = (
            f"Query:\n{query}\n\nChunks:\n" + "\n".join(lines) + "\n\n"
            "Return JSON: {\"scores\": [{\"chunk_id\": \"...\", \"score\": 0.0}]}"
        )

        try:
            call = self.provider.chat(
                system_prompt=system,
                user_prompt=user,
                model=self.settings.openai_verifier_model,
                temperature=0.0,
                max_tokens=500,
            )
            usage.add_call(call)
            payload = extract_json(call.text)
            score_rows = payload.get("scores", []) if isinstance(payload, dict) else []
            mapped: dict[str, float] = {}
            for row in score_rows:
                if not isinstance(row, dict):
                    continue
                chunk_id = str(row.get("chunk_id", "")).strip()
                if not chunk_id:
                    continue
                try:
                    mapped[chunk_id] = float(row.get("score", 0.0))
                except (TypeError, ValueError):
                    continue

            rescored: list[tuple[int, float]] = []
            for idx in ranked_indices:
                chunk = index.chunks[idx]
                llm_score = mapped.get(chunk.chunk_id)
                if llm_score is None:
                    llm_score = 0.35
                rescored.append((idx, llm_score))

            rescored.sort(key=lambda item: item[1], reverse=True)
            return [idx for idx, _ in rescored]
        except Exception:
            return ranked_indices

    def _rewrite_query(self, query: str, strategy: StrategyConfig, usage: _UsageCounter) -> str:
        system = "You rewrite user questions into retrieval-optimized queries while preserving intent."
        user = f"Original query:\n{query}\n\nReturn only rewritten retrieval query."
        try:
            call = self.provider.chat(
                system_prompt=system,
                user_prompt=user,
                model=self.settings.openai_chat_model,
                temperature=0.0,
                max_tokens=100,
            )
            usage.add_call(call)
            return call.text.strip()
        except Exception:
            return ""

    def _hyde_document(self, query: str, strategy: StrategyConfig, usage: _UsageCounter) -> str:
        system = "Generate a concise hypothetical answer passage to improve retrieval recall."
        user = f"Query:\n{query}\n\nWrite a 3-4 sentence hypothetical answer passage."
        try:
            call = self.provider.chat(
                system_prompt=system,
                user_prompt=user,
                model=self.settings.openai_chat_model,
                temperature=0.2,
                max_tokens=180,
            )
            usage.add_call(call)
            return call.text.strip()
        except Exception:
            return ""

    def _generate_answer(
        self,
        query: str,
        contexts: list[_Chunk],
        strategy: StrategyConfig,
        critique: str | None = None,
    ) -> LLMCallResult:
        context_block = "\n\n".join(
            f"[{chunk.chunk_id}] {chunk.text[: self.settings.max_context_chars]}" for chunk in contexts
        )

        rules = [
            "Use only provided evidence.",
            "If evidence is insufficient, say so explicitly.",
        ]
        if strategy.citation_enforcement:
            rules.append("Cite evidence chunk IDs inline after each factual claim, e.g., [doc#c2].")
        if critique:
            rules.append(f"Address this reviewer critique: {critique}")

        system = "You are a high-precision RAG answer assistant. " + " ".join(rules)
        user = f"Question:\n{query}\n\nEvidence:\n{context_block}\n\nAnswer:"
        return self.provider.chat(
            system_prompt=system,
            user_prompt=user,
            model=self.settings.openai_chat_model,
            temperature=0.0,
            max_tokens=self.settings.max_generation_tokens,
        )

    def _self_rag_reflection(
        self,
        query: str,
        answer: str,
        contexts: list[_Chunk],
        strategy: StrategyConfig,
        usage: _UsageCounter,
    ) -> str:
        context_block = "\n\n".join(f"[{chunk.chunk_id}] {chunk.text[:320]}" for chunk in contexts[:4])
        system = "You are a factuality reviewer. Find unsupported claims and concise revision guidance."
        user = (
            f"Question:\n{query}\n\nAnswer:\n{answer}\n\nEvidence:\n{context_block}\n\n"
            "Return JSON: {\"needs_revision\": true/false, \"critique\": \"...\"}."
        )
        try:
            call = self.provider.chat(
                system_prompt=system,
                user_prompt=user,
                model=self.settings.openai_verifier_model,
                temperature=0.0,
                max_tokens=180,
            )
            usage.add_call(call)
            payload = extract_json(call.text)
            if isinstance(payload, dict) and bool(payload.get("needs_revision")):
                return str(payload.get("critique", "")).strip()
            return ""
        except Exception:
            return ""

    def _verify_claims(
        self,
        query: str,
        answer: str,
        contexts: list[_Chunk],
        strategy: StrategyConfig,
        usage: _UsageCounter,
    ) -> list[ClaimVerification]:
        claims = self._claims_from_answer(answer)[: self.settings.max_verifier_claims]
        if not claims:
            return []

        evidence_block = "\n\n".join(
            f"[{chunk.chunk_id}] {chunk.text[: self.settings.max_context_chars]}" for chunk in contexts[:5]
        )
        claim_block = "\n".join(f"{idx + 1}. {claim}" for idx, claim in enumerate(claims))

        system = "You are a strict factual verifier. Judge if each claim is fully supported by evidence."
        user = (
            f"Question:\n{query}\n\nEvidence:\n{evidence_block}\n\nClaims:\n{claim_block}\n\n"
            "Return JSON: {\"claims\": [{\"claim\": \"...\", \"supported\": true/false, "
            "\"confidence\": 0.0-1.0, \"evidence\": \"chunk id and snippet\", \"reasoning\": \"...\"}]}"
        )

        try:
            call = self.provider.chat(
                system_prompt=system,
                user_prompt=user,
                model=self.settings.openai_verifier_model,
                temperature=0.0,
                max_tokens=900,
            )
            usage.add_call(call)
            payload = extract_json(call.text)
            rows = payload.get("claims", []) if isinstance(payload, dict) else []
        except (JsonParseError, Exception):
            rows = []

        verifications: list[ClaimVerification] = []
        if rows:
            for row in rows:
                if not isinstance(row, dict):
                    continue
                claim = str(row.get("claim", "")).strip()
                if not claim:
                    continue
                supported = bool(row.get("supported", False))
                confidence = self._clamp_float(row.get("confidence", 0.0))
                evidence = str(row.get("evidence", "")).strip()
                reasoning = str(row.get("reasoning", "")).strip()
                verifications.append(
                    ClaimVerification(
                        claim=claim,
                        supported=supported,
                        confidence=confidence,
                        evidence=evidence,
                        reasoning=reasoning,
                    )
                )

        if not verifications:
            # Conservative fallback if verifier output is malformed.
            return [ClaimVerification(claim=claim, supported=False, confidence=0.0) for claim in claims]

        return verifications

    def _filter_to_supported_claims(
        self,
        answer: str,
        verifications: list[ClaimVerification],
        citation_ids: list[str],
        enforce_citations: bool,
    ) -> str:
        supported = [item.claim.strip() for item in verifications if item.supported and item.claim.strip()]
        if not supported:
            return "Insufficient grounded evidence to provide a reliable answer."

        if enforce_citations and citation_ids:
            citation = f"[{citation_ids[0]}]"
            with_citations = []
            for claim in supported:
                if CITATION_PATTERN.search(claim):
                    with_citations.append(claim)
                else:
                    with_citations.append(f"{claim} {citation}")
            return " ".join(with_citations)

        return " ".join(supported)

    def _ensure_citations(self, answer: str, citation_ids: list[str]) -> str:
        if not answer.strip() or not citation_ids:
            return answer
        if CITATION_PATTERN.search(answer):
            return answer

        citation = f"[{citation_ids[0]}]"
        sentences = [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(answer.strip()) if s.strip()]
        cited = [f"{sentence} {citation}" for sentence in sentences]
        return " ".join(cited)

    def _claims_from_answer(self, answer: str) -> list[str]:
        claims = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(answer.strip()) if part.strip()]
        return claims

    def _extract_effective_date(self, tags: list[str]) -> datetime | None:
        for tag in tags:
            clean = str(tag).strip()
            if ":" in clean:
                key, value = clean.split(":", 1)
                if key.strip().lower() in {"effective_date", "published", "updated", "date"}:
                    parsed = self._try_parse_date(value.strip())
                    if parsed is not None:
                        return parsed
        return None

    def _try_parse_date(self, value: str) -> datetime | None:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(value[:10], fmt)
            except ValueError:
                continue
        return None

    def _cosine(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for x, y in zip(a, b):
            dot += x * y
            norm_a += x * x
            norm_b += y * y
        if norm_a <= 0.0 or norm_b <= 0.0:
            return 0.0
        return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

    def _clamp_float(self, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, number))
