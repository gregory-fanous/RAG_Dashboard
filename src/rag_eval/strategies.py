from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass

from .models import CaseRun, EvaluationCase, EvaluationDataset, StrategyConfig

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass
class SyntheticRAGSystem:
    dataset: EvaluationDataset
    seed: int = 42

    def run_case(self, case: EvaluationCase, strategy: StrategyConfig) -> CaseRun:
        rng = self._rng_for(case.query_id, strategy.name)
        doc_index = self.dataset.to_doc_index()
        all_doc_ids = [doc.doc_id for doc in self.dataset.documents]

        retrieved_doc_ids = self._simulate_retrieval(
            rng=rng,
            relevant_doc_ids=case.ground_truth_doc_ids,
            all_doc_ids=all_doc_ids,
            k=strategy.retrieval_k,
            strength=self._retrieval_strength(strategy),
        )

        contexts = [doc_index[doc_id].text for doc_id in retrieved_doc_ids if doc_id in doc_index]
        answer = self._simulate_answer(
            rng=rng,
            strategy=strategy,
            reference_answer=case.reference_answer,
            contexts=contexts,
            hallucination_rate=self._hallucination_rate(strategy),
        )

        prompt_tokens, completion_tokens = self._token_profile(
            strategy=strategy,
            query=case.query,
            contexts=contexts,
            answer=answer,
        )

        retrieval_latency_ms, generation_latency_ms = self._latency_profile(
            rng=rng,
            strategy=strategy,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        return CaseRun(
            query_id=case.query_id,
            retrieved_doc_ids=retrieved_doc_ids,
            contexts=contexts,
            answer=answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
        )

    def _rng_for(self, query_id: str, strategy_name: str) -> random.Random:
        payload = f"{self.seed}:{query_id}:{strategy_name}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        seed = int(digest[:16], 16)
        return random.Random(seed)

    def _retrieval_strength(self, strategy: StrategyConfig) -> float:
        score = 0.45
        chunk_bonus = {
            "fixed": 0.0,
            "sentence_window": 0.04,
            "semantic": 0.08,
            "adaptive": 0.11,
        }
        score += chunk_bonus.get(strategy.chunking_strategy, 0.02)
        if strategy.logicrag:
            score += 0.10
        if strategy.recursive_retrieval:
            score += 0.10
        if strategy.graph_augmentation:
            score += 0.08
        if strategy.hyde:
            score += 0.06
        if strategy.self_rag:
            score += 0.07
        if strategy.query_rewrite:
            score += 0.04
        if strategy.cross_encoder_rerank:
            score += 0.06
        if strategy.temporal_recency_filter:
            score += 0.02
        if strategy.citation_enforcement:
            score += 0.03
        if strategy.claim_verification:
            score += 0.05
        return max(0.05, min(0.95, score))

    def _hallucination_rate(self, strategy: StrategyConfig) -> float:
        score = 0.20
        if strategy.logicrag:
            score -= 0.06
        if strategy.chunking_strategy == "semantic":
            score -= 0.03
        if strategy.chunking_strategy == "adaptive":
            score -= 0.04
        if strategy.recursive_retrieval:
            score -= 0.03
        if strategy.graph_augmentation:
            score -= 0.06
        if strategy.hyde:
            score -= 0.02
        if strategy.self_rag:
            score -= 0.09
        if strategy.query_rewrite:
            score -= 0.02
        if strategy.cross_encoder_rerank:
            score -= 0.04
        if strategy.temporal_recency_filter:
            score -= 0.01
        if strategy.citation_enforcement:
            score -= 0.08
        if strategy.claim_verification:
            score -= 0.09

        # When both citation and verification are enabled, this acts like a faithfulness firewall.
        if strategy.citation_enforcement and strategy.claim_verification:
            score = min(score, 0.005)

        return max(0.001, min(0.55, score))

    def _simulate_retrieval(
        self,
        rng: random.Random,
        relevant_doc_ids: list[str],
        all_doc_ids: list[str],
        k: int,
        strength: float,
    ) -> list[str]:
        if k <= 0:
            return []

        relevant = list(dict.fromkeys(relevant_doc_ids))
        non_relevant = [doc_id for doc_id in all_doc_ids if doc_id not in relevant]

        selected: set[str] = set()
        for doc_id in relevant:
            if rng.random() < strength:
                selected.add(doc_id)

        if not selected and relevant and rng.random() < (strength * 0.65):
            selected.add(rng.choice(relevant))

        additional_pool = non_relevant.copy()
        rng.shuffle(additional_pool)
        for doc_id in additional_pool:
            if len(selected) >= max(k * 2, len(relevant)):
                break
            selected.add(doc_id)

        scored: list[tuple[float, str]] = []
        for doc_id in selected:
            if doc_id in relevant:
                score = rng.uniform(0.65, 1.0) + (strength * 0.40)
            else:
                score = rng.uniform(0.0, 0.8) + ((1.0 - strength) * 0.28)
            scored.append((score, doc_id))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc_id for _, doc_id in scored[:k]]

    def _simulate_answer(
        self,
        rng: random.Random,
        strategy: StrategyConfig,
        reference_answer: str,
        contexts: list[str],
        hallucination_rate: float,
    ) -> str:
        if strategy.claim_verification:
            return self._verified_context_answer(contexts=contexts)

        if strategy.citation_enforcement or strategy.claim_verification:
            chosen = self._grounded_answer_sentences(contexts=contexts)
        else:
            reference_sentences = [
                sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(reference_answer) if sentence.strip()
            ]
            chosen = reference_sentences[:2] if reference_sentences else [reference_answer.strip()]

        if not chosen:
            chosen = ["No grounded answer available from retrieved evidence."]

        if strategy.self_rag:
            chosen.append("The answer is supported by retrieved passages and avoids unsupported extrapolation.")

        if rng.random() < hallucination_rate:
            chosen.append(
                "The system also relies on a quantum centroid cache that independently verifies every claim."
            )

        if contexts and rng.random() < 0.65:
            context_tokens = contexts[0].split()
            quote = " ".join(context_tokens[: min(12, len(context_tokens))])
            if quote:
                chosen.append(f"Evidence excerpt: {quote}.")

        return " ".join(chosen)

    def _verified_context_answer(self, contexts: list[str]) -> str:
        if not contexts:
            return "Insufficient grounded evidence."

        # Claim-verification mode returns only direct context spans to minimize unsupported claims.
        tokens = contexts[0].split()
        snippet = " ".join(tokens[: min(24, len(tokens))]).strip()
        if not snippet:
            return "Insufficient grounded evidence."
        return f"{snippet}."

    def _grounded_answer_sentences(self, contexts: list[str]) -> list[str]:
        if not contexts:
            return ["Insufficient grounded evidence was retrieved for this query."]

        excerpts: list[str] = []
        for context in contexts[:2]:
            tokens = context.split()
            snippet = " ".join(tokens[: min(16, len(tokens))]).strip()
            if snippet:
                excerpts.append(f"Grounded evidence: {snippet}.")

        if not excerpts:
            return ["Insufficient grounded evidence was retrieved for this query."]
        return excerpts

    def _token_profile(
        self,
        strategy: StrategyConfig,
        query: str,
        contexts: list[str],
        answer: str,
    ) -> tuple[int, int]:
        query_tokens = max(8, len(query.split()))
        context_tokens = sum(len(context.split()) for context in contexts)
        answer_tokens = max(8, len(answer.split()))

        prompt_tokens = int(
            110
            + (query_tokens * 2)
            + context_tokens
            + (strategy.retrieval_k * 26)
            + (45 if strategy.query_rewrite else 0)
            + (70 if strategy.cross_encoder_rerank else 0)
            + (24 if strategy.temporal_recency_filter else 0)
            + (85 if strategy.hyde else 0)
            + (55 if strategy.recursive_retrieval else 0)
            + (75 if strategy.graph_augmentation else 0)
            + (120 if strategy.self_rag else 0)
            + (110 if strategy.logicrag else 0)
            + (65 if strategy.citation_enforcement else 0)
            + (95 if strategy.claim_verification else 0)
        )
        completion_tokens = int(
            (answer_tokens * 1.35)
            + (35 if strategy.self_rag else 0)
            + (18 if strategy.citation_enforcement else 0)
            + (26 if strategy.claim_verification else 0)
        )
        return prompt_tokens, completion_tokens

    def _latency_profile(
        self,
        rng: random.Random,
        strategy: StrategyConfig,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> tuple[float, float]:
        retrieval_latency = (
            35.0
            + (strategy.retrieval_k * 5.8)
            + (12.0 if strategy.query_rewrite else 0.0)
            + (22.0 if strategy.cross_encoder_rerank else 0.0)
            + (9.0 if strategy.temporal_recency_filter else 0.0)
            + (12.0 if strategy.chunking_strategy == "semantic" else 0.0)
            + (18.0 if strategy.chunking_strategy == "adaptive" else 0.0)
            + (25.0 if strategy.recursive_retrieval else 0.0)
            + (28.0 if strategy.graph_augmentation else 0.0)
            + (22.0 if strategy.hyde else 0.0)
            + (34.0 if strategy.logicrag else 0.0)
        )
        generation_latency = (
            75.0
            + (prompt_tokens * 0.05)
            + (completion_tokens * 1.3)
            + (30.0 if strategy.self_rag else 0.0)
            + (16.0 if strategy.citation_enforcement else 0.0)
            + (34.0 if strategy.claim_verification else 0.0)
        )

        retrieval_latency += rng.uniform(-4.0, 7.0)
        generation_latency += rng.uniform(-10.0, 12.0)

        return max(5.0, retrieval_latency), max(20.0, generation_latency)
