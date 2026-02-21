from __future__ import annotations

import re
from statistics import mean

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def precision_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    if not retrieved_doc_ids:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    relevant = set(relevant_doc_ids)
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / float(k)


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: list[str], k: int) -> float:
    if not relevant_doc_ids:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    relevant = set(relevant_doc_ids)
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / float(len(relevant))


def reciprocal_rank(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    relevant = set(relevant_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant:
            return 1.0 / float(rank)
    return 0.0


def _normalize_tokens(text: str) -> set[str]:
    raw_tokens = TOKEN_PATTERN.findall(text.lower())
    return {tok for tok in raw_tokens if len(tok) > 2 and tok not in STOPWORDS}


def _split_claims(answer: str) -> list[str]:
    segments = [segment.strip() for segment in SENTENCE_SPLIT_PATTERN.split(answer.strip())]
    return [segment for segment in segments if segment]


def hallucination_detection_score(answer: str, contexts: list[str], support_threshold: float = 0.45) -> float:
    claims = _split_claims(answer)
    if not claims:
        return 0.0

    context_tokens = _normalize_tokens(" ".join(contexts))
    if not context_tokens:
        return 0.0

    supported = 0
    for claim in claims:
        claim_tokens = _normalize_tokens(claim)
        if not claim_tokens:
            continue
        overlap_ratio = len(claim_tokens & context_tokens) / float(len(claim_tokens))
        if overlap_ratio >= support_threshold:
            supported += 1

    return supported / float(len(claims))


def composite_quality_score(
    precision: float,
    recall: float,
    mrr: float,
    hallucination_score: float,
    weights: tuple[float, float, float, float] = (0.25, 0.35, 0.25, 0.15),
) -> float:
    p_w, r_w, m_w, h_w = weights
    return (
        (precision * p_w)
        + (recall * r_w)
        + (mrr * m_w)
        + (hallucination_score * h_w)
    )


def mean_metric(values: list[float]) -> float:
    if not values:
        return 0.0
    return mean(values)
