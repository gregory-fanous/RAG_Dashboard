from __future__ import annotations

import math
import re
from statistics import mean
from typing import Any

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
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


def _split_claims(answer: str) -> list[str]:
    parts = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(answer.strip())]
    return [part for part in parts if part]


def _normalize_tokens(text: str) -> set[str]:
    tokens = TOKEN_PATTERN.findall(text.lower())
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def _support_ratio(claim: str, contexts: list[str]) -> float:
    claim_tokens = _normalize_tokens(claim)
    if not claim_tokens:
        return 0.0
    context_tokens = _normalize_tokens(" ".join(contexts))
    if not context_tokens:
        return 0.0
    return len(claim_tokens & context_tokens) / float(len(claim_tokens))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(math.ceil((pct / 100.0) * len(sorted_values))) - 1
    index = min(max(index, 0), len(sorted_values) - 1)
    return sorted_values[index]


def extract_unsupported_claims(
    report: dict[str, Any],
    query_lookup: dict[str, dict[str, Any]],
    support_threshold: float = 0.45,
    top_n: int = 30,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    for strategy_result in report.get("strategy_results", []):
        strategy_name = strategy_result.get("aggregate", {}).get("strategy_name", "unknown")
        case_runs = strategy_result.get("case_runs", [])

        for case_run in case_runs:
            query_id = case_run.get("query_id", "unknown")
            query_meta = query_lookup.get(query_id, {})
            query_text = query_meta.get("query", "")
            contexts = case_run.get("contexts", [])
            answer = case_run.get("answer", "")
            claim_verifications = case_run.get("claim_verifications", [])

            if isinstance(claim_verifications, list) and claim_verifications:
                for row in claim_verifications:
                    if not isinstance(row, dict):
                        continue
                    if bool(row.get("supported", False)):
                        continue
                    claim = str(row.get("claim", "")).strip()
                    if not claim:
                        continue
                    try:
                        confidence = float(row.get("confidence", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        confidence = 0.0
                    findings.append(
                        {
                            "strategy_name": strategy_name,
                            "query_id": query_id,
                            "query": query_text,
                            "intent": query_meta.get("intent", "general"),
                            "claim": claim,
                            "support_ratio": round(max(0.0, min(1.0, 1.0 - confidence)), 4),
                            "contexts": contexts[:2],
                        }
                    )
                continue

            for claim in _split_claims(answer):
                ratio = _support_ratio(claim, contexts)
                if ratio >= support_threshold:
                    continue
                findings.append(
                    {
                        "strategy_name": strategy_name,
                        "query_id": query_id,
                        "query": query_text,
                        "intent": query_meta.get("intent", "general"),
                        "claim": claim,
                        "support_ratio": round(ratio, 4),
                        "contexts": contexts[:2],
                    }
                )

    findings.sort(key=lambda item: item["support_ratio"])
    return findings[:top_n]


def build_intent_breakdown(
    report: dict[str, Any],
    query_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[tuple[float, float]]] = {}

    for strategy_result in report.get("strategy_results", []):
        strategy_name = strategy_result.get("aggregate", {}).get("strategy_name", "unknown")
        for case_metric in strategy_result.get("case_metrics", []):
            query_id = case_metric.get("query_id", "unknown")
            intent = query_lookup.get(query_id, {}).get("intent", "general")
            key = (strategy_name, intent)
            support_score = float(case_metric.get("hallucination_score", 0.0))
            hallucination_rate = float(case_metric.get("hallucination_rate", 1.0 - support_score))
            buckets.setdefault(key, []).append((support_score, hallucination_rate))

    rows = [
        {
            "strategy_name": strategy_name,
            "intent": intent,
            "avg_grounded_claim_ratio": round(mean(item[0] for item in values), 4),
            "avg_hallucination_rate": round(mean(item[1] for item in values), 4),
            "case_count": len(values),
        }
        for (strategy_name, intent), values in buckets.items()
    ]
    rows.sort(key=lambda item: (item["strategy_name"], item["intent"]))
    return rows


def build_latency_slo_snapshot(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for strategy_result in report.get("strategy_results", []):
        aggregate = strategy_result.get("aggregate", {})
        strategy_name = aggregate.get("strategy_name", "unknown")
        latencies = [float(item.get("total_latency_ms", 0.0)) for item in strategy_result.get("case_metrics", [])]

        rows.append(
            {
                "strategy_name": strategy_name,
                "p50_latency_ms": round(_percentile(latencies, 50), 2),
                "p95_latency_ms": round(_percentile(latencies, 95), 2),
                "avg_latency_ms": round(float(aggregate.get("avg_latency_ms", 0.0)), 2),
            }
        )

    rows.sort(key=lambda item: item["avg_latency_ms"])
    return rows


def build_cost_per_quality(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy_result in report.get("strategy_results", []):
        aggregate = strategy_result.get("aggregate", {})
        quality = float(aggregate.get("avg_quality_score", 0.0))
        total_cost = float(aggregate.get("total_token_cost_usd", 0.0))
        rows.append(
            {
                "strategy_name": aggregate.get("strategy_name", "unknown"),
                "avg_quality_score": round(quality, 4),
                "total_token_cost_usd": round(total_cost, 6),
                "cost_per_quality_point": round(total_cost / max(quality, 1e-9), 6),
            }
        )

    rows.sort(key=lambda item: item["cost_per_quality_point"])
    return rows


def build_safety_violations(
    report: dict[str, Any],
    query_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []

    for strategy_result in report.get("strategy_results", []):
        strategy_name = strategy_result.get("aggregate", {}).get("strategy_name", "unknown")
        for case_run in strategy_result.get("case_runs", []):
            query_id = case_run.get("query_id", "")
            answer = str(case_run.get("answer", ""))
            guardrails = query_lookup.get(query_id, {}).get("must_not_include", [])
            lower_answer = answer.lower()

            triggered = [rule for rule in guardrails if str(rule).lower() in lower_answer]
            if triggered:
                violations.append(
                    {
                        "strategy_name": strategy_name,
                        "query_id": query_id,
                        "triggered_rules": triggered,
                        "answer_excerpt": answer[:200],
                    }
                )

    return violations
