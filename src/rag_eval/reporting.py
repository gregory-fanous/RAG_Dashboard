from __future__ import annotations

from .models import BenchmarkReport


def build_markdown_summary(report: BenchmarkReport) -> str:
    rows = sorted(
        [result.aggregate for result in report.strategy_results],
        key=lambda item: item.avg_quality_score,
        reverse=True,
    )

    lines = [
        f"# {report.benchmark_name}",
        "",
        f"Run ID: `{report.run_id}`",
        f"Generated: `{report.created_at}`",
        f"Dataset: `{report.dataset_path}`",
        "",
        "## Benchmark Manifesto",
        "",
        "Most teams measure RAG wrong. Here is the evaluation stack a Staff Engineer should require:",
        "",
        "- Retrieval fidelity: Precision@k, Recall@k, MRR",
        "- Answer faithfulness: hallucination detection on generated responses vs retrieved evidence",
        "- Systems economics: latency/quality and token-cost tradeoffs",
        "- Architectural variants: chunking, LogicRAG, recursive retrieval, graph augmentation, HyDE, Self-RAG",
        "",
        "## Leaderboard",
        "",
        "| Rank | Strategy | Backend | Precision@k | Recall@k | MRR | Hallucination Rate | Grounded Claim Ratio | Quality | Avg Latency (ms) | Avg Cost ($) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| "
            + f"{idx} | {row.strategy_name} | {row.retrieval_backend} | {row.avg_precision_at_k:.3f} | {row.avg_recall_at_k:.3f} "
            + f"| {row.mrr:.3f} | {row.avg_hallucination_rate:.3f} | {row.avg_hallucination_score:.3f} | {row.avg_quality_score:.3f} "
            + f"| {row.avg_latency_ms:.1f} | {row.avg_token_cost_usd:.5f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Higher quality strategies should be evaluated against latency and cost budgets, not in isolation.",
            "- Hallucination rate must be monitored as a release gate, not a dashboard afterthought.",
            "- Benchmark variants should remain deterministic so architecture changes are attributable.",
        ]
    )

    return "\n".join(lines)
