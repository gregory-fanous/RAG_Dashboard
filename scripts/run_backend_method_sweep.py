#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.datasets import DatasetRegistry
from backend.app.schemas import WorkflowRunRequest
from backend.app.workflows import WorkflowService

from consolidate_method_sweeps import consolidate_method_sweeps


@dataclass(frozen=True)
class MethodPreset:
    method_id: str
    chunking_strategy: str
    retrieval_k: int
    retrieval_backend: str = "in_memory"
    logicrag: bool = False
    recursive_retrieval: bool = False
    graph_augmentation: bool = False
    hyde: bool = False
    self_rag: bool = False
    query_rewrite: bool = False
    cross_encoder_rerank: bool = False
    temporal_recency_filter: bool = False
    citation_enforcement: bool = True
    claim_verification: bool = True


METHOD_PRESETS: dict[str, MethodPreset] = {
    "baseline_fixed": MethodPreset(
        method_id="baseline_fixed",
        chunking_strategy="fixed",
        retrieval_k=5,
    ),
    "baseline_fixed_zvec": MethodPreset(
        method_id="baseline_fixed_zvec",
        chunking_strategy="fixed",
        retrieval_backend="zvec",
        retrieval_k=5,
    ),
    "hybrid_dense": MethodPreset(
        method_id="hybrid_dense",
        chunking_strategy="semantic",
        retrieval_k=6,
        query_rewrite=True,
        cross_encoder_rerank=True,
        recursive_retrieval=True,
        citation_enforcement=True,
        claim_verification=True,
    ),
    "hybrid_dense_zvec": MethodPreset(
        method_id="hybrid_dense_zvec",
        chunking_strategy="semantic",
        retrieval_backend="zvec",
        retrieval_k=6,
        query_rewrite=True,
        cross_encoder_rerank=True,
        recursive_retrieval=True,
        citation_enforcement=True,
        claim_verification=True,
    ),
    "graphrag_plus": MethodPreset(
        method_id="graphrag_plus",
        chunking_strategy="adaptive",
        retrieval_k=7,
        query_rewrite=True,
        cross_encoder_rerank=True,
        recursive_retrieval=True,
        graph_augmentation=True,
        hyde=True,
        temporal_recency_filter=True,
        citation_enforcement=True,
        claim_verification=True,
    ),
    "graphrag_plus_zvec": MethodPreset(
        method_id="graphrag_plus_zvec",
        chunking_strategy="adaptive",
        retrieval_backend="zvec",
        retrieval_k=7,
        query_rewrite=True,
        cross_encoder_rerank=True,
        recursive_retrieval=True,
        graph_augmentation=True,
        hyde=True,
        temporal_recency_filter=True,
        citation_enforcement=True,
        claim_verification=True,
    ),
    "self_guarded": MethodPreset(
        method_id="self_guarded",
        chunking_strategy="adaptive",
        retrieval_k=7,
        query_rewrite=True,
        cross_encoder_rerank=True,
        recursive_retrieval=True,
        graph_augmentation=True,
        hyde=True,
        self_rag=True,
        temporal_recency_filter=True,
        citation_enforcement=True,
        claim_verification=True,
    ),
    "self_guarded_zvec": MethodPreset(
        method_id="self_guarded_zvec",
        chunking_strategy="adaptive",
        retrieval_backend="zvec",
        retrieval_k=7,
        query_rewrite=True,
        cross_encoder_rerank=True,
        recursive_retrieval=True,
        graph_augmentation=True,
        hyde=True,
        self_rag=True,
        temporal_recency_filter=True,
        citation_enforcement=True,
        claim_verification=True,
    ),
    "logicrag_guarded": MethodPreset(
        method_id="logicrag_guarded",
        chunking_strategy="semantic",
        retrieval_k=6,
        logicrag=True,
        query_rewrite=True,
        hyde=True,
        temporal_recency_filter=True,
        citation_enforcement=True,
        claim_verification=True,
    ),
}


def _build_request(
    *,
    dataset_id: str,
    sample_size: int,
    random_seed: int,
    target_hallucination_rate: float,
    method: MethodPreset,
) -> WorkflowRunRequest:
    return WorkflowRunRequest(
        dataset_id=dataset_id,
        workflow_name=f"sweep_{dataset_id}_{method.method_id}",
        execution_mode="real",
        mode="single",
        sample_size=sample_size,
        random_seed=random_seed,
        chunking_strategy=method.chunking_strategy,  # type: ignore[arg-type]
        retrieval_backend=method.retrieval_backend,  # type: ignore[arg-type]
        retrieval_k=method.retrieval_k,
        logicrag=method.logicrag,
        recursive_retrieval=method.recursive_retrieval,
        graph_augmentation=method.graph_augmentation,
        hyde=method.hyde,
        self_rag=method.self_rag,
        query_rewrite=method.query_rewrite,
        cross_encoder_rerank=method.cross_encoder_rerank,
        temporal_recency_filter=method.temporal_recency_filter,
        citation_enforcement=method.citation_enforcement,
        claim_verification=method.claim_verification,
        include_baseline=False,
        target_hallucination_rate=target_hallucination_rate,
    )


def _fingerprint(request: WorkflowRunRequest) -> tuple[Any, ...]:
    return (
        request.dataset_id,
        request.execution_mode,
        request.mode,
        request.sample_size,
        request.random_seed,
        request.chunking_strategy,
        request.retrieval_backend,
        request.retrieval_k,
        request.logicrag,
        request.recursive_retrieval,
        request.graph_augmentation,
        request.hyde,
        request.self_rag,
        request.query_rewrite,
        request.cross_encoder_rerank,
        request.temporal_recency_filter,
        request.citation_enforcement,
        request.claim_verification,
        request.target_hallucination_rate,
        request.include_baseline,
    )


def _existing_fingerprints(workflows_dir: Path) -> set[tuple[Any, ...]]:
    found: set[tuple[Any, ...]] = set()
    for path in sorted(workflows_dir.glob("*.json")):
        if path.name == "latest.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        request = payload.get("request")
        if not isinstance(request, dict):
            continue
        try:
            req_model = WorkflowRunRequest.model_validate(request)
        except Exception:
            continue
        found.add(_fingerprint(req_model))
    return found


def _aggregate_row(detail: dict[str, Any]) -> dict[str, Any]:
    report = detail.get("report", {})
    strategy_results = report.get("strategy_results", [])
    if not strategy_results:
        return {}
    aggregate = strategy_results[0].get("aggregate", {})
    return {
        "strategy_name": aggregate.get("strategy_name"),
        "retrieval_backend": aggregate.get("retrieval_backend"),
        "avg_quality_score": aggregate.get("avg_quality_score"),
        "avg_hallucination_rate": aggregate.get("avg_hallucination_rate"),
        "avg_precision_at_k": aggregate.get("avg_precision_at_k"),
        "avg_recall_at_k": aggregate.get("avg_recall_at_k"),
        "mrr": aggregate.get("mrr"),
        "avg_latency_ms": aggregate.get("avg_latency_ms"),
        "avg_token_cost_usd": aggregate.get("avg_token_cost_usd"),
        "total_token_cost_usd": aggregate.get("total_token_cost_usd"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run backend real-mode method sweeps with cache-aware skipping."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["public_eval_set", "ragcare_qa"],
        help="Dataset IDs to sweep.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["baseline_fixed", "hybrid_dense", "graphrag_plus", "self_guarded", "logicrag_guarded"],
        help=f"Method IDs to run. Available: {', '.join(sorted(METHOD_PRESETS.keys()))}",
    )
    parser.add_argument("--sample-size", type=int, default=10, help="Cases per run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--target-hallucination-rate",
        type=float,
        default=0.01,
        help="Governance target used in workflow requests.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/workflows",
        help="Workflow artifacts directory.",
    )
    parser.add_argument(
        "--summary-dir",
        default="artifacts/sweeps",
        help="Summary output directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache and rerun even if an identical request exists.",
    )
    parser.add_argument(
        "--skip-consolidate",
        action="store_true",
        help="Skip refreshing latest_consolidated_method_metrics outputs after the sweep.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workflows_dir = Path(args.artifacts_dir).resolve()
    workflows_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = Path(args.summary_dir).resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)

    methods: list[MethodPreset] = []
    for method_id in args.methods:
        preset = METHOD_PRESETS.get(method_id)
        if preset is None:
            print(f"Unknown method preset: {method_id}", file=sys.stderr)
            return 2
        methods.append(preset)

    registry = DatasetRegistry()
    service = WorkflowService(artifacts_dir=workflows_dir)

    existing = _existing_fingerprints(workflows_dir) if not args.force else set()

    rows: list[dict[str, Any]] = []
    skipped = 0
    started_at = datetime.now(timezone.utc).isoformat()

    for dataset_id in args.datasets:
        for method in methods:
            request = _build_request(
                dataset_id=dataset_id,
                sample_size=args.sample_size,
                random_seed=args.seed,
                target_hallucination_rate=args.target_hallucination_rate,
                method=method,
            )
            key = _fingerprint(request)
            if key in existing:
                skipped += 1
                print(f"[skip] dataset={dataset_id} method={method.method_id} (cached)")
                continue

            print(f"[run] dataset={dataset_id} method={method.method_id}")
            detail = service.run_workflow(request=request, dataset_registry=registry).model_dump(mode="json")
            aggregate = _aggregate_row(detail)
            row = {
                "run_id": detail.get("summary", {}).get("run_id"),
                "dataset_id": dataset_id,
                "method_id": method.method_id,
                **aggregate,
            }
            rows.append(row)
            existing.add(key)

            print(
                "      quality={:.4f} halluc={:.4f} latency_ms={:.1f} avg_cost={:.5f}".format(
                    float(row.get("avg_quality_score") or 0.0),
                    float(row.get("avg_hallucination_rate") or 0.0),
                    float(row.get("avg_latency_ms") or 0.0),
                    float(row.get("avg_token_cost_usd") or 0.0),
                )
            )

    finished_at = datetime.now(timezone.utc).isoformat()
    out = {
        "started_at": started_at,
        "finished_at": finished_at,
        "datasets": args.datasets,
        "methods": [method.method_id for method in methods],
        "sample_size": args.sample_size,
        "seed": args.seed,
        "target_hallucination_rate": args.target_hallucination_rate,
        "skipped_cached_runs": skipped,
        "executed_runs": len(rows),
        "rows": rows,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = summary_dir / f"{ts}_method_sweep.json"
    latest_path = summary_dir / "latest_method_sweep.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    if args.skip_consolidate:
        print(f"[done] summary={out_path}")
        return 0

    consolidated_json, consolidated_md, row_count = consolidate_method_sweeps(summary_dir=summary_dir)
    print(
        f"[done] summary={out_path} consolidated_rows={row_count} "
        f"consolidated_json={consolidated_json} consolidated_md={consolidated_md}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
