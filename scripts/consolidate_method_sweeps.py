#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConsolidatedRow:
    run_id: str
    dataset: str
    method: str
    retrieval_backend: str
    quality: float
    halluc: float
    precision: float
    recall: float
    mrr: float
    latency_ms: float
    avg_cost: float
    total_cost: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "method": self.method,
            "retrieval_backend": self.retrieval_backend,
            "quality": self.quality,
            "halluc": self.halluc,
            "precision": self.precision,
            "recall": self.recall,
            "mrr": self.mrr,
            "latency_ms": self.latency_ms,
            "avg_cost": self.avg_cost,
            "total_cost": self.total_cost,
        }


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _infer_backend(method: str, backend: str | None) -> str:
    if backend:
        normalized = backend.strip().lower()
        if normalized in {"in_memory", "zvec"}:
            return normalized
    if method.endswith("_zvec"):
        return "zvec"
    return "in_memory"


def _extract_run_id(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_method_sweep"):
        return stem.replace("_method_sweep", "")
    return stem


def _normalize_row(row: dict[str, Any], fallback_run_id: str) -> ConsolidatedRow | None:
    dataset = str(row.get("dataset_id", row.get("dataset", ""))).strip()
    method = str(row.get("method_id", row.get("method", ""))).strip()
    run_id = str(row.get("run_id", "")).strip() or fallback_run_id
    if not dataset or not method:
        return None

    backend = _infer_backend(method=method, backend=row.get("retrieval_backend"))
    return ConsolidatedRow(
        run_id=run_id,
        dataset=dataset,
        method=method,
        retrieval_backend=backend,
        quality=_as_float(row.get("avg_quality_score", row.get("quality", 0.0))),
        halluc=_as_float(row.get("avg_hallucination_rate", row.get("halluc", 0.0))),
        precision=_as_float(row.get("avg_precision_at_k", row.get("precision", 0.0))),
        recall=_as_float(row.get("avg_recall_at_k", row.get("recall", 0.0))),
        mrr=_as_float(row.get("mrr", 0.0)),
        latency_ms=_as_float(row.get("avg_latency_ms", row.get("latency_ms", 0.0))),
        avg_cost=_as_float(row.get("avg_token_cost_usd", row.get("avg_cost", 0.0))),
        total_cost=_as_float(row.get("total_token_cost_usd", row.get("total_cost", 0.0))),
    )


def build_consolidated_rows(summary_dir: Path) -> list[ConsolidatedRow]:
    latest_for_key: dict[tuple[str, str, str], ConsolidatedRow] = {}

    files = sorted(summary_dir.glob("*_method_sweep.json"))
    for path in files:
        if path.name == "latest_method_sweep.json":
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        fallback_run_id = _extract_run_id(path)
        rows = payload.get("rows", [])
        if not isinstance(rows, list):
            continue

        for item in rows:
            if not isinstance(item, dict):
                continue
            normalized = _normalize_row(item, fallback_run_id=fallback_run_id)
            if normalized is None:
                continue

            key = (normalized.dataset, normalized.method, normalized.retrieval_backend)
            current = latest_for_key.get(key)
            if current is None or normalized.run_id > current.run_id:
                latest_for_key[key] = normalized

    return sorted(
        latest_for_key.values(),
        key=lambda row: (row.dataset, -row.quality, row.method),
    )


def write_consolidated_outputs(summary_dir: Path, rows: list[ConsolidatedRow]) -> tuple[Path, Path]:
    out_json = summary_dir / "latest_consolidated_method_metrics.json"
    out_md = summary_dir / "latest_consolidated_method_metrics.md"

    payload = {"rows": [row.as_dict() for row in rows]}
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Backend Method Sweep (Real Mode)",
        "",
        "| Dataset | Method | Backend | Quality | Hallucination | Precision@k | Recall@k | MRR | Latency (ms) | Avg Cost ($) | Total Cost ($) | Run ID |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        lines.append(
            "| "
            + f"{row.dataset} | {row.method} | {row.retrieval_backend} | "
            + f"{row.quality:.4f} | {row.halluc:.4f} | {row.precision:.4f} | {row.recall:.4f} | "
            + f"{row.mrr:.4f} | {row.latency_ms:.1f} | {row.avg_cost:.5f} | {row.total_cost:.5f} | {row.run_id} |"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_json, out_md


def consolidate_method_sweeps(summary_dir: Path) -> tuple[Path, Path, int]:
    rows = build_consolidated_rows(summary_dir=summary_dir)
    out_json, out_md = write_consolidated_outputs(summary_dir=summary_dir, rows=rows)
    return out_json, out_md, len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate method sweep summaries into latest comparison tables.")
    parser.add_argument(
        "--summary-dir",
        default="artifacts/sweeps",
        help="Directory containing *_method_sweep.json files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_dir = Path(args.summary_dir).resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_json, out_md, count = consolidate_method_sweeps(summary_dir=summary_dir)
    print(f"[done] consolidated_rows={count} json={out_json} md={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
