import json
import sys
from pathlib import Path


def _load_module():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import consolidate_method_sweeps  # type: ignore

    return consolidate_method_sweeps


def test_consolidation_dedupes_to_latest_run_and_infers_backend(tmp_path):
    module = _load_module()

    first = {
        "rows": [
            {
                "run_id": "20260101T000000Z",
                "dataset_id": "retrievalqa",
                "method_id": "baseline_fixed",
                "avg_quality_score": 0.5,
                "avg_hallucination_rate": 0.2,
                "avg_precision_at_k": 0.1,
                "avg_recall_at_k": 0.2,
                "mrr": 0.3,
                "avg_latency_ms": 1000,
                "avg_token_cost_usd": 0.01,
                "total_token_cost_usd": 0.1,
            }
        ]
    }
    second = {
        "rows": [
            {
                "run_id": "20260102T000000Z",
                "dataset_id": "retrievalqa",
                "method_id": "baseline_fixed",
                "avg_quality_score": 0.6,
                "avg_hallucination_rate": 0.1,
                "avg_precision_at_k": 0.2,
                "avg_recall_at_k": 0.3,
                "mrr": 0.4,
                "avg_latency_ms": 900,
                "avg_token_cost_usd": 0.02,
                "total_token_cost_usd": 0.2,
            },
            {
                "run_id": "20260102T000000Z",
                "dataset_id": "retrievalqa",
                "method_id": "baseline_fixed_zvec",
                "avg_quality_score": 0.7,
                "avg_hallucination_rate": 0.05,
                "avg_precision_at_k": 0.25,
                "avg_recall_at_k": 0.35,
                "mrr": 0.45,
                "avg_latency_ms": 850,
                "avg_token_cost_usd": 0.03,
                "total_token_cost_usd": 0.3,
            },
        ]
    }

    (tmp_path / "20260101T000000Z_method_sweep.json").write_text(json.dumps(first), encoding="utf-8")
    (tmp_path / "20260102T000000Z_method_sweep.json").write_text(json.dumps(second), encoding="utf-8")

    out_json, out_md, count = module.consolidate_method_sweeps(summary_dir=tmp_path)
    assert count == 2
    assert out_json.exists()
    assert out_md.exists()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    rows = payload["rows"]
    assert len(rows) == 2

    in_memory = next(row for row in rows if row["method"] == "baseline_fixed")
    assert in_memory["run_id"] == "20260102T000000Z"
    assert in_memory["retrieval_backend"] == "in_memory"

    zvec = next(row for row in rows if row["method"] == "baseline_fixed_zvec")
    assert zvec["retrieval_backend"] == "zvec"
