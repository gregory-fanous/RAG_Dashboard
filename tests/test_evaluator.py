from pathlib import Path

from rag_eval.evaluator import BenchmarkEvaluator
from rag_eval.io import load_benchmark_config, load_dataset
from rag_eval.models import BenchmarkReport
from rag_eval.strategies import SyntheticRAGSystem


def test_benchmark_evaluator_runs_end_to_end():
    workspace = Path(__file__).resolve().parents[1]
    config = load_benchmark_config(workspace / "benchmarks" / "public_benchmark.json")
    dataset = load_dataset(workspace / "data" / "public_eval_set.json")

    runner = SyntheticRAGSystem(dataset=dataset, seed=config.random_seed)
    evaluator = BenchmarkEvaluator(runner=runner)
    report = evaluator.evaluate(config)

    assert report.benchmark_name == "public_rag_stack_benchmark"
    assert len(report.strategy_results) == len(config.strategies)
    assert all(result.aggregate.avg_quality_score >= 0.0 for result in report.strategy_results)
    assert all(result.aggregate.avg_latency_ms > 0.0 for result in report.strategy_results)


def test_benchmark_report_run_id_is_unique():
    first = BenchmarkReport.create("bench", "data.json", [])
    second = BenchmarkReport.create("bench", "data.json", [])
    assert first.run_id != second.run_id
