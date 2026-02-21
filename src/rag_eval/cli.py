from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .dashboard import build_dashboard_html_from_payload, write_dashboard
from .evaluator import BenchmarkEvaluator
from .io import (
    load_benchmark_config,
    load_dataset,
    load_json,
    write_report,
    write_text,
)
from .logicrag_client import LogicRAGClient
from .models import BenchmarkConfig
from .providers import OpenAIProvider
from .real_runner import RealRAGSystem
from .reporting import build_markdown_summary
from .runtime import RuntimeSettings
from .strategies import SyntheticRAGSystem


def _resolve_dataset_path(config: BenchmarkConfig, benchmark_path: Path) -> BenchmarkConfig:
    dataset_path = Path(config.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = (benchmark_path.parent / dataset_path).resolve()
    return replace(config, dataset_path=str(dataset_path))


def run_benchmark(args: argparse.Namespace) -> int:
    benchmark_path = Path(args.benchmark).resolve()
    config = load_benchmark_config(benchmark_path)
    config = _resolve_dataset_path(config, benchmark_path)

    dataset = load_dataset(config.dataset_path)
    runner = _build_runner(
        dataset=dataset,
        seed=config.random_seed,
        execution_mode=args.execution_mode,
    )
    evaluator = BenchmarkEvaluator(runner=runner)
    report = evaluator.evaluate(config=config)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_json = output_dir / f"{report.run_id}.json"
    latest_json = output_dir / "latest.json"
    run_md = output_dir / f"{report.run_id}.md"
    latest_md = output_dir / "latest.md"

    write_report(report, run_json)
    write_report(report, latest_json)

    summary = build_markdown_summary(report)
    write_text(summary, run_md)
    write_text(summary, latest_md)

    dashboard_out = Path(args.dashboard_out).resolve() if args.dashboard_out else output_dir / "dashboard.html"
    write_dashboard(report, dashboard_out)

    print(f"Benchmark run complete: {run_json}")
    print(f"Summary report: {run_md}")
    print(f"Dashboard: {dashboard_out}")
    return 0


def _build_runner(dataset, seed: int, execution_mode: str):
    mode = (execution_mode or "real").strip().lower()
    settings = RuntimeSettings.from_env(load_dotenv_file=True)
    if mode == "synthetic":
        settings.validate_for_synthetic_mode()
        return SyntheticRAGSystem(dataset=dataset, seed=seed)

    settings.validate_for_real_mode()
    provider = OpenAIProvider(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout_sec=settings.request_timeout_sec,
    )
    logicrag_client = None
    if settings.logicrag_service_url:
        logicrag_client = LogicRAGClient(
            base_url=settings.logicrag_service_url,
            timeout_sec=settings.logicrag_timeout_sec,
        )
    return RealRAGSystem(
        dataset=dataset,
        provider=provider,
        settings=settings,
        logicrag_client=logicrag_client,
        seed=seed,
    )


def render_dashboard(args: argparse.Namespace) -> int:
    payload = load_json(args.report)
    html = build_dashboard_html_from_payload(payload)
    output = Path(args.output).resolve()
    write_text(html, output)

    print(f"Dashboard written: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG evaluation benchmark and dashboard")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute benchmark and emit artifacts")
    run_parser.add_argument(
        "--benchmark",
        default="benchmarks/public_benchmark.json",
        help="Path to benchmark configuration JSON",
    )
    run_parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory to write benchmark artifacts",
    )
    run_parser.add_argument(
        "--dashboard-out",
        default="dashboard/public/index.html",
        help="Output path for generated dashboard HTML",
    )
    run_parser.add_argument(
        "--execution-mode",
        default="real",
        choices=["real", "synthetic"],
        help="Execution mode for benchmark runs. 'real' uses external LLM APIs.",
    )
    run_parser.set_defaults(func=run_benchmark)

    dashboard_parser = subparsers.add_parser("dashboard", help="Render dashboard from existing report")
    dashboard_parser.add_argument("--report", required=True, help="Path to benchmark JSON report")
    dashboard_parser.add_argument(
        "--output",
        default="dashboard/public/index.html",
        help="Output path for dashboard HTML",
    )
    dashboard_parser.set_defaults(func=render_dashboard)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
