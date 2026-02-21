from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analysis import (
    build_cost_per_quality,
    build_intent_breakdown,
    build_latency_slo_snapshot,
    build_safety_violations,
    extract_unsupported_claims,
)
from .bootstrap import ensure_src_path
from .datasets import DatasetRegistry
from .schemas import (
    TechniqueCatalog,
    UseCasePreset,
    WorkflowRunDetail,
    WorkflowRunRequest,
    WorkflowRunSummary,
)

ROOT = ensure_src_path()

from rag_eval.evaluator import BenchmarkEvaluator  # noqa: E402
from rag_eval.logicrag_client import LogicRAGClient  # noqa: E402
from rag_eval.models import BenchmarkConfig, StrategyConfig, dataclass_to_dict  # noqa: E402
from rag_eval.providers import OpenAIProvider  # noqa: E402
from rag_eval.real_runner import RealRAGSystem  # noqa: E402
from rag_eval.runtime import RuntimeSettings  # noqa: E402
from rag_eval.strategies import SyntheticRAGSystem  # noqa: E402


class WorkflowService:
    def __init__(self, artifacts_dir: Path | None = None) -> None:
        self.artifacts_dir = (artifacts_dir or (ROOT / "artifacts" / "workflows")).resolve()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def technique_catalog(self) -> TechniqueCatalog:
        return TechniqueCatalog(
            chunking_strategies=["fixed", "sentence_window", "semantic", "adaptive"],
            retrieval_backends=["in_memory", "zvec"],
        )

    def use_case_presets(self) -> list[UseCasePreset]:
        return [
            UseCasePreset(
                id="regulated_member_support",
                label="Regulated Member Support Copilot",
                description=(
                    "High-stakes benefits and claims assistant where hallucination and refusal quality are release gates."
                ),
                recommended_dataset_ids=["ragcare_qa", "retrievalqa"],
                recommended_mode="ablation",
                recommended_chunking_strategy="semantic",
                recommended_retrieval_k=7,
                recommended_techniques={
                    "logicrag": True,
                    "recursive_retrieval": True,
                    "graph_augmentation": True,
                    "hyde": False,
                    "self_rag": True,
                    "query_rewrite": True,
                    "cross_encoder_rerank": True,
                    "temporal_recency_filter": False,
                    "citation_enforcement": True,
                    "claim_verification": True,
                },
                evaluation_focus=[
                    "Hallucination rate by intent",
                    "Refusal quality and safety boundaries",
                    "Latency p95 vs SLO",
                ],
            ),
            UseCasePreset(
                id="enterprise_policy_truth_machine",
                label="Internal Policy Truth Machine",
                description="Enterprise HR/compliance policy retrieval with citation strictness and regression gates.",
                recommended_dataset_ids=["open_ragbench", "retrievalqa", "public_eval_set"],
                recommended_mode="ablation",
                recommended_chunking_strategy="semantic",
                recommended_retrieval_k=6,
                recommended_techniques={
                    "logicrag": True,
                    "recursive_retrieval": True,
                    "graph_augmentation": True,
                    "hyde": True,
                    "self_rag": False,
                    "query_rewrite": True,
                    "cross_encoder_rerank": True,
                    "temporal_recency_filter": True,
                    "citation_enforcement": True,
                    "claim_verification": True,
                },
                evaluation_focus=[
                    "Precision@k and MRR",
                    "Citation faithfulness",
                    "No-regression governance thresholds",
                ],
            ),
            UseCasePreset(
                id="finance_research_copilot",
                label="Finance Research Copilot",
                description="Portfolio memo assistant requiring numeric faithfulness, drift awareness, and cost discipline.",
                recommended_dataset_ids=["finder", "open_ragbench", "paperzilla"],
                recommended_mode="ablation",
                recommended_chunking_strategy="adaptive",
                recommended_retrieval_k=8,
                recommended_techniques={
                    "logicrag": True,
                    "recursive_retrieval": True,
                    "graph_augmentation": True,
                    "hyde": True,
                    "self_rag": True,
                    "query_rewrite": True,
                    "cross_encoder_rerank": True,
                    "temporal_recency_filter": True,
                    "citation_enforcement": True,
                    "claim_verification": True,
                },
                evaluation_focus=[
                    "Claim-level grounding",
                    "Numeric consistency",
                    "Cost per useful answer",
                ],
            ),
        ]

    def run_workflow(self, request: WorkflowRunRequest, dataset_registry: DatasetRegistry) -> WorkflowRunDetail:
        loaded = dataset_registry.load_dataset(
            dataset_id=request.dataset_id,
            sample_size=request.sample_size,
            seed=request.random_seed,
        )

        if not loaded.dataset.cases or not loaded.dataset.documents:
            raise ValueError(f"Dataset '{request.dataset_id}' produced no evaluation cases or documents")

        strategies = self._build_strategies(request)
        benchmark_name = request.workflow_name or f"workflow_{request.dataset_id}_{request.mode}"

        config = BenchmarkConfig(
            benchmark_name=benchmark_name,
            dataset_path=str(loaded.record.source_path),
            random_seed=request.random_seed,
            strategies=strategies,
        )
        runner = self._build_runner(
            dataset=loaded.dataset,
            execution_mode=request.execution_mode,
            seed=request.random_seed,
        )
        evaluator = BenchmarkEvaluator(runner=runner)
        report = evaluator.evaluate(config)
        report_payload = dataclass_to_dict(report)

        query_lookup: dict[str, dict[str, Any]] = {}
        for case in loaded.dataset.cases:
            query_lookup[case.query_id] = {
                "query": case.query,
                **loaded.query_metadata.get(case.query_id, {}),
            }

        governance = {
            "unsupported_claims": extract_unsupported_claims(report_payload, query_lookup),
            "intent_hallucination": build_intent_breakdown(report_payload, query_lookup),
            "latency_slo": build_latency_slo_snapshot(report_payload),
            "cost_per_quality": build_cost_per_quality(report_payload),
            "safety_violations": build_safety_violations(report_payload, query_lookup),
            "hallucination_target": self._hallucination_target_snapshot(
                report_payload=report_payload,
                target=request.target_hallucination_rate,
            ),
        }

        summary = self._to_summary(
            report_payload=report_payload,
            request=request,
            dataset_name=loaded.record.name,
        )
        detail = WorkflowRunDetail(
            summary=summary,
            request=request,
            report=report_payload,
            governance=governance,
        )

        self._persist_run(detail)
        return detail

    def _build_runner(self, dataset, execution_mode: str, seed: int):
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

    def list_runs(self, limit: int = 20) -> list[WorkflowRunSummary]:
        summaries: list[WorkflowRunSummary] = []
        run_files = sorted(self.artifacts_dir.glob("*.json"), reverse=True)

        for path in run_files:
            if path.name == "latest.json":
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                summary = payload.get("summary")
                if not isinstance(summary, dict):
                    continue
                summaries.append(WorkflowRunSummary.model_validate(summary))
            except Exception:
                continue
            if len(summaries) >= limit:
                break

        return summaries

    def get_run(self, run_id: str) -> WorkflowRunDetail:
        run_path = self.artifacts_dir / f"{run_id}.json"
        if not run_path.exists():
            raise FileNotFoundError(f"Run '{run_id}' not found")

        payload = json.loads(run_path.read_text(encoding="utf-8"))
        return WorkflowRunDetail.model_validate(payload)

    def _persist_run(self, detail: WorkflowRunDetail) -> None:
        out = detail.model_dump(mode="json")
        run_id = detail.summary.run_id
        run_path = self.artifacts_dir / f"{run_id}.json"
        latest_path = self.artifacts_dir / "latest.json"

        run_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        latest_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    def _to_summary(
        self,
        report_payload: dict[str, Any],
        request: WorkflowRunRequest,
        dataset_name: str,
    ) -> WorkflowRunSummary:
        aggregates = [item.get("aggregate", {}) for item in report_payload.get("strategy_results", [])]
        if aggregates:
            best = max(aggregates, key=lambda item: item.get("avg_quality_score", 0.0))
            best_strategy = str(best.get("strategy_name", "unknown"))
            best_score = float(best.get("avg_quality_score", 0.0))
        else:
            best_strategy = "unknown"
            best_score = 0.0

        created_at = datetime.fromisoformat(report_payload["created_at"])
        return WorkflowRunSummary(
            run_id=report_payload["run_id"],
            workflow_name=report_payload["benchmark_name"],
            dataset_id=request.dataset_id,
            dataset_name=dataset_name,
            mode=request.mode,
            created_at=created_at,
            strategy_count=len(aggregates),
            best_strategy=best_strategy,
            best_quality_score=best_score,
        )

    def _build_strategies(self, request: WorkflowRunRequest) -> list[StrategyConfig]:
        if request.mode == "single":
            selected = StrategyConfig(
                name="selected_stack",
                chunking_strategy=request.chunking_strategy,
                retrieval_backend=request.retrieval_backend,
                retrieval_k=request.retrieval_k,
                logicrag=request.logicrag,
                recursive_retrieval=request.recursive_retrieval,
                graph_augmentation=request.graph_augmentation,
                hyde=request.hyde,
                self_rag=request.self_rag,
                query_rewrite=request.query_rewrite,
                cross_encoder_rerank=request.cross_encoder_rerank,
                temporal_recency_filter=request.temporal_recency_filter,
                citation_enforcement=request.citation_enforcement,
                claim_verification=request.claim_verification,
            )
            strategies = [selected]
            if request.include_baseline:
                baseline = StrategyConfig(
                    name="baseline_fixed",
                    chunking_strategy="fixed",
                    retrieval_backend=request.retrieval_backend,
                    retrieval_k=request.retrieval_k,
                )
                strategies.insert(0, baseline)
            return self._dedupe_strategies(strategies)

        strategies: list[StrategyConfig] = []
        if request.include_baseline:
            strategies.append(
                StrategyConfig(
                    name="baseline_fixed",
                    chunking_strategy="fixed",
                    retrieval_backend=request.retrieval_backend,
                    retrieval_k=request.retrieval_k,
                )
            )

        current = StrategyConfig(
            name="chunking_only",
            chunking_strategy=request.chunking_strategy,
            retrieval_backend=request.retrieval_backend,
            retrieval_k=request.retrieval_k,
            logicrag=False,
            recursive_retrieval=False,
            graph_augmentation=False,
            hyde=False,
            self_rag=False,
        )
        strategies.append(current)

        if request.query_rewrite:
            current = StrategyConfig(
                name="query_rewrite",
                chunking_strategy=request.chunking_strategy,
                retrieval_backend=request.retrieval_backend,
                retrieval_k=request.retrieval_k,
                logicrag=False,
                query_rewrite=True,
            )
            strategies.append(current)

        if request.recursive_retrieval:
            current = StrategyConfig(
                name="recursive_retrieval",
                chunking_strategy=request.chunking_strategy,
                retrieval_backend=request.retrieval_backend,
                retrieval_k=request.retrieval_k,
                logicrag=False,
                query_rewrite=request.query_rewrite,
                recursive_retrieval=True,
                graph_augmentation=False,
                hyde=False,
                self_rag=False,
            )
            strategies.append(current)

        if request.cross_encoder_rerank:
            current = StrategyConfig(
                name="cross_encoder_rerank",
                chunking_strategy=request.chunking_strategy,
                retrieval_backend=request.retrieval_backend,
                retrieval_k=request.retrieval_k,
                logicrag=False,
                query_rewrite=request.query_rewrite,
                recursive_retrieval=request.recursive_retrieval,
                cross_encoder_rerank=True,
            )
            strategies.append(current)

        if request.logicrag:
            current = StrategyConfig(
                name="logicrag",
                chunking_strategy=request.chunking_strategy,
                retrieval_backend=request.retrieval_backend,
                retrieval_k=request.retrieval_k,
                logicrag=True,
                query_rewrite=request.query_rewrite,
                cross_encoder_rerank=request.cross_encoder_rerank,
                citation_enforcement=request.citation_enforcement,
            )
            strategies.append(current)

        if request.graph_augmentation:
            current = StrategyConfig(
                name="graph_augmented",
                chunking_strategy=request.chunking_strategy,
                retrieval_backend=request.retrieval_backend,
                retrieval_k=request.retrieval_k,
                logicrag=request.logicrag,
                query_rewrite=request.query_rewrite,
                recursive_retrieval=request.recursive_retrieval,
                cross_encoder_rerank=request.cross_encoder_rerank,
                graph_augmentation=True,
                hyde=False,
                self_rag=False,
            )
            strategies.append(current)

        if request.hyde:
            current = StrategyConfig(
                name="hyde_augmented",
                chunking_strategy=request.chunking_strategy,
                retrieval_backend=request.retrieval_backend,
                retrieval_k=request.retrieval_k,
                logicrag=request.logicrag,
                query_rewrite=request.query_rewrite,
                recursive_retrieval=request.recursive_retrieval,
                cross_encoder_rerank=request.cross_encoder_rerank,
                graph_augmentation=request.graph_augmentation,
                hyde=True,
                self_rag=False,
            )
            strategies.append(current)

        if request.temporal_recency_filter:
            current = StrategyConfig(
                name="temporal_filter",
                chunking_strategy=request.chunking_strategy,
                retrieval_backend=request.retrieval_backend,
                retrieval_k=request.retrieval_k,
                logicrag=request.logicrag,
                query_rewrite=request.query_rewrite,
                recursive_retrieval=request.recursive_retrieval,
                cross_encoder_rerank=request.cross_encoder_rerank,
                graph_augmentation=request.graph_augmentation,
                hyde=request.hyde,
                temporal_recency_filter=True,
            )
            strategies.append(current)

        if request.self_rag:
            current = StrategyConfig(
                name="self_rag_stack",
                chunking_strategy=request.chunking_strategy,
                retrieval_backend=request.retrieval_backend,
                retrieval_k=request.retrieval_k,
                logicrag=request.logicrag,
                query_rewrite=request.query_rewrite,
                recursive_retrieval=request.recursive_retrieval,
                cross_encoder_rerank=request.cross_encoder_rerank,
                graph_augmentation=request.graph_augmentation,
                hyde=request.hyde,
                temporal_recency_filter=request.temporal_recency_filter,
                self_rag=True,
            )
            strategies.append(current)

        if request.citation_enforcement:
            current = StrategyConfig(
                name="citation_enforced",
                chunking_strategy=request.chunking_strategy,
                retrieval_backend=request.retrieval_backend,
                retrieval_k=request.retrieval_k,
                logicrag=request.logicrag,
                query_rewrite=request.query_rewrite,
                recursive_retrieval=request.recursive_retrieval,
                cross_encoder_rerank=request.cross_encoder_rerank,
                graph_augmentation=request.graph_augmentation,
                hyde=request.hyde,
                temporal_recency_filter=request.temporal_recency_filter,
                self_rag=request.self_rag,
                citation_enforcement=True,
            )
            strategies.append(current)

        if request.claim_verification:
            current = StrategyConfig(
                name="claim_verified",
                chunking_strategy=request.chunking_strategy,
                retrieval_backend=request.retrieval_backend,
                retrieval_k=request.retrieval_k,
                logicrag=request.logicrag,
                query_rewrite=request.query_rewrite,
                recursive_retrieval=request.recursive_retrieval,
                cross_encoder_rerank=request.cross_encoder_rerank,
                graph_augmentation=request.graph_augmentation,
                hyde=request.hyde,
                temporal_recency_filter=request.temporal_recency_filter,
                self_rag=request.self_rag,
                citation_enforcement=request.citation_enforcement,
                claim_verification=True,
            )
            strategies.append(current)

        target = StrategyConfig(
            name="target_stack",
            chunking_strategy=request.chunking_strategy,
            retrieval_backend=request.retrieval_backend,
            retrieval_k=request.retrieval_k,
            logicrag=request.logicrag,
            recursive_retrieval=request.recursive_retrieval,
            graph_augmentation=request.graph_augmentation,
            hyde=request.hyde,
            self_rag=request.self_rag,
            query_rewrite=request.query_rewrite,
            cross_encoder_rerank=request.cross_encoder_rerank,
            temporal_recency_filter=request.temporal_recency_filter,
            citation_enforcement=request.citation_enforcement,
            claim_verification=request.claim_verification,
        )
        strategies.append(target)
        return self._dedupe_strategies(strategies)

    def _hallucination_target_snapshot(self, report_payload: dict[str, Any], target: float) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for strategy_result in report_payload.get("strategy_results", []):
            aggregate = strategy_result.get("aggregate", {})
            hallucination_rate = float(aggregate.get("avg_hallucination_rate", 1.0))
            rows.append(
                {
                    "strategy_name": str(aggregate.get("strategy_name", "unknown")),
                    "avg_hallucination_rate": hallucination_rate,
                    "target_hallucination_rate": target,
                    "passes_target": hallucination_rate <= target,
                }
            )

        rows.sort(key=lambda item: item["avg_hallucination_rate"])
        passing_count = sum(1 for item in rows if item["passes_target"])
        return {
            "target_hallucination_rate": target,
            "strategies": rows,
            "all_passed": all(item["passes_target"] for item in rows) if rows else False,
            "passing_count": passing_count,
            "total_count": len(rows),
            "has_passing_strategy": passing_count > 0,
        }

    def _dedupe_strategies(self, strategies: list[StrategyConfig]) -> list[StrategyConfig]:
        deduped: list[StrategyConfig] = []
        seen: set[tuple[Any, ...]] = set()

        for strategy in strategies:
            key = (
                strategy.chunking_strategy,
                strategy.retrieval_backend,
                strategy.retrieval_k,
                strategy.logicrag,
                strategy.recursive_retrieval,
                strategy.graph_augmentation,
                strategy.hyde,
                strategy.self_rag,
                strategy.query_rewrite,
                strategy.cross_encoder_rerank,
                strategy.temporal_recency_filter,
                strategy.citation_enforcement,
                strategy.claim_verification,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(strategy)

        return deduped
