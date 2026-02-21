from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .cost import token_cost_usd
from .metrics import (
    composite_quality_score,
    hallucination_detection_score,
    mean_metric,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .models import (
    BenchmarkConfig,
    BenchmarkReport,
    CaseMetrics,
    CaseRun,
    EvaluationCase,
    StrategyConfig,
    StrategyAggregate,
    StrategyResult,
)


class CaseRunner(Protocol):
    def run_case(self, case: EvaluationCase, strategy: StrategyConfig) -> CaseRun:
        ...


@dataclass
class BenchmarkEvaluator:
    runner: CaseRunner

    def evaluate(self, config: BenchmarkConfig) -> BenchmarkReport:
        strategy_results: list[StrategyResult] = []

        for strategy in config.strategies:
            case_runs = [
                self.runner.run_case(case=case, strategy=strategy)
                for case in self.runner.dataset.cases
            ]

            case_metrics: list[CaseMetrics] = []
            for case, case_run in zip(self.runner.dataset.cases, case_runs):
                precision = precision_at_k(
                    retrieved_doc_ids=case_run.retrieved_doc_ids,
                    relevant_doc_ids=case.ground_truth_doc_ids,
                    k=strategy.retrieval_k,
                )
                recall = recall_at_k(
                    retrieved_doc_ids=case_run.retrieved_doc_ids,
                    relevant_doc_ids=case.ground_truth_doc_ids,
                    k=strategy.retrieval_k,
                )
                mrr = reciprocal_rank(
                    retrieved_doc_ids=case_run.retrieved_doc_ids,
                    relevant_doc_ids=case.ground_truth_doc_ids,
                )
                hallucination_score = hallucination_detection_score(
                    answer=case_run.answer,
                    contexts=case_run.contexts,
                )
                if case_run.claim_verifications:
                    supported = sum(1 for item in case_run.claim_verifications if item.supported)
                    total = len(case_run.claim_verifications)
                    hallucination_score = supported / float(total) if total > 0 else 0.0
                cost = token_cost_usd(
                    prompt_tokens=case_run.prompt_tokens,
                    completion_tokens=case_run.completion_tokens,
                    prompt_price_per_1k=strategy.prompt_token_price_per_1k,
                    completion_price_per_1k=strategy.completion_token_price_per_1k,
                )
                quality = composite_quality_score(
                    precision=precision,
                    recall=recall,
                    mrr=mrr,
                    hallucination_score=hallucination_score,
                )
                hallucination_rate = max(0.0, 1.0 - hallucination_score)

                case_metrics.append(
                    CaseMetrics(
                        query_id=case.query_id,
                        precision_at_k=precision,
                        recall_at_k=recall,
                        reciprocal_rank=mrr,
                        hallucination_score=hallucination_score,
                        hallucination_rate=hallucination_rate,
                        total_latency_ms=case_run.total_latency_ms,
                        token_cost_usd=cost,
                        quality_score=quality,
                    )
                )

            aggregate = self._aggregate(
                strategy_name=strategy.name,
                strategy=strategy,
                case_metrics=case_metrics,
                case_runs=case_runs,
            )
            strategy_results.append(
                StrategyResult(
                    config=strategy,
                    case_runs=case_runs,
                    case_metrics=case_metrics,
                    aggregate=aggregate,
                )
            )

        return BenchmarkReport.create(
            benchmark_name=config.benchmark_name,
            dataset_path=config.dataset_path,
            strategy_results=strategy_results,
        )

    def _aggregate(
        self,
        strategy_name: str,
        strategy: StrategyConfig,
        case_metrics: list[CaseMetrics],
        case_runs: list[CaseRun],
    ) -> StrategyAggregate:
        precision_values = [item.precision_at_k for item in case_metrics]
        recall_values = [item.recall_at_k for item in case_metrics]
        mrr_values = [item.reciprocal_rank for item in case_metrics]
        hallucination_values = [item.hallucination_score for item in case_metrics]
        hallucination_rates = [item.hallucination_rate for item in case_metrics]
        latency_values = [item.total_latency_ms for item in case_metrics]
        quality_values = [item.quality_score for item in case_metrics]
        cost_values = [item.token_cost_usd for item in case_metrics]
        prompt_tokens = [item.prompt_tokens for item in case_runs]
        completion_tokens = [item.completion_tokens for item in case_runs]

        total_cost = sum(cost_values)
        avg_cost = total_cost / len(cost_values) if cost_values else 0.0
        total_tokens = sum(prompt_tokens) + sum(completion_tokens)

        return StrategyAggregate(
            strategy_name=strategy_name,
            chunking_strategy=strategy.chunking_strategy,
            retrieval_backend=strategy.retrieval_backend,
            logicrag=strategy.logicrag,
            recursive_retrieval=strategy.recursive_retrieval,
            graph_augmentation=strategy.graph_augmentation,
            hyde=strategy.hyde,
            self_rag=strategy.self_rag,
            query_rewrite=strategy.query_rewrite,
            cross_encoder_rerank=strategy.cross_encoder_rerank,
            temporal_recency_filter=strategy.temporal_recency_filter,
            citation_enforcement=strategy.citation_enforcement,
            claim_verification=strategy.claim_verification,
            retrieval_k=strategy.retrieval_k,
            avg_precision_at_k=mean_metric(precision_values),
            avg_recall_at_k=mean_metric(recall_values),
            mrr=mean_metric(mrr_values),
            avg_hallucination_score=mean_metric(hallucination_values),
            avg_hallucination_rate=mean_metric(hallucination_rates),
            avg_latency_ms=mean_metric(latency_values),
            avg_quality_score=mean_metric(quality_values),
            avg_prompt_tokens=mean_metric(prompt_tokens),
            avg_completion_tokens=mean_metric(completion_tokens),
            total_tokens=total_tokens,
            total_token_cost_usd=total_cost,
            avg_token_cost_usd=avg_cost,
        )
