from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvaluationCase:
    query_id: str
    query: str
    reference_answer: str
    ground_truth_doc_ids: list[str]


@dataclass(frozen=True)
class StrategyConfig:
    name: str
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
    citation_enforcement: bool = False
    claim_verification: bool = False
    prompt_token_price_per_1k: float = 0.003
    completion_token_price_per_1k: float = 0.015
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyConfig":
        return cls(
            name=data["name"],
            chunking_strategy=data["chunking_strategy"],
            retrieval_k=int(data.get("retrieval_k", 5)),
            retrieval_backend=str(data.get("retrieval_backend", "in_memory")),
            logicrag=bool(data.get("logicrag", False)),
            recursive_retrieval=bool(data.get("recursive_retrieval", False)),
            graph_augmentation=bool(data.get("graph_augmentation", False)),
            hyde=bool(data.get("hyde", False)),
            self_rag=bool(data.get("self_rag", False)),
            query_rewrite=bool(data.get("query_rewrite", False)),
            cross_encoder_rerank=bool(data.get("cross_encoder_rerank", False)),
            temporal_recency_filter=bool(data.get("temporal_recency_filter", False)),
            citation_enforcement=bool(data.get("citation_enforcement", False)),
            claim_verification=bool(data.get("claim_verification", False)),
            prompt_token_price_per_1k=float(data.get("prompt_token_price_per_1k", 0.003)),
            completion_token_price_per_1k=float(data.get("completion_token_price_per_1k", 0.015)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class ClaimVerification:
    claim: str
    supported: bool
    confidence: float
    evidence: str = ""
    reasoning: str = ""


@dataclass(frozen=True)
class CaseRun:
    query_id: str
    retrieved_doc_ids: list[str]
    contexts: list[str]
    answer: str
    prompt_tokens: int
    completion_tokens: int
    retrieval_latency_ms: float
    generation_latency_ms: float
    claim_verifications: list[ClaimVerification] = field(default_factory=list)
    citation_ids: list[str] = field(default_factory=list)
    execution_mode: str = "synthetic"

    @property
    def total_latency_ms(self) -> float:
        return self.retrieval_latency_ms + self.generation_latency_ms


@dataclass(frozen=True)
class CaseMetrics:
    query_id: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    hallucination_score: float
    hallucination_rate: float
    total_latency_ms: float
    token_cost_usd: float
    quality_score: float


@dataclass(frozen=True)
class StrategyAggregate:
    strategy_name: str
    chunking_strategy: str
    retrieval_backend: str
    logicrag: bool
    recursive_retrieval: bool
    graph_augmentation: bool
    hyde: bool
    self_rag: bool
    query_rewrite: bool
    cross_encoder_rerank: bool
    temporal_recency_filter: bool
    citation_enforcement: bool
    claim_verification: bool
    retrieval_k: int
    avg_precision_at_k: float
    avg_recall_at_k: float
    mrr: float
    avg_hallucination_score: float
    avg_hallucination_rate: float
    avg_latency_ms: float
    avg_quality_score: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    total_tokens: int
    total_token_cost_usd: float
    avg_token_cost_usd: float


@dataclass(frozen=True)
class StrategyResult:
    config: StrategyConfig
    case_runs: list[CaseRun]
    case_metrics: list[CaseMetrics]
    aggregate: StrategyAggregate


@dataclass(frozen=True)
class BenchmarkReport:
    benchmark_name: str
    run_id: str
    created_at: str
    dataset_path: str
    strategy_results: list[StrategyResult]

    @classmethod
    def create(
        cls,
        benchmark_name: str,
        dataset_path: str,
        strategy_results: list[StrategyResult],
    ) -> "BenchmarkReport":
        ts = datetime.now(timezone.utc)
        run_id = ts.strftime("%Y%m%dT%H%M%SZ")
        return cls(
            benchmark_name=benchmark_name,
            run_id=run_id,
            created_at=ts.isoformat(),
            dataset_path=dataset_path,
            strategy_results=strategy_results,
        )


@dataclass(frozen=True)
class EvaluationDataset:
    name: str
    documents: list[Document]
    cases: list[EvaluationCase]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationDataset":
        docs = [Document(**doc) for doc in data.get("documents", [])]
        cases = [EvaluationCase(**case) for case in data.get("cases", [])]
        return cls(name=data.get("name", "dataset"), documents=docs, cases=cases)

    def to_doc_index(self) -> dict[str, Document]:
        return {doc.doc_id: doc for doc in self.documents}


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark_name: str
    dataset_path: str
    random_seed: int
    strategies: list[StrategyConfig]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkConfig":
        strategies = [StrategyConfig.from_dict(item) for item in data.get("strategies", [])]
        return cls(
            benchmark_name=data.get("benchmark_name", "rag-benchmark"),
            dataset_path=data["dataset_path"],
            random_seed=int(data.get("random_seed", 42)),
            strategies=strategies,
        )


def dataclass_to_dict(value: Any) -> Any:
    if isinstance(value, list):
        return [dataclass_to_dict(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {key: dataclass_to_dict(item) for key, item in asdict(value).items()}
    return value
