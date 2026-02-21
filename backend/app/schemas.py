from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    id: str
    name: str
    description: str
    source_path: str
    format: str
    approx_documents: int
    approx_queries: int
    tags: list[str] = Field(default_factory=list)
    domain: str = "general"


class TechniqueCatalog(BaseModel):
    chunking_strategies: list[str]
    retrieval_backends: list[str] = Field(default_factory=lambda: ["in_memory", "zvec"])
    supports_logicrag: bool = True
    supports_recursive_retrieval: bool = True
    supports_graph_augmentation: bool = True
    supports_hyde: bool = True
    supports_self_rag: bool = True
    supports_query_rewrite: bool = True
    supports_cross_encoder_rerank: bool = True
    supports_temporal_recency_filter: bool = True
    supports_citation_enforcement: bool = True
    supports_claim_verification: bool = True


class UseCasePreset(BaseModel):
    id: str
    label: str
    description: str
    recommended_dataset_ids: list[str]
    recommended_mode: Literal["single", "ablation"] = "ablation"
    recommended_chunking_strategy: str = "semantic"
    recommended_retrieval_k: int = 6
    recommended_techniques: dict[str, bool]
    evaluation_focus: list[str]


class WorkflowRunRequest(BaseModel):
    dataset_id: str
    workflow_name: str | None = None
    execution_mode: Literal["real", "synthetic"] = "real"
    mode: Literal["single", "ablation"] = "ablation"
    sample_size: int = Field(default=80, ge=10, le=500)
    random_seed: int = Field(default=42, ge=1, le=999999)
    chunking_strategy: Literal["fixed", "sentence_window", "semantic", "adaptive"] = "semantic"
    retrieval_backend: Literal["in_memory", "zvec"] = "in_memory"
    retrieval_k: int = Field(default=6, ge=2, le=20)
    logicrag: bool = False
    recursive_retrieval: bool = True
    graph_augmentation: bool = False
    hyde: bool = False
    self_rag: bool = False
    query_rewrite: bool = True
    cross_encoder_rerank: bool = True
    temporal_recency_filter: bool = False
    citation_enforcement: bool = True
    claim_verification: bool = True
    include_baseline: bool = True
    target_hallucination_rate: float = Field(default=0.01, ge=0.0, le=0.5)
    use_case_id: str | None = None


class WorkflowRunSummary(BaseModel):
    run_id: str
    workflow_name: str
    dataset_id: str
    dataset_name: str
    mode: str
    created_at: datetime
    strategy_count: int
    best_strategy: str
    best_quality_score: float


class WorkflowRunDetail(BaseModel):
    summary: WorkflowRunSummary
    request: WorkflowRunRequest
    report: dict[str, Any]
    governance: dict[str, Any]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    timestamp: datetime
