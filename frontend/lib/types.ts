export type DatasetSummary = {
  id: string;
  name: string;
  description: string;
  source_path: string;
  format: string;
  approx_documents: number;
  approx_queries: number;
  tags: string[];
  domain: string;
};

export type TechniqueCatalog = {
  chunking_strategies: string[];
  retrieval_backends: Array<"in_memory" | "zvec">;
  supports_logicrag: boolean;
  supports_recursive_retrieval: boolean;
  supports_graph_augmentation: boolean;
  supports_hyde: boolean;
  supports_self_rag: boolean;
  supports_query_rewrite: boolean;
  supports_cross_encoder_rerank: boolean;
  supports_temporal_recency_filter: boolean;
  supports_citation_enforcement: boolean;
  supports_claim_verification: boolean;
};

export type UseCasePreset = {
  id: string;
  label: string;
  description: string;
  recommended_dataset_ids: string[];
  recommended_mode: "single" | "ablation";
  recommended_chunking_strategy: "fixed" | "sentence_window" | "semantic" | "adaptive";
  recommended_retrieval_k: number;
  recommended_techniques: {
    logicrag?: boolean;
    recursive_retrieval?: boolean;
    graph_augmentation?: boolean;
    hyde?: boolean;
    self_rag?: boolean;
    query_rewrite?: boolean;
    cross_encoder_rerank?: boolean;
    temporal_recency_filter?: boolean;
    citation_enforcement?: boolean;
    claim_verification?: boolean;
  };
  evaluation_focus: string[];
};

export type WorkflowRunRequest = {
  dataset_id: string;
  workflow_name?: string;
  execution_mode: "real" | "synthetic";
  mode: "single" | "ablation";
  sample_size: number;
  random_seed: number;
  chunking_strategy: "fixed" | "sentence_window" | "semantic" | "adaptive";
  retrieval_backend: "in_memory" | "zvec";
  retrieval_k: number;
  logicrag: boolean;
  recursive_retrieval: boolean;
  graph_augmentation: boolean;
  hyde: boolean;
  self_rag: boolean;
  query_rewrite: boolean;
  cross_encoder_rerank: boolean;
  temporal_recency_filter: boolean;
  citation_enforcement: boolean;
  claim_verification: boolean;
  include_baseline: boolean;
  target_hallucination_rate: number;
  use_case_id?: string;
};

export type WorkflowRunSummary = {
  run_id: string;
  workflow_name: string;
  dataset_id: string;
  dataset_name: string;
  mode: string;
  created_at: string;
  strategy_count: number;
  best_strategy: string;
  best_quality_score: number;
};

export type StrategyAggregate = {
  strategy_name: string;
  chunking_strategy: string;
  retrieval_backend: string;
  logicrag: boolean;
  recursive_retrieval: boolean;
  graph_augmentation: boolean;
  hyde: boolean;
  self_rag: boolean;
  query_rewrite: boolean;
  cross_encoder_rerank: boolean;
  temporal_recency_filter: boolean;
  citation_enforcement: boolean;
  claim_verification: boolean;
  retrieval_k: number;
  avg_precision_at_k: number;
  avg_recall_at_k: number;
  mrr: number;
  avg_hallucination_score: number;
  avg_hallucination_rate: number;
  avg_latency_ms: number;
  avg_quality_score: number;
  avg_prompt_tokens: number;
  avg_completion_tokens: number;
  total_tokens: number;
  total_token_cost_usd: number;
  avg_token_cost_usd: number;
};

export type StrategyResult = {
  aggregate: StrategyAggregate;
  case_runs: Array<{
    query_id: string;
    answer: string;
    contexts: string[];
  }>;
  case_metrics: Array<{
    query_id: string;
    hallucination_score: number;
    hallucination_rate: number;
    total_latency_ms: number;
    token_cost_usd: number;
    quality_score: number;
  }>;
};

export type WorkflowRunDetail = {
  summary: WorkflowRunSummary;
  request: WorkflowRunRequest;
  report: {
    benchmark_name: string;
    run_id: string;
    created_at: string;
    strategy_results: StrategyResult[];
  };
  governance: {
    unsupported_claims: Array<{
      strategy_name: string;
      query_id: string;
      query: string;
      intent: string;
      claim: string;
      support_ratio: number;
      contexts: string[];
    }>;
    intent_hallucination: Array<{
      strategy_name: string;
      intent: string;
      avg_grounded_claim_ratio: number;
      avg_hallucination_rate: number;
      case_count: number;
    }>;
    latency_slo: Array<{
      strategy_name: string;
      p50_latency_ms: number;
      p95_latency_ms: number;
      avg_latency_ms: number;
    }>;
    cost_per_quality: Array<{
      strategy_name: string;
      avg_quality_score: number;
      total_token_cost_usd: number;
      cost_per_quality_point: number;
    }>;
    safety_violations: Array<{
      strategy_name: string;
      query_id: string;
      triggered_rules: string[];
      answer_excerpt: string;
    }>;
    hallucination_target: {
      target_hallucination_rate: number;
      all_passed: boolean;
      passing_count: number;
      total_count: number;
      has_passing_strategy: boolean;
      strategies: Array<{
        strategy_name: string;
        avg_hallucination_rate: number;
        target_hallucination_rate: number;
        passes_target: boolean;
      }>;
    };
  };
};
