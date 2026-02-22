"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

import {
  getDatasets,
  getTechniques,
  getUseCases,
  getWorkflowRun,
  getWorkflowRuns,
  runWorkflow,
} from "../lib/api";
import {
  DatasetSummary,
  TechniqueCatalog,
  UseCasePreset,
  WorkflowRunDetail,
  WorkflowRunRequest,
  WorkflowRunSummary,
} from "../lib/types";

const DEFAULT_FORM: WorkflowRunRequest = {
  dataset_id: "",
  workflow_name: "",
  execution_mode: "real",
  mode: "ablation",
  sample_size: 80,
  random_seed: 42,
  chunking_strategy: "semantic",
  retrieval_backend: "in_memory",
  retrieval_k: 6,
  logicrag: false,
  recursive_retrieval: true,
  graph_augmentation: false,
  hyde: false,
  self_rag: false,
  query_rewrite: true,
  cross_encoder_rerank: true,
  temporal_recency_filter: false,
  citation_enforcement: true,
  claim_verification: true,
  include_baseline: true,
  target_hallucination_rate: 0.01,
  use_case_id: undefined,
};

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function usd(value: number): string {
  return `$${value.toFixed(4)}`;
}

function ms(value: number): string {
  return `${value.toFixed(1)} ms`;
}

function metricValue(values: number[], fallback = 0): number {
  if (values.length === 0) return fallback;
  return values.reduce((acc, value) => acc + value, 0) / values.length;
}

function messageForError(error: unknown, fallback: string): string {
  if (error instanceof Error) {
    return error.message;
  }
  return fallback;
}

export default function DashboardApp() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [techniques, setTechniques] = useState<TechniqueCatalog | null>(null);
  const [useCases, setUseCases] = useState<UseCasePreset[]>([]);
  const [runs, setRuns] = useState<WorkflowRunSummary[]>([]);

  const [form, setForm] = useState<WorkflowRunRequest>(DEFAULT_FORM);
  const [activeRun, setActiveRun] = useState<WorkflowRunDetail | null>(null);

  const [loading, setLoading] = useState<boolean>(true);
  const [running, setRunning] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    async function bootstrap() {
      setLoading(true);
      setError("");
      const issues: string[] = [];

      const [dsResult, techniquesResult, useCasesResult, runsResult] = await Promise.allSettled([
        getDatasets(),
        getTechniques(),
        getUseCases(),
        getWorkflowRuns(),
      ]);

      if (dsResult.status === "fulfilled") {
        const ds = dsResult.value;
        setDatasets(ds);
        const defaultDataset = ds[0]?.id ?? "";
        setForm((prev) => ({ ...prev, dataset_id: prev.dataset_id || defaultDataset }));
      } else {
        setDatasets([]);
        issues.push(`Datasets: ${messageForError(dsResult.reason, "Failed to load datasets.")}`);
      }

      if (techniquesResult.status === "fulfilled") {
        setTechniques(techniquesResult.value);
      } else {
        setTechniques(null);
        issues.push(
          `Techniques: ${messageForError(techniquesResult.reason, "Failed to load technique catalog.")}`,
        );
      }

      if (useCasesResult.status === "fulfilled") {
        setUseCases(useCasesResult.value);
      } else {
        setUseCases([]);
        issues.push(`Use cases: ${messageForError(useCasesResult.reason, "Failed to load use-case presets.")}`);
      }

      if (runsResult.status === "fulfilled") {
        const history = runsResult.value;
        setRuns(history);
        if (history.length > 0) {
          try {
            const latest = await getWorkflowRun(history[0].run_id);
            setActiveRun(latest);
          } catch (error) {
            issues.push(`Latest run: ${messageForError(error, "Failed to load latest run detail.")}`);
          }
        }
      } else {
        setRuns([]);
        issues.push(`Workflow runs: ${messageForError(runsResult.reason, "Failed to load workflow history.")}`);
      }

      if (issues.length > 0) {
        setError(issues.join(" "));
      }
      setLoading(false);
    }

    bootstrap();
  }, []);

  const selectedUseCase = useMemo(
    () => useCases.find((item) => item.id === form.use_case_id),
    [form.use_case_id, useCases],
  );

  const aggregates = useMemo(() => {
    if (!activeRun) return [];
    return [...activeRun.report.strategy_results]
      .map((item) => item.aggregate)
      .sort((a, b) => b.avg_quality_score - a.avg_quality_score);
  }, [activeRun]);

  const bestStrategy = aggregates[0];
  const fastestStrategy = useMemo(() => {
    if (aggregates.length === 0) return null;
    return [...aggregates].sort((a, b) => a.avg_latency_ms - b.avg_latency_ms)[0];
  }, [aggregates]);
  const cheapestStrategy = useMemo(() => {
    if (aggregates.length === 0) return null;
    return [...aggregates].sort((a, b) => a.avg_token_cost_usd - b.avg_token_cost_usd)[0];
  }, [aggregates]);
  const hallucinationGate =
    activeRun?.governance.hallucination_target ??
    ({
      target_hallucination_rate: form.target_hallucination_rate,
      all_passed: false,
      passing_count: 0,
      total_count: 0,
      has_passing_strategy: false,
      strategies: [],
    } as const);

  const scatterPoints = useMemo(() => {
    if (aggregates.length === 0) return null;

    const width = 680;
    const height = 300;
    const padding = { left: 56, right: 28, top: 18, bottom: 42 };

    const latencies = aggregates.map((item) => item.avg_latency_ms);
    const qualities = aggregates.map((item) => item.avg_quality_score);

    const minLatency = Math.min(...latencies);
    const maxLatency = Math.max(...latencies);
    const minQuality = Math.min(...qualities);
    const maxQuality = Math.max(...qualities);

    const latencySpan = Math.max(maxLatency - minLatency, 1);
    const qualitySpan = Math.max(maxQuality - minQuality, 0.01);

    const scaleX = (value: number) =>
      padding.left + ((value - minLatency) / latencySpan) * (width - padding.left - padding.right);
    const scaleY = (value: number) =>
      height - padding.bottom - ((value - minQuality) / qualitySpan) * (height - padding.top - padding.bottom);

    const points = aggregates.map((item) => ({
      strategy: item.strategy_name,
      x: scaleX(item.avg_latency_ms),
      y: scaleY(item.avg_quality_score),
      latency: item.avg_latency_ms,
      quality: item.avg_quality_score,
    }));

    return { points, width, height, padding };
  }, [aggregates]);

  function applyUseCase(preset: UseCasePreset) {
    const preferredDataset = preset.recommended_dataset_ids.find((id) =>
      datasets.some((dataset) => dataset.id === id),
    );

    setForm((prev) => ({
      ...prev,
      dataset_id: preferredDataset ?? prev.dataset_id,
      mode: preset.recommended_mode,
      chunking_strategy: preset.recommended_chunking_strategy,
      retrieval_k: preset.recommended_retrieval_k,
      logicrag: preset.recommended_techniques.logicrag ?? prev.logicrag,
      recursive_retrieval: preset.recommended_techniques.recursive_retrieval ?? prev.recursive_retrieval,
      graph_augmentation: preset.recommended_techniques.graph_augmentation ?? prev.graph_augmentation,
      hyde: preset.recommended_techniques.hyde ?? prev.hyde,
      self_rag: preset.recommended_techniques.self_rag ?? prev.self_rag,
      query_rewrite: preset.recommended_techniques.query_rewrite ?? prev.query_rewrite,
      cross_encoder_rerank: preset.recommended_techniques.cross_encoder_rerank ?? prev.cross_encoder_rerank,
      temporal_recency_filter:
        preset.recommended_techniques.temporal_recency_filter ?? prev.temporal_recency_filter,
      citation_enforcement: preset.recommended_techniques.citation_enforcement ?? prev.citation_enforcement,
      claim_verification: preset.recommended_techniques.claim_verification ?? prev.claim_verification,
      use_case_id: preset.id,
    }));
  }

  async function handleRun(event: FormEvent) {
    event.preventDefault();
    if (!form.dataset_id) {
      setError("Choose a dataset before running a workflow.");
      return;
    }

    try {
      setRunning(true);
      setError("");
      const response = await runWorkflow(form);
      setActiveRun(response);
      setRuns((prev) => [response.summary, ...prev.filter((item) => item.run_id !== response.summary.run_id)]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Workflow execution failed");
    } finally {
      setRunning(false);
    }
  }

  async function loadRun(runId: string) {
    try {
      setError("");
      const run = await getWorkflowRun(runId);
      setActiveRun(run);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load run detail");
    }
  }

  if (loading) {
    return <div className="loading">Loading workflow studio...</div>;
  }

  return (
    <main className="shell">
      <section className="hero">
        <p className="eyebrow">Production RAG Evaluation Studio</p>
        <h1>Measure architecture, not prompts.</h1>
        <p>
          Evaluate chunking, LogicRAG planning, query rewrite, reranking, recursive retrieval, graph augmentation,
          HyDE, Self-RAG, and faithfulness guards against real datasets, then inspect quality, latency, hallucination,
          and token economics per workflow run.
        </p>
      </section>

      {error ? <div className="error-box">{error}</div> : null}

      <section className="panel-grid">
        <form className="panel" onSubmit={handleRun}>
          <h2>Workflow Builder</h2>

          <label>
            Use-case preset
            <select
              value={form.use_case_id ?? ""}
              onChange={(event) => {
                const id = event.target.value;
                if (!id) {
                  setForm((prev) => ({ ...prev, use_case_id: undefined }));
                  return;
                }
                const preset = useCases.find((item) => item.id === id);
                if (preset) applyUseCase(preset);
              }}
            >
              <option value="">Custom workflow</option>
              {useCases.map((preset) => (
                <option value={preset.id} key={preset.id}>
                  {preset.label}
                </option>
              ))}
            </select>
          </label>

          {selectedUseCase ? (
            <div className="preset-notes">
              <strong>{selectedUseCase.label}</strong>
              <p>{selectedUseCase.description}</p>
              <ul>
                {selectedUseCase.evaluation_focus.map((focus) => (
                  <li key={focus}>{focus}</li>
                ))}
              </ul>
            </div>
          ) : null}

          <label>
            Workflow name
            <input
              type="text"
              placeholder="Optional display name"
              value={form.workflow_name ?? ""}
              onChange={(event) => setForm((prev) => ({ ...prev, workflow_name: event.target.value }))}
            />
          </label>

          <div className="inline-fields">
            <label>
              Dataset
              <select
                value={form.dataset_id}
                disabled={datasets.length === 0}
                onChange={(event) => setForm((prev) => ({ ...prev, dataset_id: event.target.value }))}
              >
                {datasets.length === 0 ? (
                  <option value="">No datasets available</option>
                ) : (
                  datasets.map((dataset) => (
                    <option value={dataset.id} key={dataset.id}>
                      {dataset.name}
                    </option>
                  ))
                )}
              </select>
            </label>

            <label>
              Mode
              <select
                value={form.mode}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, mode: event.target.value as WorkflowRunRequest["mode"] }))
                }
              >
                <option value="ablation">Ablation</option>
                <option value="single">Single stack</option>
              </select>
            </label>

            <label>
              Execution
              <select
                value={form.execution_mode}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    execution_mode: event.target.value as WorkflowRunRequest["execution_mode"],
                  }))
                }
              >
                <option value="real">Real (LLM APIs)</option>
                <option value="synthetic">Synthetic (test only)</option>
              </select>
            </label>
          </div>

          <div className="inline-fields">
            <label>
              Sample size
              <input
                type="number"
                min={10}
                max={500}
                value={form.sample_size}
                onChange={(event) => setForm((prev) => ({ ...prev, sample_size: Number(event.target.value) }))}
              />
            </label>

            <label>
              Retrieval k
              <input
                type="number"
                min={2}
                max={20}
                value={form.retrieval_k}
                onChange={(event) => setForm((prev) => ({ ...prev, retrieval_k: Number(event.target.value) }))}
              />
            </label>

            <label>
              Hallucination target
              <input
                type="number"
                min={0}
                max={0.5}
                step={0.005}
                value={form.target_hallucination_rate}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, target_hallucination_rate: Number(event.target.value) }))
                }
              />
            </label>

            <label>
              Chunking
              <select
                value={form.chunking_strategy}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    chunking_strategy: event.target.value as WorkflowRunRequest["chunking_strategy"],
                  }))
                }
              >
                {(techniques?.chunking_strategies ?? ["fixed", "sentence_window", "semantic", "adaptive"]).map(
                  (value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ),
                )}
              </select>
            </label>

            <label>
              Retrieval backend
              <select
                value={form.retrieval_backend}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    retrieval_backend: event.target.value as WorkflowRunRequest["retrieval_backend"],
                  }))
                }
              >
                {(techniques?.retrieval_backends ?? ["in_memory", "zvec"]).map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <fieldset>
            <legend>RAG techniques</legend>
            <label>
              <input
                type="checkbox"
                checked={form.logicrag}
                onChange={(event) => setForm((prev) => ({ ...prev, logicrag: event.target.checked }))}
              />
              LogicRAG dependency planner
            </label>
            <label>
              <input
                type="checkbox"
                checked={form.recursive_retrieval}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, recursive_retrieval: event.target.checked }))
                }
              />
              Recursive retrieval
            </label>
            <label>
              <input
                type="checkbox"
                checked={form.graph_augmentation}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, graph_augmentation: event.target.checked }))
                }
              />
              Graph augmentation
            </label>
            <label>
              <input
                type="checkbox"
                checked={form.hyde}
                onChange={(event) => setForm((prev) => ({ ...prev, hyde: event.target.checked }))}
              />
              HyDE
            </label>
            <label>
              <input
                type="checkbox"
                checked={form.self_rag}
                onChange={(event) => setForm((prev) => ({ ...prev, self_rag: event.target.checked }))}
              />
              Self-RAG
            </label>
            <label>
              <input
                type="checkbox"
                checked={form.query_rewrite}
                onChange={(event) => setForm((prev) => ({ ...prev, query_rewrite: event.target.checked }))}
              />
              Query rewrite
            </label>
            <label>
              <input
                type="checkbox"
                checked={form.cross_encoder_rerank}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, cross_encoder_rerank: event.target.checked }))
                }
              />
              Cross-encoder rerank
            </label>
            <label>
              <input
                type="checkbox"
                checked={form.temporal_recency_filter}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, temporal_recency_filter: event.target.checked }))
                }
              />
              Temporal recency filter
            </label>
            <label>
              <input
                type="checkbox"
                checked={form.citation_enforcement}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, citation_enforcement: event.target.checked }))
                }
              />
              Citation enforcement
            </label>
            <label>
              <input
                type="checkbox"
                checked={form.claim_verification}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, claim_verification: event.target.checked }))
                }
              />
              Claim verification firewall
            </label>
          </fieldset>

          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={form.include_baseline}
              onChange={(event) => setForm((prev) => ({ ...prev, include_baseline: event.target.checked }))}
            />
            Include baseline strategy in comparison
          </label>

          <button type="submit" disabled={running || datasets.length === 0}>
            {running ? "Running workflow..." : "Run Evaluation"}
          </button>
        </form>

        <aside className="panel">
          <h2>Dataset Inventory</h2>
          <div className="dataset-list">
            {datasets.length === 0 ? <p>No datasets found. Verify files under the `data/` directory.</p> : null}
            {datasets.map((dataset) => (
              <article className="dataset-card" key={dataset.id}>
                <h3>{dataset.name}</h3>
                <p>{dataset.description}</p>
                <div className="meta-line">
                  <span>{dataset.domain}</span>
                  <span>{dataset.format}</span>
                  <span>{dataset.approx_queries.toLocaleString()} queries</span>
                </div>
              </article>
            ))}
          </div>
        </aside>
      </section>

      <section className="panel">
        <h2>Run History</h2>
        <div className="run-history">
          {runs.length === 0 ? <p>No workflow runs yet.</p> : null}
          {runs.map((run) => (
            <button className="run-item" key={run.run_id} onClick={() => loadRun(run.run_id)}>
              <strong>{run.workflow_name}</strong>
              <span>
                {run.dataset_name} | {new Date(run.created_at).toLocaleString()}
              </span>
              <span>
                Best: {run.best_strategy} ({pct(run.best_quality_score)})
              </span>
            </button>
          ))}
        </div>
      </section>

      {activeRun ? (
        <section className="results-grid">
          <article className="panel stat-panel">
            <small>Top quality</small>
            <strong>{bestStrategy?.strategy_name ?? "n/a"}</strong>
            <span>{bestStrategy ? pct(bestStrategy.avg_quality_score) : "-"}</span>
          </article>
          <article className="panel stat-panel">
            <small>Fastest</small>
            <strong>{fastestStrategy?.strategy_name ?? "n/a"}</strong>
            <span>{fastestStrategy ? ms(fastestStrategy.avg_latency_ms) : "-"}</span>
          </article>
          <article className="panel stat-panel">
            <small>Cheapest per query</small>
            <strong>{cheapestStrategy?.strategy_name ?? "n/a"}</strong>
            <span>{cheapestStrategy ? usd(cheapestStrategy.avg_token_cost_usd) : "-"}</span>
          </article>

          <article className="panel full">
            <h2>Leaderboard</h2>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Strategy</th>
                    <th>Precision@k</th>
                    <th>Recall@k</th>
                    <th>MRR</th>
                    <th>Hallucination rate</th>
                    <th>Grounded claims</th>
                    <th>Quality</th>
                    <th>Latency</th>
                    <th>Avg cost</th>
                  </tr>
                </thead>
                <tbody>
                  {aggregates.map((item) => (
                    <tr key={item.strategy_name}>
                      <td>{item.strategy_name}</td>
                      <td>{pct(item.avg_precision_at_k)}</td>
                      <td>{pct(item.avg_recall_at_k)}</td>
                      <td>{item.mrr.toFixed(3)}</td>
                      <td>{pct(item.avg_hallucination_rate)}</td>
                      <td>{pct(item.avg_hallucination_score)}</td>
                      <td>{pct(item.avg_quality_score)}</td>
                      <td>{ms(item.avg_latency_ms)}</td>
                      <td>{usd(item.avg_token_cost_usd)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>

          <article className="panel full">
            <h2>Latency vs Quality</h2>
            {!scatterPoints ? null : (
              <svg viewBox={`0 0 ${scatterPoints.width} ${scatterPoints.height}`} className="scatter">
                <line
                  x1={scatterPoints.padding.left}
                  x2={scatterPoints.width - scatterPoints.padding.right}
                  y1={scatterPoints.height - scatterPoints.padding.bottom}
                  y2={scatterPoints.height - scatterPoints.padding.bottom}
                  stroke="#6f8c6e"
                />
                <line
                  x1={scatterPoints.padding.left}
                  x2={scatterPoints.padding.left}
                  y1={scatterPoints.padding.top}
                  y2={scatterPoints.height - scatterPoints.padding.bottom}
                  stroke="#6f8c6e"
                />

                {scatterPoints.points.map((point) => (
                  <g key={point.strategy}>
                    <circle cx={point.x} cy={point.y} r={5.5} fill="#d65c2f" />
                    <text x={point.x + 8} y={point.y - 8}>
                      {point.strategy}
                    </text>
                  </g>
                ))}
              </svg>
            )}
          </article>

          <article className="panel full">
            <h2>Unsupported Claims (Top Findings)</h2>
            <div className="claim-list">
              {activeRun.governance.unsupported_claims.slice(0, 12).map((claim, idx) => (
                <article className="claim-card" key={`${claim.query_id}-${idx}`}>
                  <header>
                    <strong>{claim.strategy_name}</strong>
                    <span>support ratio {claim.support_ratio.toFixed(2)}</span>
                  </header>
                  <p className="claim-text">{claim.claim}</p>
                  <p className="claim-query">Q: {claim.query}</p>
                  <div className="context-box">{claim.contexts[0] ?? "No context captured."}</div>
                </article>
              ))}
            </div>
          </article>

          <article className="panel full">
            <h2>Intent Hallucination Slice</h2>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Strategy</th>
                    <th>Intent</th>
                    <th>Avg hallucination rate</th>
                    <th>Avg grounded claims</th>
                    <th>Cases</th>
                  </tr>
                </thead>
                <tbody>
                  {activeRun.governance.intent_hallucination.map((row) => (
                    <tr key={`${row.strategy_name}-${row.intent}`}>
                      <td>{row.strategy_name}</td>
                      <td>{row.intent}</td>
                      <td>{pct(row.avg_hallucination_rate ?? (1 - (row.avg_grounded_claim_ratio ?? 0)))}</td>
                      <td>{pct(row.avg_grounded_claim_ratio ?? 0)}</td>
                      <td>{row.case_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>

          <article className="panel full">
            <h2>Cost per Quality Point</h2>
            <div className="cost-bars">
              {activeRun.governance.cost_per_quality.map((row) => {
                const values = activeRun.governance.cost_per_quality.map((item) => item.cost_per_quality_point);
                const maxValue = Math.max(...values, 0.00001);
                const width = (row.cost_per_quality_point / maxValue) * 100;

                return (
                  <div className="cost-row" key={row.strategy_name}>
                    <span>{row.strategy_name}</span>
                    <div className="bar-track">
                      <div className="bar-fill" style={{ width: `${width}%` }} />
                    </div>
                    <span>{row.cost_per_quality_point.toFixed(4)}</span>
                  </div>
                );
              })}
            </div>
          </article>

          <article className="panel full">
            <h2>Governance Snapshot</h2>
            <div className="snapshot-row">
              <div>
                <small>Average p95 latency</small>
                <strong>
                  {ms(metricValue(activeRun.governance.latency_slo.map((item) => item.p95_latency_ms), 0))}
                </strong>
              </div>
              <div>
                <small>Safety violations</small>
                <strong>{activeRun.governance.safety_violations.length}</strong>
              </div>
              <div>
                <small>Unsupported claims logged</small>
                <strong>{activeRun.governance.unsupported_claims.length}</strong>
              </div>
              <div>
                <small>Hallucination target gate</small>
                <strong>
                  {hallucinationGate.has_passing_strategy
                    ? `PASS ${hallucinationGate.passing_count}/${hallucinationGate.total_count} <= ${pct(hallucinationGate.target_hallucination_rate)}`
                    : `FAIL ${hallucinationGate.passing_count}/${hallucinationGate.total_count} <= ${pct(hallucinationGate.target_hallucination_rate)}`}
                </strong>
              </div>
            </div>
          </article>
        </section>
      ) : null}
    </main>
  );
}
