import {
  DatasetSummary,
  TechniqueCatalog,
  UseCasePreset,
  WorkflowRunDetail,
  WorkflowRunRequest,
  WorkflowRunSummary,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function getDatasets(): Promise<DatasetSummary[]> {
  return request<DatasetSummary[]>("/api/datasets");
}

export function getTechniques(): Promise<TechniqueCatalog> {
  return request<TechniqueCatalog>("/api/techniques");
}

export function getUseCases(): Promise<UseCasePreset[]> {
  return request<UseCasePreset[]>("/api/use-cases");
}

export function getWorkflowRuns(): Promise<WorkflowRunSummary[]> {
  return request<WorkflowRunSummary[]>("/api/workflows");
}

export function getWorkflowRun(runId: string): Promise<WorkflowRunDetail> {
  return request<WorkflowRunDetail>(`/api/workflows/${runId}`);
}

export function runWorkflow(payload: WorkflowRunRequest): Promise<WorkflowRunDetail> {
  return request<WorkflowRunDetail>("/api/workflows/run", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
