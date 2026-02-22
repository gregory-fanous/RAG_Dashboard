import {
  DatasetSummary,
  TechniqueCatalog,
  UseCasePreset,
  WorkflowRunDetail,
  WorkflowRunRequest,
  WorkflowRunSummary,
} from "./types";

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE ?? "").replace(/\/$/, "");
const API_TARGET = API_BASE || "Next.js /api proxy";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = API_BASE ? `${API_BASE}${path}` : path;
  let response: Response;
  try {
    response = await fetch(url, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
      cache: "no-store",
    });
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error(
        `Unable to reach API via ${API_TARGET}. Start the backend and verify NEXT_PUBLIC_API_BASE or BACKEND_API_BASE.`,
      );
    }
    throw error;
  }

  if (!response.ok) {
    const body = await response.text();
    let detail = body;
    try {
      const parsed = JSON.parse(body) as { detail?: unknown };
      if (typeof parsed.detail === "string" && parsed.detail.trim()) {
        detail = parsed.detail;
      }
    } catch {
      // Leave non-JSON bodies unchanged.
    }
    throw new Error(detail || `Request failed (${response.status})`);
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
