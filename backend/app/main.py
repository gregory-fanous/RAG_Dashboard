from __future__ import annotations

from datetime import datetime, timezone
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .datasets import DatasetRegistry
from .schemas import (
    DatasetSummary,
    HealthResponse,
    TechniqueCatalog,
    UseCasePreset,
    WorkflowRunDetail,
    WorkflowRunRequest,
    WorkflowRunSummary,
)
from .workflows import WorkflowService

app = FastAPI(
    title="RAG Evaluation Workflow API",
    description="Dataset and technique orchestration API for RAG benchmark experimentation",
    version="0.2.0",
)


def _cors_origins() -> list[str]:
    default_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://0.0.0.0:3000",
    ]
    configured = [
        item.strip()
        for item in os.getenv("RAG_EVAL_CORS_ALLOW_ORIGINS", "").split(",")
        if item.strip()
    ]
    return sorted({*default_origins, *configured})


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _registry() -> DatasetRegistry:
    return DatasetRegistry()


def _service() -> WorkflowService:
    return WorkflowService()


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", timestamp=datetime.now(timezone.utc))


@app.get("/api/datasets", response_model=list[DatasetSummary])
def list_datasets() -> list[DatasetSummary]:
    try:
        return _registry().list_datasets()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/techniques", response_model=TechniqueCatalog)
def techniques() -> TechniqueCatalog:
    return _service().technique_catalog()


@app.get("/api/use-cases", response_model=list[UseCasePreset])
def use_cases() -> list[UseCasePreset]:
    return _service().use_case_presets()


@app.get("/api/workflows", response_model=list[WorkflowRunSummary])
def list_workflows(limit: int = 20) -> list[WorkflowRunSummary]:
    try:
        return _service().list_runs(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/workflows/{run_id}", response_model=WorkflowRunDetail)
def get_workflow(run_id: str) -> WorkflowRunDetail:
    try:
        return _service().get_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/workflows/run", response_model=WorkflowRunDetail)
def run_workflow(request: WorkflowRunRequest) -> WorkflowRunDetail:
    service = _service()
    registry = _registry()

    try:
        return service.run_workflow(request=request, dataset_registry=registry)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
