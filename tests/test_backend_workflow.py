from backend.app.datasets import DatasetRegistry
from backend.app.schemas import WorkflowRunRequest
from backend.app.workflows import WorkflowService


def test_dataset_registry_lists_expected_sources():
    ids = {item.id for item in DatasetRegistry().list_datasets()}
    expected = {
        "public_eval_set",
        "retrievalqa",
        "ragcare_qa",
        "open_ragbench",
        "natural_questions",
        "finder",
        "paperzilla",
    }
    assert expected.issubset(ids)


def test_dataset_registry_skips_partially_restored_dataset(tmp_path):
    # Simulate a restored dataset directory missing required metadata files.
    (tmp_path / "open_ragbench" / "pdf" / "arxiv").mkdir(parents=True)

    registry = DatasetRegistry(data_root=tmp_path)
    summaries = registry.list_datasets()

    assert summaries == []


def test_technique_catalog_includes_retrieval_backends():
    service = WorkflowService()
    catalog = service.technique_catalog()
    assert "in_memory" in catalog.retrieval_backends
    assert "zvec" in catalog.retrieval_backends


def test_workflow_service_runs_single_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_ALLOW_SYNTHETIC_MODE", "true")
    registry = DatasetRegistry()
    service = WorkflowService(artifacts_dir=tmp_path)

    request = WorkflowRunRequest(
        dataset_id="public_eval_set",
        execution_mode="synthetic",
        mode="single",
        sample_size=10,
        retrieval_k=5,
        chunking_strategy="semantic",
        recursive_retrieval=True,
        graph_augmentation=False,
        hyde=False,
        self_rag=False,
    )

    result = service.run_workflow(request, registry)

    assert result.summary.strategy_count >= 1
    assert result.request.retrieval_backend == "in_memory"
    assert result.report["strategy_results"]
    assert "unsupported_claims" in result.governance
    assert "latency_slo" in result.governance
    assert "hallucination_target" in result.governance
    assert result.governance["hallucination_target"]["total_count"] >= 1


def test_hallucination_target_can_reach_one_percent(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_ALLOW_SYNTHETIC_MODE", "true")
    registry = DatasetRegistry()
    service = WorkflowService(artifacts_dir=tmp_path)

    request = WorkflowRunRequest(
        dataset_id="public_eval_set",
        execution_mode="synthetic",
        mode="ablation",
        sample_size=10,
        retrieval_k=6,
        chunking_strategy="adaptive",
        recursive_retrieval=True,
        graph_augmentation=True,
        hyde=True,
        self_rag=True,
        query_rewrite=True,
        cross_encoder_rerank=True,
        citation_enforcement=True,
        claim_verification=True,
        target_hallucination_rate=0.01,
    )

    result = service.run_workflow(request, registry)
    gate = result.governance["hallucination_target"]
    assert gate["has_passing_strategy"] is True
    assert any(
        row["passes_target"] and row["avg_hallucination_rate"] <= 0.01
        for row in gate["strategies"]
    )


def test_workflow_service_accepts_zvec_backend_in_synthetic_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_ALLOW_SYNTHETIC_MODE", "true")
    registry = DatasetRegistry()
    service = WorkflowService(artifacts_dir=tmp_path)

    request = WorkflowRunRequest(
        dataset_id="public_eval_set",
        execution_mode="synthetic",
        mode="single",
        sample_size=10,
        chunking_strategy="fixed",
        retrieval_backend="zvec",
        retrieval_k=5,
    )

    result = service.run_workflow(request, registry)
    assert result.request.retrieval_backend == "zvec"


def test_list_runs_skips_malformed_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_ALLOW_SYNTHETIC_MODE", "true")
    registry = DatasetRegistry()
    service = WorkflowService(artifacts_dir=tmp_path)

    request = WorkflowRunRequest(
        dataset_id="public_eval_set",
        execution_mode="synthetic",
        mode="single",
        sample_size=10,
    )
    service.run_workflow(request, registry)

    (tmp_path / "broken.json").write_text("{not-valid-json", encoding="utf-8")
    runs = service.list_runs(limit=20)
    assert len(runs) >= 1
