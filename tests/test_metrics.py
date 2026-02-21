from rag_eval.metrics import (
    composite_quality_score,
    hallucination_detection_score,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_recall_and_mrr():
    retrieved = ["d1", "d2", "d3", "d4"]
    relevant = ["d3", "d4"]

    assert precision_at_k(retrieved, relevant, 2) == 0.0
    assert recall_at_k(retrieved, relevant, 3) == 0.5
    assert reciprocal_rank(retrieved, relevant) == 1.0 / 3.0


def test_hallucination_detection_score():
    answer = "Precision at k measures ranking purity. Quantum centroid cache guarantees factuality."
    contexts = ["Precision at k captures ranking purity for retrieval systems."]

    score = hallucination_detection_score(answer, contexts)
    assert 0.0 < score < 1.0


def test_composite_quality_score_weighting():
    score = composite_quality_score(precision=0.8, recall=0.6, mrr=0.5, hallucination_score=0.9)
    assert round(score, 4) == 0.67
