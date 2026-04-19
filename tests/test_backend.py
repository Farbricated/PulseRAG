"""
PulseRAG — Unit Tests
Run: pytest tests/ -v
"""
import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import backend as bk


@pytest.fixture(scope="session", autouse=True)
def bootstrap():
    bk.bootstrap()


# ── Pure function tests (no API needed) ───────────────────────────────────────

def test_embed_shape():
    emb = bk.embed(["gradient descent"])
    assert emb.shape == (1, 384)


def test_embed_normalized():
    emb = bk.embed(["hello world"])
    norm = float(np.linalg.norm(emb[0]))
    assert abs(norm - 1.0) < 1e-4


def test_detect_topic_ml():
    assert bk.detect_topic("explain gradient descent in machine learning") == "machine_learning"


def test_detect_topic_rag():
    assert bk.detect_topic("how does RAG retrieval work") == "rag"


def test_detect_topic_none():
    assert bk.detect_topic("hello how are you") is None


def test_bm25_score_positive():
    score = bk.bm25_score("gradient descent", "gradient descent minimizes the loss function")
    assert score > 0


def test_bm25_score_zero_no_match():
    score = bk.bm25_score("unrelated query", "completely different topic about cooking")
    assert score == 0.0


def test_hallucination_risk_low():
    docs = [{"text": "gradient descent minimizes loss by updating weights iteratively"}]
    risk = bk.hallucination_risk("gradient descent minimizes loss", docs)
    assert 0 <= risk <= 1
    assert risk < 0.5


def test_hallucination_risk_high():
    docs = [{"text": "photosynthesis converts sunlight into glucose in plants"}]
    risk = bk.hallucination_risk("quantum mechanics describes particle behavior", docs)
    assert risk > 0.5


def test_response_diversity_score():
    score = bk.response_diversity_score("the quick brown fox jumps over the lazy dog")
    assert 0 <= score <= 1
    assert score > 0.5  # all unique words


def test_response_diversity_repetitive():
    score = bk.response_diversity_score("the the the the the the the the the the")
    assert score < 0.3


def test_build_feature_dict_12_features():
    raw = {
        "time_to_read": 30.0, "follow_up_asked": 1, "similar_followup": 0,
        "session_query_count": 3, "response_length": 400,
        "avg_retrieval_score": 0.72, "query_complexity": 0.4,
        "hallucination_risk": 0.2, "response_diversity": 0.65,
    }
    feat = bk.build_feature_dict(raw)
    assert len(feat) == 12
    assert feat["quality_weighted_length"] == pytest.approx(400 * 0.72)
    assert feat["engagement_depth"] == pytest.approx(3 * 1)
    assert feat["risk_under_load"] == pytest.approx(0.2 * 0.4)


def test_smart_chunk_non_empty():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = bk.smart_chunk(text)
    assert len(chunks) >= 1
    assert all(len(c) > 0 for c in chunks)


def test_smart_chunk_respects_max_chars():
    text = " ".join(["word"] * 500)
    chunks = bk.smart_chunk(text, max_chars=200)
    assert all(len(c) <= 250 for c in chunks)


def test_compute_query_complexity_low():
    score = bk.compute_query_complexity("what is ML")
    assert 0 <= score <= 1


def test_compute_query_complexity_high():
    score = bk.compute_query_complexity(
        "explain the difference between bias and variance tradeoff in deep learning models"
    )
    assert score > 0.3


def test_psi_stable_distributions():
    a = np.random.default_rng(42).normal(0, 1, 500)
    psi = bk.compute_psi(a, a)
    assert psi < 0.1


def test_psi_different_distributions():
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 500)
    b = rng.normal(3, 1, 500)
    psi = bk.compute_psi(a, b)
    assert psi > 0.2


def test_rrf_merge_combines_lists():
    list1 = [{"doc_id": "a", "text": "hello", "score": 0.9, "topic": "ml"},
             {"doc_id": "b", "text": "world", "score": 0.7, "topic": "ml"}]
    list2 = [{"doc_id": "b", "text": "world", "score": 0.8, "topic": "ml"},
             {"doc_id": "c", "text": "foo",   "score": 0.6, "topic": "ml"}]
    merged = bk._rrf_merge([list1, list2], top_k=3)
    ids = [d["doc_id"] for d in merged]
    assert "b" in ids  # appeared in both lists, should rank high
    assert len(merged) == 3


# ── ML model tests ────────────────────────────────────────────────────────────

def test_synthetic_data_shape():
    df = bk.generate_synthetic_data(100)
    assert len(df) == 100
    assert "understood" in df.columns
    assert df["understood"].nunique() == 2


def test_synthetic_data_balanced():
    df = bk.generate_synthetic_data(200)
    counts = df["understood"].value_counts()
    assert abs(counts[0] - counts[1]) <= 10


def test_train_model_returns_metrics():
    df  = bk.generate_synthetic_data(200)
    met = bk.train_model(df)
    assert "roc_auc" in met
    assert met["roc_auc"] > 0.5
    assert met["roc_auc"] <= 1.0
    assert "opt_threshold" in met
    assert 0 < met["opt_threshold"] < 1


def test_predict_understanding_valid_output():
    feat = {
        "time_to_read": 45, "follow_up_asked": 0, "similar_followup": 0,
        "session_query_count": 1, "response_length": 500,
        "avg_retrieval_score": 0.75, "query_complexity": 0.35,
        "hallucination_risk": 0.18, "response_diversity": 0.72,
    }
    pred = bk.predict_understanding(feat)
    assert 0 <= pred["probability"] <= 1
    assert pred["understood"] in [0, 1]
    assert "threshold" in pred
    assert "confidence" in pred


def test_predict_high_retrieval_understood():
    feat = {
        "time_to_read": 90, "follow_up_asked": 0, "similar_followup": 0,
        "session_query_count": 1, "response_length": 600,
        "avg_retrieval_score": 0.95, "query_complexity": 0.2,
        "hallucination_risk": 0.05, "response_diversity": 0.80,
    }
    pred = bk.predict_understanding(feat)
    assert pred["probability"] > 0.5


def test_predict_confusion_signals_not_understood():
    feat = {
        "time_to_read": 3, "follow_up_asked": 1, "similar_followup": 1,
        "session_query_count": 10, "response_length": 100,
        "avg_retrieval_score": 0.3, "query_complexity": 0.9,
        "hallucination_risk": 0.85, "response_diversity": 0.3,
    }
    pred = bk.predict_understanding(feat)
    assert pred["probability"] < 0.5


# ── RAGAS metric tests ────────────────────────────────────────────────────────

def test_context_precision_range():
    docs = bk.hybrid_retrieve("gradient descent", 4)
    cp = bk.compute_context_precision("gradient descent", docs)
    assert 0 <= cp <= 1


def test_context_recall_high_overlap():
    docs = [{"text": "gradient descent minimizes the loss function iteratively"}]
    cr = bk.compute_context_recall("gradient descent minimizes loss", docs)
    assert cr > 0.3


def test_faithfulness_range():
    docs = [{"text": "neural networks learn by backpropagation through the chain rule"}]
    f = bk.compute_faithfulness("neural networks use backpropagation", docs)
    assert 0 <= f <= 1


def test_ragas_all_three():
    docs = bk.hybrid_retrieve("transformer attention", 4)
    r = bk.compute_ragas_metrics("transformer attention", "attention uses queries and keys", docs)
    assert set(r.keys()) == {"context_precision", "context_recall", "faithfulness"}
    assert all(0 <= v <= 1 for v in r.values())


# ── Retrieval tests ───────────────────────────────────────────────────────────

def test_dense_retrieve_returns_docs():
    docs = bk.dense_retrieve("what is backpropagation", 3)
    assert len(docs) > 0
    assert all("text" in d and "score" in d for d in docs)


def test_bm25_retrieve_returns_docs():
    docs = bk.bm25_retrieve("gradient descent optimizer", 3)
    assert len(docs) > 0


def test_filter_by_score_keeps_minimum():
    docs = [{"doc_id": "a", "text": "x", "score": 0.1, "dense_score": 0.1},
            {"doc_id": "b", "text": "y", "score": 0.8, "dense_score": 0.8},
            {"doc_id": "c", "text": "z", "score": 0.9, "dense_score": 0.9}]
    filtered = bk.filter_by_score(docs, threshold=0.5)
    assert len(filtered) == 2
    assert filtered[0]["doc_id"] == "b"


def test_filter_by_score_minimum_two():
    # Even if all below threshold, should keep at least 2
    docs = [{"doc_id": str(i), "text": "x", "score": 0.1, "dense_score": 0.1} for i in range(5)]
    filtered = bk.filter_by_score(docs, threshold=0.9)
    assert len(filtered) >= 2


def test_ab_group_deterministic():
    g1 = bk.assign_ab_group("session_abc")
    g2 = bk.assign_ab_group("session_abc")
    assert g1 == g2


def test_ab_group_distribution():
    groups = set(bk.assign_ab_group(f"s{i}") for i in range(30))
    assert len(groups) >= 2


# ── DB tests ─────────────────────────────────────────────────────────────────

def test_init_db_creates_tables():
    bk.init_db()
    df = bk._load_table("interactions")
    assert isinstance(df, pd.DataFrame)


def test_memory_add_and_retrieve():
    bk.add_to_memory("test_session_pytest", "user", "What is ML?")
    bk.add_to_memory("test_session_pytest", "assistant", "ML is machine learning.")
    ctx = bk.get_memory_context("test_session_pytest")
    assert "ML" in ctx
    assert len(ctx) > 10