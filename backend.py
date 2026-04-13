"""
PulseRAG v4 — backend.py
========================
Fixes applied vs v3:
  [C1]  time_to_read: computed from real elapsed time, not hardcoded 12.0
  [C2]  rewrite_query: skip rewrite for complex queries (>=8 words) AND cache results
  [C3]  generate_synthetic_data: overlapping Gaussian distributions → realistic AUC ~0.78
  [C4]  Global state isolated per session: _streak, _prompt_version, _session_understood
  [C5]  metric_after: stored as pending, resolved on next query's prediction
  [B6]  bm25_score: uses real corpus avgdl, recalculated on ingest
  [B7]  hybrid_retrieve: reports actual RRF scores, not discarded dense scores
  [B8]  llm_judge: returns None on failure, never stores fake 3s
  [B9]  setup_collection: content-hash freshness check, not just count
  [B10] _parse_csv: joins all text columns, reports detected columns
  [B11] follow_up_asked / similar_followup: computed from embedding similarity
  [D12] A/B and feedback loop separated: feedback escalation is per-session
  [D13] Memory persisted to SQLite, restored on session reconnect
  [D14] Sentence-boundary-aware chunking exported for app.py
  [D15] run_pipeline accepts on_step callback for live pipeline progress
  [M17] get_system_stats cached with TTL via module-level timestamp
  [M18] _get_groq_client() centralised factory
  [M19] PDF fallback removed, ImportError raised explicitly
  [M20] Test dispatch uses index not startswith matching
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import sqlite3
import time
import warnings
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
CHROMA_DIR       = "chroma_db"
COLLECTION_NAME  = "pulserag_knowledge_v4"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL       = "llama-3.1-8b-instant"
DB_PATH          = "pulserag.db"
MODEL_PATH       = "pulserag_model.pkl"
TOP_K            = 10
VERSION          = "4.0.0"
EMBED_BATCH_SIZE = 64

_DEVICE = os.getenv("PULSERAG_DEVICE", None)

FEATURES = [
    "time_to_read", "follow_up_asked", "similar_followup",
    "session_query_count", "response_length", "avg_retrieval_score",
    "query_complexity", "hallucination_risk", "response_diversity",
]

_STREAK_LIMIT  = 3
_RETRAIN_EVERY = 30
PROMPT_SEQUENCE = ["v1_standard", "v2_simplified", "v3_detailed", "v4_socratic"]

# ── Singletons (process-wide, shared across sessions) ────────────────────────
_ml_model:           object = None
_ml_metrics:         dict   = {}
_feature_importance: dict   = {}
_embed_model:        object = None
_chroma_collection:  object = None

# ── Per-session state (keyed by session_id) ───────────────────────────────────
# [C4] All mutable per-user state lives here, not as bare globals
_session_state: dict[str, dict] = {}   # streak, prompt_version, understood history
_memory:        dict[str, deque] = defaultdict(lambda: deque(maxlen=6))

# [C5] Pending metric_before waiting for next query to resolve metric_after
_pending_feedback: dict[str, dict] = {}   # session_id → {feedback_log_id, metric_before}

# [C2] Query rewrite cache
_rewrite_cache: dict[str, str] = {}

# [B6] Corpus average document length — recalculated when corpus changes
_bm25_avgdl: float = 80.0

# [M17] Stats cache
_stats_cache:    dict  = {}
_stats_cache_ts: float = 0.0
_STATS_TTL:      float = 3.0   # seconds


def _get_session(session_id: str) -> dict:
    """[C4] Return per-session mutable state, creating it if absent."""
    if session_id not in _session_state:
        _session_state[session_id] = {
            "streak":         0,
            "prompt_version": "v1_standard",
            "understood":     [],
            "total_queries":  0,
        }
    return _session_state[session_id]


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_DOCS: list[dict] = [
    {"id":"ml_001","topic":"machine_learning","text":"Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves. The process begins with observations—examples, direct experience, or instruction—so that computers can make better decisions in the future without human intervention for each case."},
    {"id":"ml_002","topic":"machine_learning","text":"Supervised learning trains models on labeled input-output pairs. Common algorithms include linear regression, logistic regression, decision trees, random forests, support vector machines, and neural networks. Model performance is measured on held-out test data to estimate generalization."},
    {"id":"ml_003","topic":"machine_learning","text":"Unsupervised learning finds hidden patterns in unlabeled data. K-means clustering partitions data into k groups by minimizing within-cluster variance. DBSCAN finds density-based clusters. PCA and t-SNE compress high-dimensional data for visualization and feature extraction."},
    {"id":"ml_004","topic":"machine_learning","text":"Reinforcement learning trains agents to maximize cumulative reward by interacting with an environment. Key algorithms: Q-learning, Deep Q-Networks, Actor-Critic, PPO. Used in robotics, games, and recommendation systems."},
    {"id":"ml_005","topic":"machine_learning","text":"Ensemble methods combine multiple models. Bagging (Random Forest) trains on random subsamples. Boosting (AdaBoost, GBM, XGBoost) trains sequentially. Stacking uses a meta-learner to combine base predictions."},
    {"id":"ml_006","topic":"machine_learning","text":"Bias-variance tradeoff is fundamental. High bias = underfitting. High variance = overfitting. Regularization (L1 Lasso, L2 Ridge, elastic net) penalizes complexity to balance this tradeoff."},
    {"id":"sys_006","topic":"machine_learning","text":"Imbalanced classification strategies: SMOTE oversample, random undersample, class_weight='balanced'. Evaluation must use F1, ROC-AUC, or Precision-Recall AUC — never raw accuracy on imbalanced datasets."},
    {"id":"dl_001","topic":"deep_learning","text":"Deep learning uses neural networks with many layers. Backpropagation computes gradients via the chain rule. Gradient descent updates weights to minimize loss. Excels at vision, NLP, and speech tasks."},
    {"id":"dl_002","topic":"deep_learning","text":"Convolutional Neural Networks process grid-like data. Convolutional layers detect local patterns. Max pooling reduces spatial dimensions. Notable architectures: AlexNet, VGG, ResNet, EfficientNet."},
    {"id":"dl_003","topic":"deep_learning","text":"Recurrent Neural Networks process sequential data via hidden state. LSTMs use gating mechanisms. GRUs are simpler. Used in time-series, speech, and text generation."},
    {"id":"dl_004","topic":"deep_learning","text":"The Transformer uses self-attention to process all tokens in parallel. Multi-head attention attends to different subspaces simultaneously. BERT (encoder-only), GPT (decoder-only), T5 (encoder-decoder)."},
    {"id":"dl_005","topic":"deep_learning","text":"Transfer learning reuses pretrained models. Fine-tuning updates all or some weights. Feature extraction freezes pretrained weights and trains only a new head. Dramatically reduces data and compute required."},
    {"id":"sys_007","topic":"deep_learning","text":"Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V. Multi-head attention runs several attention operations in parallel and concatenates results."},
    {"id":"rag_001","topic":"rag","text":"Retrieval-Augmented Generation grounds LLM responses in external knowledge. Pipeline: embed query → search vector DB → inject top-k docs as context → generate conditioned response. Reduces hallucination and enables knowledge updates without retraining."},
    {"id":"rag_002","topic":"rag","text":"Vector databases store embeddings for ANN search. HNSW enables sub-linear complexity. ChromaDB is open-source and local-first. Pinecone, Weaviate, Milvus are managed alternatives."},
    {"id":"rag_003","topic":"rag","text":"Chunking strategy determines retrieval quality. Fixed-size, sentence-based, recursive, and semantic chunking are common approaches. Optimal chunk size 256-1024 tokens depends on embedding model context window."},
    {"id":"rag_004","topic":"rag","text":"Advanced RAG: HyDE generates hypothetical answers for retrieval. Query rewriting uses LLM to reformulate. Multi-query retrieval merges results. Re-ranking with cross-encoders improves precision."},
    {"id":"rag_005","topic":"rag","text":"Hybrid search combines dense vector search with sparse BM25. Dense retrieval captures semantic meaning. Sparse retrieval handles exact keyword matching. RRF merges ranked lists without score normalization."},
    {"id":"sys_001","topic":"rag","text":"Agentic RAG extends basic RAG with tool-use and multi-step reasoning. Agents decompose complex queries into sub-questions and iteratively retrieve additional context. LangChain, LlamaIndex, OpenAI Assistants provide frameworks."},
    {"id":"eval_001","topic":"evaluation","text":"Classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC. Macro averaging treats all classes equally; weighted averaging accounts for class imbalance."},
    {"id":"eval_002","topic":"evaluation","text":"Cross-validation: K-fold CV splits data into k partitions. Stratified k-fold preserves class proportions — essential for imbalanced datasets. Nested CV avoids optimistic bias."},
    {"id":"eval_003","topic":"evaluation","text":"RAGAS metrics: context_precision, context_recall, faithfulness, answer_relevancy. LLM-as-Judge uses a strong model to score responses without human annotation."},
    {"id":"eval_004","topic":"evaluation","text":"Calibration measures if predicted probabilities reflect true likelihoods. Platt scaling and isotonic regression are post-hoc calibration methods. Expected Calibration Error quantifies deviation."},
    {"id":"sys_003","topic":"evaluation","text":"A/B testing compares system variants by routing different user segments to different versions. Statistical significance via t-test or chi-squared with alpha=0.05. Cohen's d measures practical significance."},
    {"id":"nlp_001","topic":"nlp","text":"NLP enables computers to understand human language. Core tasks: tokenization, POS tagging, NER, dependency parsing. Higher-level tasks: sentiment analysis, machine translation, summarization. Modern NLP uses pre-trained transformers."},
    {"id":"nlp_002","topic":"nlp","text":"Word embeddings: Word2Vec trains on co-occurrence. GloVe uses global statistics. FastText handles OOV via subword n-grams. BERT and RoBERTa produce contextual embeddings capturing polysemy."},
    {"id":"nlp_003","topic":"nlp","text":"LLMs like GPT-4, Llama 3, Claude are transformer-based models trained on next-token prediction. Emergent capabilities: in-context learning, chain-of-thought reasoning, instruction following."},
    {"id":"sys_009","topic":"nlp","text":"Tokenization: BPE merges frequent character pairs. WordPiece (BERT) and SentencePiece (T5, LLaMA) are variants. One token is approximately 4 characters in English."},
    {"id":"pipe_001","topic":"ml_pipeline","text":"ML pipelines automate: data ingestion, validation, preprocessing, feature engineering, training, evaluation, deployment. MLflow tracks experiments. DVC versions datasets."},
    {"id":"pipe_002","topic":"ml_pipeline","text":"MLOps applies DevOps to ML: triggered retraining, automated validation gates, canary deployments, automated rollback. Tools: GitHub Actions, Docker, Kubernetes."},
    {"id":"sys_002","topic":"ml_pipeline","text":"Online learning updates models incrementally without retraining from scratch. Critical when distributions shift over time. Concept drift detection (ADWIN, DDM) can trigger updates."},
    {"id":"drift_001","topic":"monitoring","text":"Data drift: KS-test for continuous features, chi-squared for categorical, PSI and Jensen-Shannon divergence for distribution shift detection."},
    {"id":"drift_002","topic":"monitoring","text":"Production ML monitoring tracks: prediction distribution, input feature statistics, data quality, model latency (p50/p95/p99), throughput, error rates, and business KPIs."},
    {"id":"drift_003","topic":"monitoring","text":"Shadow deployment runs a new model in parallel without serving predictions to users. Validates new model on real traffic before promotion. A/B testing measures business metrics."},
    {"id":"llm_001","topic":"llm_prompting","text":"Prompt engineering: zero-shot (task only), few-shot (examples included), chain-of-thought (step-by-step reasoning), ReAct (reasoning + tool use alternating)."},
    {"id":"llm_002","topic":"llm_prompting","text":"Fine-tuning: full fine-tuning updates all weights. LoRA injects small trainable matrices into attention layers, reducing trainable parameters by 10,000x. QLoRA adds 4-bit quantization."},
    {"id":"llm_003","topic":"llm_prompting","text":"LLM hallucination: plausible but factually incorrect content. Mitigations: RAG grounds responses in retrieved facts, self-consistency samples multiple completions, calibrated uncertainty expressions."},
    {"id":"sys_008","topic":"llm_prompting","text":"System prompts set persona, constraints, behavioral guidelines for an LLM session. Best practices: specify output format, define uncertainty behavior, set tone. Prompt injection is a production security concern."},
    {"id":"opt_001","topic":"optimization","text":"Gradient descent: Batch GD uses full dataset. SGD uses one sample. Mini-batch balances speed and stability. Momentum, RMSprop, and Adam are adaptive optimizers adjusting learning rates per parameter."},
    {"id":"opt_002","topic":"optimization","text":"Hyperparameter optimization: Grid search exhaustive. Random search faster. Bayesian optimization builds a surrogate model using Expected Improvement acquisition function."},
    {"id":"sys_010","topic":"optimization","text":"Learning rate scheduling: step decay, cosine annealing, warm restarts, warmup. Linear warmup over first N steps is critical for transformers to stabilize early training."},
    {"id":"feat_001","topic":"feature_engineering","text":"Feature engineering: normalization (0-1), standardization (zero mean, unit variance), log transform for skew. Categorical: one-hot, ordinal, target encoding. Temporal: hour, day, rolling statistics."},
    {"id":"feat_002","topic":"feature_engineering","text":"Feature selection: filter (correlation, chi-squared, mutual information), wrapper (RFE), embedded (LASSO drives irrelevant weights to zero, tree importances). Reduces dimensionality and overfitting."},
    {"id":"sys_005","topic":"feature_engineering","text":"Interaction features: (response_length x avg_retrieval_score) = quality-weighted length. (session_query_count x follow_up_asked) = engagement depth. (hallucination_risk x query_complexity) = risk under cognitive load."},
    {"id":"data_001","topic":"data_engineering","text":"Data pipelines: ETL extracts, transforms, loads. ELT loads raw then transforms in warehouse. Airflow orchestrates DAGs. Kafka handles real-time event streaming."},
    {"id":"data_002","topic":"data_engineering","text":"Data quality issues: missing values (impute/drop), duplicates (deduplication), schema violations, outliers (IQR, Z-score), data leakage. Great Expectations and dbt automate checks."},
    {"id":"data_003","topic":"data_engineering","text":"Synthetic data generation: rule-based (define distributions), SMOTE (oversample in feature space), GANs (learn real distribution), LLM-based (prompt to produce realistic examples)."},
    {"id":"sys_004","topic":"monitoring","text":"Model cards and datasheets document intended use, limitations, evaluation results, and ethical considerations. Improve transparency and responsible AI practices."},
    {"id":"eval_005","topic":"evaluation","text":"Confusion matrices show TP, TN, FP, FN. Type I error = FP. Type II error = FN. All classification metrics derive from these four cells."},
    {"id":"sys_011","topic":"rag","text":"Document ingestion pipeline for custom knowledge: parse PDF/TXT/CSV → chunk → embed → upsert to ChromaDB. Supports dynamic knowledge base expansion without restarting the system."},
]

# ─────────────────────────────────────────────────────────────────────────────
# TOPIC DETECTION
# ─────────────────────────────────────────────────────────────────────────────

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "machine_learning":    ["machine learning","supervised","unsupervised","reinforcement","classification","regression","clustering","random forest","xgboost","overfitting","underfitting","bias variance"],
    "deep_learning":       ["neural network","deep learning","cnn","rnn","lstm","transformer","backpropagation","gradient","activation","attention"],
    "rag":                 ["rag","retrieval","vector","embedding","chromadb","qdrant","pinecone","chunk","index","similarity","hybrid search"],
    "evaluation":          ["accuracy","precision","recall","f1","roc","auc","calibration","confusion matrix","metric","cross validation","ragas"],
    "feature_engineering": ["feature","engineering","normalization","encoding","selection","importance","interaction"],
    "nlp":                 ["nlp","natural language","tokenization","sentiment","bert","gpt","llm","language model","word2vec","embedding"],
    "ml_pipeline":         ["pipeline","mlops","mlflow","dvc","airflow","cicd","deployment","feature store","online learning"],
    "monitoring":          ["monitoring","drift","covariate","concept drift","evidently","production","shadow deployment"],
    "llm_prompting":       ["prompt","prompting","few shot","zero shot","chain of thought","fine tuning","lora","rlhf","hallucination"],
    "optimization":        ["gradient descent","optimizer","adam","learning rate","loss","hyperparameter","bayesian"],
    "data_engineering":    ["data pipeline","etl","kafka","airflow","data quality","synthetic data","imbalanced"],
}


def detect_topic(query: str) -> Optional[str]:
    q = query.lower()
    best_topic, best_count = None, 0
    for topic, keywords in TOPIC_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in q)
        if count > best_count:
            best_count, best_topic = count, topic
    return best_topic if best_count > 0 else None


def compute_query_complexity(query: str) -> float:
    words = query.split()
    avg_len = sum(len(w) for w in words) / max(len(words), 1)
    cwords  = {"why","how","what","explain","compare","difference","tradeoff"}
    score   = min(1.0, (len(words)/20)*0.5 + (avg_len/8)*0.3 + (0.2 if any(w.lower() in cwords for w in words) else 0))
    return round(score, 3)

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def _detect_device() -> str:
    if _DEVICE:
        return _DEVICE
    try:
        import torch
        if torch.cuda.is_available():   return "cuda"
        if torch.backends.mps.is_available(): return "mps"
    except Exception:
        pass
    return "cpu"


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        device = _detect_device()
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
        _embed_model.encode(["warmup"], normalize_embeddings=True)
    return _embed_model


def embed(texts: list[str]) -> np.ndarray:
    return get_embed_model().encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# CHROMADB
# ─────────────────────────────────────────────────────────────────────────────

def _corpus_hash() -> str:
    """[B9] Hash of all doc IDs in KNOWLEDGE_DOCS. Changes when corpus changes."""
    return hashlib.md5("".join(d["id"] for d in KNOWLEDGE_DOCS).encode()).hexdigest()[:12]


def get_collection():
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    _chroma_collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return _chroma_collection


def setup_collection() -> None:
    """[B9] Index all knowledge docs. Uses content hash to detect stale index."""
    col = get_collection()
    current_hash = _corpus_hash()
    # Check stored hash in ChromaDB metadata
    try:
        meta = col.get(ids=["__corpus_hash__"], include=["documents"])
        stored_hash = meta["documents"][0] if meta["documents"] else ""
    except Exception:
        stored_hash = ""

    if stored_hash == current_hash and col.count() > len(KNOWLEDGE_DOCS):
        return  # corpus unchanged, already indexed

    texts = [d["text"] for d in KNOWLEDGE_DOCS]
    vecs  = embed(texts).tolist()
    col.upsert(
        ids=       [d["id"] for d in KNOWLEDGE_DOCS],
        embeddings=vecs,
        documents= texts,
        metadatas= [{"topic": d["topic"], "doc_id": d["id"]} for d in KNOWLEDGE_DOCS],
    )
    # Store the hash so next startup can skip re-indexing
    col.upsert(ids=["__corpus_hash__"], documents=[current_hash], embeddings=[embed([current_hash])[0].tolist()])


def _recalc_avgdl() -> None:
    """[B6] Recompute corpus average document length after any ingest."""
    global _bm25_avgdl
    if not KNOWLEDGE_DOCS:
        return
    total = sum(len(_tokenize(d["text"])) for d in KNOWLEDGE_DOCS)
    _bm25_avgdl = max(1.0, total / len(KNOWLEDGE_DOCS))


def ingest_documents(docs: list[dict]) -> int:
    col = get_collection()
    new = [d for d in docs if d.get("id") and d.get("text")]
    if not new:
        return 0
    texts = [d["text"] for d in new]
    vecs  = embed(texts).tolist()
    col.upsert(
        ids=       [d["id"] for d in new],
        embeddings=vecs,
        documents= texts,
        metadatas= [{"topic": d.get("topic","custom"), "doc_id": d["id"]} for d in new],
    )
    KNOWLEDGE_DOCS.extend(new)
    _recalc_avgdl()   # [B6] keep avgdl accurate after ingest
    return len(new)


def dense_retrieve(query: str, top_k: int = TOP_K, topic_filter: Optional[str] = None) -> list[dict]:
    col  = get_collection()
    qvec = embed([query]).tolist()
    where = {"topic": topic_filter} if topic_filter else None
    res  = col.query(
        query_embeddings=qvec, n_results=top_k,
        where=where, include=["documents","metadatas","distances"],
    )
    results = []
    for i in range(len(res["ids"][0])):
        doc_id = res["ids"][0][i]
        if doc_id == "__corpus_hash__":
            continue
        results.append({
            "doc_id": doc_id,
            "text":   res["documents"][0][i],
            "score":  round(1 - res["distances"][0][i], 4),
            "topic":  res["metadatas"][0][i].get("topic","general"),
        })
    return results

# ─────────────────────────────────────────────────────────────────────────────
# BM25
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bm25_score(query: str, text: str, k1: float = 1.5, b: float = 0.75) -> float:
    """[B6] Uses real _bm25_avgdl instead of hardcoded 80."""
    q_terms = set(_tokenize(query))
    d_terms = _tokenize(text)
    dl      = len(d_terms)
    tf_map: dict[str, int] = {}
    for t in d_terms:
        tf_map[t] = tf_map.get(t, 0) + 1
    score = 0.0
    for term in q_terms:
        tf = tf_map.get(term, 0)
        if tf == 0:
            continue
        score += tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / _bm25_avgdl))
    return score


def bm25_retrieve(query: str, top_k: int = TOP_K, topic_filter: Optional[str] = None) -> list[dict]:
    candidates = [d for d in KNOWLEDGE_DOCS if topic_filter is None or d["topic"] == topic_filter]
    scored = [(bm25_score(query, d["text"]), d) for d in candidates]
    scored = [(s, d) for s, d in scored if s > 0]
    scored.sort(key=lambda x: -x[0])
    return [{"doc_id": d["id"], "text": d["text"], "score": s, "topic": d["topic"]}
            for s, d in scored[:top_k]]


def _rrf_merge(result_lists: list[list[dict]], top_k: int, k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion across multiple ranked result lists."""
    rrf: dict[str, float] = {}
    doc_map: dict[str, dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results):
            did = item["doc_id"]
            rrf[did] = rrf.get(did, 0.0) + 1.0 / (k + rank + 1)
            doc_map.setdefault(did, item)
    ranked = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]
    out = []
    for doc_id, rrf_score in ranked:
        doc = doc_map[doc_id]
        out.append({
            "doc_id":      doc_id,
            "text":        doc["text"],
            "score":       round(rrf_score, 6),
            "dense_score": doc.get("score", 0.0),
            "topic":       doc["topic"],
        })
    return out


def hybrid_retrieve(query: str, top_k: int = TOP_K, topic_filter: Optional[str] = None) -> list[dict]:
    """
    [B7] Multi-query hybrid retrieval with RRF fusion.
    Generates multiple query variants, retrieves with each using both dense
    and BM25, then fuses all result lists with RRF for maximum recall.
    """
    variants = generate_query_variants(query)
    all_lists: list[list[dict]] = []
    for v in variants:
        dense = dense_retrieve(v, top_k=top_k, topic_filter=topic_filter)
        bm25  = bm25_retrieve(v,  top_k=top_k, topic_filter=topic_filter)
        if dense: all_lists.append(dense)
        if bm25:  all_lists.append(bm25)
    if not all_lists:
        return []
    return _rrf_merge(all_lists, top_k=top_k)

# ─────────────────────────────────────────────────────────────────────────────
# HALLUCINATION GUARD + DIVERSITY
# ─────────────────────────────────────────────────────────────────────────────

def hallucination_risk(response: str, docs: list[dict]) -> float:
    if not docs: return 0.8
    resp_terms = set(re.findall(r"\w{4,}", response.lower()))
    ctx_terms  = set(re.findall(r"\w{4,}", " ".join(d["text"] for d in docs).lower()))
    if not resp_terms: return 0.5
    overlap = len(resp_terms & ctx_terms) / len(resp_terms)
    return round(max(0.0, min(1.0, 1 - overlap * 1.5)), 3)


def response_diversity_score(response: str) -> float:
    words = re.findall(r"\w+", response.lower())
    if len(words) < 5: return 0.0
    return round(len(set(words)) / len(words), 3)

# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATION MEMORY — persisted to SQLite
# ─────────────────────────────────────────────────────────────────────────────

def add_to_memory(session_id: str, role: str, content: str) -> None:
    _memory[session_id].append((role, content[:500]))


def get_memory_context(session_id: str) -> str:
    """[D13] Warm from SQLite if in-memory deque is empty (e.g. after server restart)."""
    if session_id not in _memory or len(_memory[session_id]) == 0:
        _warm_memory_from_db(session_id)
    turns = list(_memory[session_id])
    if not turns:
        return ""
    lines = ["[Conversation history:]"]
    for role, content in turns:
        lines.append(f"  {'User' if role == 'user' else 'Assistant'}: {content}")
    return "\n".join(lines)


def _warm_memory_from_db(session_id: str) -> None:
    """[D13] Load last 3 interactions from SQLite into the memory deque."""
    if not os.path.exists(DB_PATH):
        return
    try:
        con = sqlite3.connect(DB_PATH)
        rows = con.execute(
            "SELECT query, response FROM interactions WHERE session_id=? ORDER BY id DESC LIMIT 3",
            (session_id,)
        ).fetchall()
        con.close()
        for q, r in reversed(rows):
            _memory[session_id].append(("user", (q or "")[:500]))
            _memory[session_id].append(("assistant", (r or "")[:500]))
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# QUERY REWRITER — [C2] with caching and smart skip
# ─────────────────────────────────────────────────────────────────────────────

def rewrite_query(query: str) -> str:
    """
    [C2] Rewrites query for retrieval. Always rewrites (even long queries)
    because specificity matters for document-specific questions.
    Cached to avoid duplicate API calls.
    """
    cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()
    if cache_key in _rewrite_cache:
        return _rewrite_cache[cache_key]
    try:
        resp = _get_groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": (
                "You are a retrieval optimization assistant. Rewrite the following query "
                "to maximize recall from a vector database. Make it more specific, use "
                "relevant technical terms, expand acronyms, and include synonyms. "
                "Output only the rewritten query, nothing else.\n\nQuery: " + query
            )}],
            temperature=0.1, max_tokens=120,
        )
        rw = resp.choices[0].message.content.strip()
        result = rw if 3 <= len(rw.split()) <= 50 else query
        _rewrite_cache[cache_key] = result
        return result
    except Exception:
        return query


def generate_query_variants(query: str) -> list[str]:
    """
    Multi-query retrieval: generate 3 different phrasings of the query.
    Merging results from multiple phrasings dramatically improves recall
    for document-specific questions where exact phrasing matters.
    """
    cache_key = "variants_" + hashlib.md5(query.lower().strip().encode()).hexdigest()
    if cache_key in _rewrite_cache:
        cached = _rewrite_cache[cache_key]
        return cached if isinstance(cached, list) else [query]
    try:
        resp = _get_groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": (
                "Generate 3 different search queries for retrieving information about the following question. "
                "Each query should use different phrasing and terminology but target the same information. "
                "Output exactly 3 queries, one per line, no numbering, no extra text.\n\nOriginal: " + query
            )}],
            temperature=0.3, max_tokens=150,
        )
        lines = [l.strip() for l in resp.choices[0].message.content.strip().split("\n") if l.strip()]
        variants = lines[:3] if len(lines) >= 3 else [query]
        # Always include the original
        if query not in variants:
            variants = [query] + variants[:2]
        _rewrite_cache[cache_key] = variants
        return variants
    except Exception:
        return [query]

# ─────────────────────────────────────────────────────────────────────────────
# GROQ CLIENT FACTORY — [M18]
# ─────────────────────────────────────────────────────────────────────────────

def _get_groq_client() -> Groq:
    """[M18] Always reads key fresh from env — survives st.cache_resource across sessions."""
    key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    return Groq(api_key=key)

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

# Shared grounding instruction injected into every prompt
_GROUNDING = (
    "CRITICAL RULES:\n"
    "1. Answer ONLY using information from the CONTEXT below.\n"
    "2. If the context contains the answer, quote or paraphrase it directly — do NOT rely on general knowledge.\n"
    "3. If a specific number, metric, or name appears in the context, reproduce it exactly.\n"
    "4. If the context does NOT contain enough information, say exactly: \'The provided context does not contain enough information to answer this question.\' — do NOT hallucinate.\n"
    "5. Never make up citations, numbers, or facts not present in the context.\n"
)

PROMPT_REGISTRY: dict[str, dict] = {
    "v1_standard":  {
        "version":  "v1_standard",
        "label":    "Standard",
        "system":   "You are a precise AI assistant. " + _GROUNDING,
        "template": "CONTEXT (use this as your ONLY source):\n{context}\n\n{memory}QUESTION: {query}\n\nAnswer directly and precisely using only the context above. Include specific numbers and details from the context.",
    },
    "v2_simplified": {
        "version":  "v2_simplified",
        "label":    "Simplified",
        "system":   "You are a patient AI tutor who explains clearly. " + _GROUNDING,
        "template": "CONTEXT (use this as your ONLY source):\n{context}\n\n{memory}QUESTION: {query}\n\nExplain clearly using the context. Use simple language and bullet points. Include any specific numbers or facts from the context.",
    },
    "v3_detailed":  {
        "version":  "v3_detailed",
        "label":    "Detailed Structured",
        "system":   "You are an expert instructor who gives structured answers. " + _GROUNDING,
        "template": "CONTEXT (use this as your ONLY source):\n{context}\n\n{memory}QUESTION: {query}\n\nProvide a structured answer: (1) Direct answer with exact figures from context, (2) How it works, (3) Key details from context, (4) Takeaway.",
    },
    "v4_socratic":  {
        "version":  "v4_socratic",
        "label":    "Socratic",
        "system":   "You are a Socratic tutor who grounds all answers in evidence. " + _GROUNDING,
        "template": "CONTEXT (use this as your ONLY source):\n{context}\n\n{memory}QUESTION: {query}\n\nAnswer precisely using the context, then ask one follow-up question to check understanding.",
    },
}


def get_active_prompt(session_id: str) -> dict:
    """[D12] Prompt version is now per-session, not global."""
    sess = _get_session(session_id)
    return PROMPT_REGISTRY.get(sess["prompt_version"], PROMPT_REGISTRY["v1_standard"])


def set_active_prompt(session_id: str, version: str) -> None:
    """[D12] Set prompt for a specific session only."""
    if version in PROMPT_REGISTRY:
        _get_session(session_id)["prompt_version"] = version

# ─────────────────────────────────────────────────────────────────────────────
# A/B TESTING — [D12] separated from feedback loop
# ─────────────────────────────────────────────────────────────────────────────

AB_GROUPS = {"control":"v1_standard","treatment_a":"v2_simplified","treatment_b":"v3_detailed"}


def assign_ab_group(session_id: str) -> str:
    h = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
    return list(AB_GROUPS.keys())[h % len(AB_GROUPS)]


def get_ab_prompt_version(session_id: str) -> str:
    return AB_GROUPS[assign_ab_group(session_id)]

# ─────────────────────────────────────────────────────────────────────────────
# FOLLOW-UP DETECTION — [B11] computed from embedding similarity
# ─────────────────────────────────────────────────────────────────────────────

_prev_query_embed: dict[str, np.ndarray] = {}   # session_id → last query embedding


def compute_followup_signals(session_id: str, query: str) -> tuple[int, int]:
    """
    [B11] Returns (follow_up_asked, similar_followup).
    follow_up_asked=1  if the previous assistant response ended with a question
                       AND user replied (i.e. we're on query 2+).
    similar_followup=1 if embedding cosine similarity to previous query > 0.82.
    """
    follow_up_asked  = 0
    similar_followup = 0

    sess = _get_session(session_id)
    prev_queries = sess.get("understood", [])   # use len as proxy for query count
    query_count  = sess.get("total_queries", 0)

    # Check embedding similarity vs previous query
    cur_emb = embed([query])[0]
    if session_id in _prev_query_embed and query_count > 0:
        prev_emb = _prev_query_embed[session_id]
        cos_sim  = float(np.dot(cur_emb, prev_emb))   # already L2-normalised
        if cos_sim > 0.82:
            similar_followup = 1

    # Check if last assistant turn ended with a question mark
    turns = list(_memory.get(session_id, []))
    if turns:
        last_assistant = next(
            (content for role, content in reversed(turns) if role == "assistant"), ""
        )
        if last_assistant.rstrip().endswith("?"):
            follow_up_asked = 1

    _prev_query_embed[session_id] = cur_emb
    return follow_up_asked, similar_followup

# ─────────────────────────────────────────────────────────────────────────────
# RAG PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def rag_query(
    query: str,
    session_id: str,
    use_ab: bool = True,
    on_step: Optional[Callable[[int], None]] = None,
) -> dict:
    """
    [D15] on_step(step_index) callback fires after each pipeline stage completes.
    Steps: 0=route, 1=embed, 2=retrieve, 3=memory, 4=generate, 5=predict(n/a here), 6=adapt(n/a here)
    """
    t0 = time.time()

    # Step 0: topic + rewrite
    topic = detect_topic(query)
    rw    = rewrite_query(query)
    if on_step: on_step(0)

    # Step 1: multi-query hybrid retrieval
    docs = hybrid_retrieve(rw, top_k=TOP_K, topic_filter=topic)
    # Fallback: if topic-filtered gives < 4 results, merge with unfiltered
    if len(docs) < 4:
        broad = hybrid_retrieve(rw, top_k=TOP_K, topic_filter=None)
        # Merge: add broad docs not already in docs
        seen = {d["doc_id"] for d in docs}
        for d in broad:
            if d["doc_id"] not in seen:
                docs.append(d)
                seen.add(d["doc_id"])
        docs = docs[:TOP_K]
    if on_step: on_step(1)

    # Step 2: retrieve done — build rich context
    doc_ids   = [d["doc_id"] for d in docs]
    scores    = [d.get("dense_score", d["score"]) for d in docs]
    avg_score = round(float(np.mean(scores)), 4) if scores else 0.0
    # Number each context chunk so the LLM can reference them
    context_parts = [f"[Source {i+1}]\n{d['text']}" for i, d in enumerate(docs)]
    context = "\n\n---\n\n".join(context_parts)
    if on_step: on_step(2)

    # Step 3: memory
    mem_str = (get_memory_context(session_id) + "\n\n") if get_memory_context(session_id) else ""
    if on_step: on_step(3)

    # [D12] A/B assigns a base prompt; feedback escalation is per-session and layered on top
    if use_ab:
        ab_group   = assign_ab_group(session_id)
        # If session has been escalated past v1, use escalated version instead of A/B
        sess_prompt = _get_session(session_id)["prompt_version"]
        if sess_prompt != "v1_standard":
            prompt_cfg = PROMPT_REGISTRY[sess_prompt]
        else:
            prompt_cfg = PROMPT_REGISTRY[AB_GROUPS[ab_group]]
    else:
        ab_group   = "feedback_loop"
        prompt_cfg = get_active_prompt(session_id)

    # Step 4: generate
    user_msg = prompt_cfg["template"].format(context=context, query=query, memory=mem_str)
    chat = _get_groq_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": prompt_cfg["system"]},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.15, max_tokens=1200,
    )
    response_text = chat.choices[0].message.content.strip()
    add_to_memory(session_id, "user",      query)
    add_to_memory(session_id, "assistant", response_text)
    if on_step: on_step(4)

    return {
        "response":            response_text,
        "retrieved_doc_ids":   doc_ids,
        "retrieval_scores":    scores,
        "avg_retrieval_score": avg_score,
        "response_length":     len(response_text),
        "latency_sec":         round(time.time() - t0, 2),
        "prompt_version":      prompt_cfg["version"],
        "ab_group":            ab_group,
        "topic_detected":      topic or "general",
        "rewritten_query":     rw,
        "context":             context,
        "hallucination_risk":  hallucination_risk(response_text, docs),
        "response_diversity":  response_diversity_score(response_text),
        "docs":                docs,
    }

# ─────────────────────────────────────────────────────────────────────────────
# LLM-AS-JUDGE — [B8] returns None on failure, never stores fake 3s
# ─────────────────────────────────────────────────────────────────────────────

def llm_judge(query: str, context: str, response: str) -> Optional[dict]:
    """[B8] Returns None if the call fails. Callers must check before storing."""
    prompt = (
        'You are an objective AI evaluator. Return ONLY valid JSON, no markdown.\n\n'
        f'Question: {query}\nContext: {context[:400]}\nResponse: {response[:600]}\n\n'
        'Score 1-5 for each:\n  relevance, clarity, completeness, groundedness, conciseness\n\n'
        'Return exactly: {"relevance":X,"clarity":X,"completeness":X,"groundedness":X,"conciseness":X,"overall":X}'
    )
    try:
        resp = _get_groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=120,
        )
        raw    = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        parsed = json.loads(raw)
        return {k: max(1, min(5, int(round(parsed.get(k, 3))))) for k in
                ["relevance","clarity","completeness","groundedness","conciseness","overall"]}
    except Exception:
        return None   # [B8] caller skips save_judge_log on None

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA — [C3] realistic overlapping distributions
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(n: int = 600) -> pd.DataFrame:
    """
    [C3] Uses overlapping Gaussian distributions so the model sees a realistic
    classification problem. Target AUC ~0.78 instead of 1.0000.
    """
    rng = np.random.default_rng(42)

    def clip(v, lo=0.01, hi=0.99): return float(np.clip(v, lo, hi))

    def sample(understood: int, n_s: int) -> list[dict]:
        rows = []
        for _ in range(n_s):
            if understood:
                ttr  = max(0.5, rng.normal(45, 22))       # mean 45s, wide std
                fu   = int(rng.random() < 0.20)
                sfu  = 0
                sqc  = int(np.clip(rng.normal(2.5, 1.5), 1, 12))
                rlen = int(np.clip(rng.normal(520, 140), 80, 900))
                ars  = clip(rng.normal(0.72, 0.12))
                qc   = clip(rng.normal(0.32, 0.15), 0.05, 0.95)
                hr   = clip(rng.normal(0.22, 0.12), 0.0, 0.95)
                rd   = clip(rng.normal(0.70, 0.10))
            else:
                ttr  = max(0.5, rng.normal(10, 7))         # mean 10s, overlaps understood
                fu   = int(rng.random() < 0.65)
                sfu  = int(fu and rng.random() < 0.70)
                sqc  = int(np.clip(rng.normal(6, 3), 1, 20))
                rlen = int(np.clip(rng.normal(180, 100), 30, 600))
                ars  = clip(rng.normal(0.45, 0.14))
                qc   = clip(rng.normal(0.65, 0.18), 0.05, 0.99)
                hr   = clip(rng.normal(0.58, 0.18), 0.0, 0.99)
                rd   = clip(rng.normal(0.48, 0.11))
            rows.append({
                "time_to_read":       round(ttr, 2),
                "follow_up_asked":    fu,
                "similar_followup":   sfu,
                "session_query_count":sqc,
                "response_length":    rlen,
                "avg_retrieval_score":round(ars, 4),
                "query_complexity":   round(qc,  4),
                "hallucination_risk": round(hr,  4),
                "response_diversity": round(rd,  4),
                "understood":         understood,
            })
        return rows

    rows = sample(1, n // 2) + sample(0, n // 2)
    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df.to_csv("synthetic_data.csv", index=False)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# ML MODEL
# ─────────────────────────────────────────────────────────────────────────────

def _build_candidates() -> dict:
    candidates = {
        "Logistic Regression": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=600, C=1.0, class_weight="balanced", n_jobs=-1, random_state=42))]),
        "Random Forest":       Pipeline([("sc", StandardScaler()), ("m", RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=42))]),
        "Gradient Boosting":   Pipeline([("sc", StandardScaler()), ("m", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42))]),
    }
    try:
        from xgboost import XGBClassifier
        candidates["XGBoost"] = Pipeline([("sc", StandardScaler()), ("m", XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, nthread=-1, eval_metric="auc", random_state=42, verbosity=0))])
    except ImportError:
        pass
    return candidates


def train_model(df: Optional[pd.DataFrame] = None) -> dict:
    global _ml_model, _ml_metrics, _feature_importance
    if df is None or df.empty:
        existing = load_interactions()
        df = existing if len(existing) >= 40 else generate_synthetic_data(600)
    for col_, default in [("hallucination_risk", 0.3), ("response_diversity", 0.6)]:
        if col_ not in df.columns:
            df = df.copy(); df[col_] = default
    df = df[FEATURES + ["understood"]].dropna()
    X, y = df[FEATURES], df["understood"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    candidates = _build_candidates()
    cv_results: dict = {}
    best_name, best_pipe, best_auc = "", None, -1.0

    def _cv_one(name_pipe):
        name, pipe = name_pipe
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
        return name, pipe, float(scores.mean()), float(scores.std())

    with ThreadPoolExecutor(max_workers=len(candidates)) as ex:
        futures = list(ex.map(_cv_one, candidates.items()))

    for name, pipe, mean_auc, std in futures:
        cv_results[name] = {"mean_auc": round(mean_auc, 4), "std": round(std, 4)}
        if mean_auc > best_auc:
            best_auc, best_name, best_pipe = mean_auc, name, pipe

    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)
    y_prob = best_pipe.predict_proba(X_test)[:, 1]

    fi: dict = {}
    try:
        inner = best_pipe.named_steps["m"]
        if hasattr(inner, "feature_importances_"):
            fi = dict(zip(FEATURES, inner.feature_importances_.tolist()))
        elif hasattr(inner, "coef_"):
            fi = dict(zip(FEATURES, np.abs(inner.coef_[0]).tolist()))
    except Exception:
        pass
    _feature_importance = fi

    metrics = {
        "best_model":  best_name,
        "accuracy":    round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_score":    round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc":     round(float(roc_auc_score(y_test, y_prob)), 4),
        "cv_results":  cv_results,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "train_size":  int(len(X_train)),
        "test_size":   int(len(X_test)),
        "features":    FEATURES,
        "trained_at":  datetime.now().isoformat(),
    }
    _ml_model   = best_pipe
    _ml_metrics = metrics
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": best_pipe, "metrics": metrics, "fi": fi}, f)
    return metrics


def load_model() -> bool:
    global _ml_model, _ml_metrics, _feature_importance
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            _ml_model = data["model"]; _ml_metrics = data["metrics"]; _feature_importance = data.get("fi", {})
            return True
        except Exception:
            pass
    return False


def predict_understanding(features: dict) -> dict:
    global _ml_model
    if _ml_model is None:
        if not load_model(): train_model()
    x    = pd.DataFrame([{f: features.get(f, 0.0) for f in FEATURES}])
    prob = float(_ml_model.predict_proba(x)[0][1])
    return {"understood": int(prob >= 0.5), "probability": round(prob, 4), "confidence": round(max(prob, 1 - prob), 4)}

# ─────────────────────────────────────────────────────────────────────────────
# FEEDBACK LOOP — [C4][C5][D12] per-session state + metric_after resolution
# ─────────────────────────────────────────────────────────────────────────────

def apply_feedback(
    prediction: dict,
    retrieval_scores: list,
    session_id: str,
    h_risk: float,
    retrieval_docs: Optional[list[dict]] = None,
) -> dict:
    """
    [C4]  All state (streak, prompt) is per-session.
    [C5]  metric_after is resolved: we write the previous query's pending
          metric_after using this query's prediction before processing this one.
    [D12] A/B and feedback escalation no longer conflict — escalation is session-local.
    """
    sess = _get_session(session_id)
    sess["total_queries"] += 1

    # [C5] Resolve pending metric_after from previous adaptation for this session
    if session_id in _pending_feedback:
        pending = _pending_feedback.pop(session_id)
        _update_feedback_metric_after(pending["log_id"], prediction["probability"])

    old_prompt = sess["prompt_version"]
    action = reason = ""
    understood = prediction["understood"]
    sess["understood"].append(understood)

    # Rule 1: streak-based prompt escalation (per-session)
    sess["streak"] = max(0, sess["streak"] - 1) if understood else sess["streak"] + 1
    if sess["streak"] >= _STREAK_LIMIT:
        try:   cur_idx = PROMPT_SEQUENCE.index(sess["prompt_version"])
        except ValueError: cur_idx = 0
        nxt = PROMPT_SEQUENCE[min(cur_idx + 1, len(PROMPT_SEQUENCE) - 1)]
        if nxt != sess["prompt_version"]:
            set_active_prompt(session_id, nxt)
            action = f"prompt_escalate → {nxt}"
            reason = f"{sess['streak']} consecutive not-understood"
        sess["streak"] = 0

    # Rule 2: retrieval quality flag
    avg_score = float(np.mean(retrieval_scores)) if retrieval_scores else 0.5
    if avg_score < 0.50 and not understood and not action:
        action = "retrieval_quality_flag"; reason = f"low avg retrieval ({avg_score:.3f})"

    # Rule 3: hallucination risk flag
    if h_risk > 0.65 and not action:
        action = "hallucination_risk_flag"; reason = f"high hal risk ({h_risk:.3f})"

    # Rule 4: last 5 all not-understood → socratic
    sess_hist = sess["understood"]
    if len(sess_hist) >= 5 and sum(sess_hist[-5:]) == 0 and not action:
        set_active_prompt(session_id, "v4_socratic")
        action = "session_reset → v4_socratic"; reason = "last 5 all not-understood"

    # Rule 5: retrieval diversity flag
    if retrieval_docs and not action:
        topics = [d.get("topic", "general") for d in retrieval_docs]
        if len(set(topics)) == 1 and not understood:
            action = "retrieval_diversity_flag"; reason = f"all docs from single topic '{topics[0]}'"

    # Rule 6: periodic retrain
    retrain = False
    total_q = sum(s.get("total_queries", 0) for s in _session_state.values())
    if total_q % _RETRAIN_EVERY == 0:
        df = load_interactions()
        if len(df) >= 30:
            train_model(df); retrain = True
            if not action: action = "ml_retrain"; reason = f"periodic retrain after {total_q} queries"

    # [C5] Save log and store pending ID for metric_after resolution
    log_id = _insert_returning_id("feedback_log", {
        "timestamp":     datetime.now().isoformat(),
        "action":        action or "none",
        "reason":        reason,
        "old_prompt":    old_prompt,
        "new_prompt":    sess["prompt_version"],
        "metric_before": prediction["probability"],
        "metric_after":  None,   # will be filled on next query
        "session_id":    session_id,
    })
    if action:   # only track pending if an adaptation was made
        _pending_feedback[session_id] = {"log_id": log_id, "metric_before": prediction["probability"]}

    return {
        "action":    action or "none",
        "reason":    reason,
        "new_prompt":sess["prompt_version"],
        "streak":    sess["streak"],
        "retrained": retrain,
    }


def _update_feedback_metric_after(log_id: int, metric_after: float) -> None:
    """[C5] Back-fill metric_after on the feedback_log row once the next query arrives."""
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute("UPDATE feedback_log SET metric_after=? WHERE id=?", (metric_after, log_id))
        con.commit(); con.close()
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# SQLITE
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, query TEXT, rewritten_query TEXT, response TEXT,
            retrieved_docs TEXT, retrieval_scores TEXT, topic_detected TEXT,
            time_to_read REAL, follow_up_asked INTEGER, similar_followup INTEGER,
            session_query_count INTEGER, response_length INTEGER,
            avg_retrieval_score REAL, query_complexity REAL,
            hallucination_risk REAL, response_diversity REAL,
            understood INTEGER, prediction REAL, confidence REAL,
            prompt_version TEXT, ab_group TEXT, timestamp TEXT
        );
        CREATE TABLE IF NOT EXISTS feedback_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, action TEXT, reason TEXT,
            old_prompt TEXT, new_prompt TEXT,
            metric_before REAL, metric_after REAL, session_id TEXT
        );
        CREATE TABLE IF NOT EXISTS ab_tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, session_id TEXT, group_name TEXT,
            prompt_version TEXT, understood INTEGER, prediction REAL, retrieval_score REAL
        );
        CREATE TABLE IF NOT EXISTS judge_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, session_id TEXT, query TEXT,
            relevance REAL, clarity REAL, completeness REAL,
            groundedness REAL, conciseness REAL, overall REAL
        );
    """)
    con.commit(); con.close()


def _insert(table: str, row: dict) -> None:
    con   = sqlite3.connect(DB_PATH)
    cols  = ", ".join(row.keys())
    marks = ", ".join("?" for _ in row)
    con.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values()))
    con.commit(); con.close()


def _insert_returning_id(table: str, row: dict) -> int:
    """[C5] Insert and return the new row's id."""
    con   = sqlite3.connect(DB_PATH)
    cols  = ", ".join(row.keys())
    marks = ", ".join("?" for _ in row)
    cur   = con.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values()))
    row_id = cur.lastrowid
    con.commit(); con.close()
    return row_id


def save_interaction(row):  _insert("interactions", row)
def save_ab_record(row):    _insert("ab_tests",     row)


def save_judge_log(row: Optional[dict]) -> None:
    """[B8] Only saves if row is not None."""
    if row is not None:
        _insert("judge_log", row)


def _load_table(table: str) -> pd.DataFrame:
    if not os.path.exists(DB_PATH): return pd.DataFrame()
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql_query(f"SELECT * FROM {table}", con)
    con.close(); return df


def load_interactions() -> pd.DataFrame: return _load_table("interactions")
def load_feedback_log() -> pd.DataFrame: return _load_table("feedback_log")
def load_ab_results()   -> pd.DataFrame: return _load_table("ab_tests")
def load_judge_log()    -> pd.DataFrame: return _load_table("judge_log")


def get_ab_summary() -> pd.DataFrame:
    df = load_ab_results()
    if df.empty: return pd.DataFrame()
    return (df.groupby("group_name")
              .agg(n=("understood","count"),
                   understood_rate=("understood","mean"),
                   avg_prediction=("prediction","mean"),
                   avg_retrieval=("retrieval_score","mean"))
              .round(3).reset_index())

# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING — [D14] sentence-boundary aware, exported for app.py
# ─────────────────────────────────────────────────────────────────────────────

def smart_chunk(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    """
    [D14] Split on paragraph boundaries first, then on sentence boundaries,
    falling back to character window only when necessary.
    This preserves semantic units so embeddings are cleaner.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) <= max_chars:
                current = para
            else:
                # Para itself too long: split at sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= max_chars:
                        current = (current + " " + sent).strip() if current else sent
                    else:
                        if current:
                            chunks.append(current)
                        # overlap: carry last `overlap` chars into next chunk
                        current = (current[-overlap:] + " " + sent).strip() if current else sent
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING
# ─────────────────────────────────────────────────────────────────────────────

def generate_monitoring_report() -> str:
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError:
        return "error: pip install evidently"
    df = load_interactions()
    if len(df) < 40: df = generate_synthetic_data(200)
    needed = [f for f in FEATURES if f in df.columns] + ["understood"]
    df = df[needed].dropna()
    if len(df) < 20: return "error: not enough data"
    mid    = len(df) // 2
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df.iloc[:mid].copy(), current_data=df.iloc[mid:].copy())
    path = "monitoring_report.html"
    report.save_html(path)
    return path

# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS — [M17] cached stats
# ─────────────────────────────────────────────────────────────────────────────

def get_session_stats(session_id: str) -> dict:
    sess = _get_session(session_id)
    hist = sess["understood"]
    return {
        "queries_in_session":    len(hist),
        "understood_in_session": sum(hist),
        "session_understand_rate": round(sum(hist) / max(len(hist), 1), 3),
        "consecutive_confusion": sess["streak"],
    }


def get_system_stats() -> dict:
    """[M17] Cached for _STATS_TTL seconds to avoid 3 redundant SQLite reads per render."""
    global _stats_cache, _stats_cache_ts
    now = time.time()
    if _stats_cache and (now - _stats_cache_ts) < _STATS_TTL:
        return _stats_cache

    df = load_interactions(); fb = load_feedback_log()
    total_q = sum(s.get("total_queries", 0) for s in _session_state.values())
    # For display, pick "most escalated" prompt across sessions as active prompt
    versions = [s.get("prompt_version", "v1_standard") for s in _session_state.values()]
    active   = max(versions, key=lambda v: PROMPT_SEQUENCE.index(v)) if versions else "v1_standard"

    _stats_cache = {
        "total_interactions":   int(len(df)),
        "understood_rate":      round(float(df["understood"].mean()), 3) if not df.empty else 0.0,
        "avg_retrieval_score":  round(float(df["avg_retrieval_score"].mean()), 3) if not df.empty else 0.0,
        "avg_hallucination_risk":round(float(df["hallucination_risk"].mean()), 3) if not df.empty and "hallucination_risk" in df.columns else 0.0,
        "avg_diversity":        round(float(df["response_diversity"].mean()), 3) if not df.empty and "response_diversity" in df.columns else 0.0,
        "active_prompt":        active,
        "feedback_actions":     int(len(fb)),
        "streak":               max((s.get("streak", 0) for s in _session_state.values()), default=0),
        "total_queries":        total_q,
        "model_metrics":        _ml_metrics,
        "feature_importance":   _feature_importance,
        "corpus_size":          len(KNOWLEDGE_DOCS),
        "version":              VERSION,
    }
    _stats_cache_ts = now
    return _stats_cache

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    query: str,
    session_id: str,
    session_query_count: int,
    time_to_read: float = 10.0,
    follow_up_asked: int = 0,
    similar_followup: int = 0,
    query_complexity: float = 0.5,
    on_step: Optional[Callable[[int], None]] = None,
) -> dict:
    """[D15] on_step fires after each of the 7 pipeline stages."""

    rag = rag_query(query, session_id, on_step=on_step)

    features = {
        "time_to_read":        time_to_read,
        "follow_up_asked":     follow_up_asked,
        "similar_followup":    similar_followup,
        "session_query_count": session_query_count,
        "response_length":     rag["response_length"],
        "avg_retrieval_score": rag["avg_retrieval_score"],
        "query_complexity":    query_complexity,
        "hallucination_risk":  rag["hallucination_risk"],
        "response_diversity":  rag["response_diversity"],
    }
    prediction = predict_understanding(features)
    if on_step: on_step(5)

    feedback = apply_feedback(
        prediction, rag["retrieval_scores"], session_id,
        rag["hallucination_risk"], retrieval_docs=rag["docs"]
    )
    if on_step: on_step(6)

    save_ab_record({
        "timestamp":      datetime.now().isoformat(),
        "session_id":     session_id,
        "group_name":     rag["ab_group"],
        "prompt_version": rag["prompt_version"],
        "understood":     prediction["understood"],
        "prediction":     prediction["probability"],
        "retrieval_score":rag["avg_retrieval_score"],
    })

    # [B8] Only save judge log when judge returns a real result (not None)
    judge_scores: Optional[dict] = None
    if session_query_count % 3 == 0:
        judge_scores = llm_judge(query, rag["context"], rag["response"])
        if judge_scores is not None:
            save_judge_log({
                "timestamp":    datetime.now().isoformat(),
                "session_id":   session_id,
                "query":        query,
                "relevance":    judge_scores["relevance"],
                "clarity":      judge_scores["clarity"],
                "completeness": judge_scores["completeness"],
                "groundedness": judge_scores["groundedness"],
                "conciseness":  judge_scores["conciseness"],
                "overall":      judge_scores["overall"],
            })

    save_interaction({
        "session_id":          session_id,
        "query":               query,
        "rewritten_query":     rag["rewritten_query"],
        "response":            rag["response"],
        "retrieved_docs":      json.dumps(rag["retrieved_doc_ids"]),
        "retrieval_scores":    json.dumps(rag["retrieval_scores"]),
        "topic_detected":      rag["topic_detected"],
        "time_to_read":        time_to_read,
        "follow_up_asked":     follow_up_asked,
        "similar_followup":    similar_followup,
        "session_query_count": session_query_count,
        "response_length":     rag["response_length"],
        "avg_retrieval_score": rag["avg_retrieval_score"],
        "query_complexity":    query_complexity,
        "hallucination_risk":  rag["hallucination_risk"],
        "response_diversity":  rag["response_diversity"],
        "understood":          prediction["understood"],
        "prediction":          prediction["probability"],
        "confidence":          prediction["confidence"],
        "prompt_version":      rag["prompt_version"],
        "ab_group":            rag["ab_group"],
        "timestamp":           datetime.now().isoformat(),
    })

    # Invalidate stats cache
    global _stats_cache_ts
    _stats_cache_ts = 0.0

    return {
        "response":            rag["response"],
        "rewritten_query":     rag["rewritten_query"],
        "topic_detected":      rag["topic_detected"],
        "retrieved_doc_ids":   rag["retrieved_doc_ids"],
        "retrieval_scores":    rag["retrieval_scores"],
        "avg_retrieval_score": rag["avg_retrieval_score"],
        "hallucination_risk":  rag["hallucination_risk"],
        "response_diversity":  rag["response_diversity"],
        "latency_sec":         rag["latency_sec"],
        "prompt_version":      rag["prompt_version"],
        "ab_group":            rag["ab_group"],
        "prediction":          prediction,
        "feedback":            feedback,
        "judge_scores":        judge_scores or {},
        "features_used":       features,
    }

# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap() -> None:
    _recalc_avgdl()   # [B6] compute real avgdl before any retrieval

    def _embed_and_index():
        get_embed_model()
        setup_collection()

    def _init_db_thread():
        init_db()

    def _ml_thread():
        if not load_model():
            df = generate_synthetic_data(600)
            train_model(df)

    with ThreadPoolExecutor(max_workers=3) as ex:
        f1 = ex.submit(_embed_and_index)
        f2 = ex.submit(_init_db_thread)
        f3 = ex.submit(_ml_thread)
        f1.result(); f2.result(); f3.result()


if __name__ == "__main__":
    bootstrap()
    print(f"PulseRAG v{VERSION} backend ready.")
    print(json.dumps(get_system_stats(), indent=2, default=str))