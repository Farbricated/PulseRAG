"""
PulseRAG v5 — backend.py
========================
Fixes applied vs v4:
  [F1]  Low-quality doc filtering: docs below score threshold excluded from context
  [F2]  Cross-encoder re-ranking via sentence-transformers cross-encoder
  [F3]  Interaction feature engineering: 3 compound features added to FEATURES
  [F4]  Prediction threshold calibrated via precision-recall curve, not hardcoded 0.5
  [F5]  RAGAS-style metrics: context_precision, context_recall, faithfulness computed
  [F6]  LLM Judge runs every query (not every 3rd); manual trigger supported
  [F7]  Judge vs ML cross-correlation computed and stored
  [F8]  Feedback rules 2,3,5 now actively change system behavior (not just log flags)
  [F9]  Bi-directional prompt: de-escalation when streak_good >= 3
  [F10] Feedback effectiveness tracking: before/after understanding rate per rule
  [F11] Latency monitoring: p50/p95/p99 tracked per session and globally
  [F12] Drift detection automated: triggers retrain when PSI > threshold
  [F13] Monitoring uses real data only, never synthetic fallback for drift report
  [F14] Unit-testable pure functions separated from side-effectful ones
  [F15] Graceful Groq degradation: returns structured error dict, never raw exception
  [F16] Calibrated threshold via isotonic regression on held-out data

Streamlit Cloud fixes (P-series):
  [P1]  Cross-encoder disabled by default — set PULSERAG_RERANK=1 env var to enable it.
        Saves ~300MB RAM on cold start which was causing OOM crash on Streamlit Cloud.
  [P2]  Bootstrap is sequential (not parallel threads) to avoid simultaneous model-load
        OOM spike. Embedding + ChromaDB + ML training no longer race for RAM at startup.
  [P3]  Synthetic training data reduced to 300 samples at bootstrap (was 600).
  [P4]  GROQ_API_KEY always re-read from env at call-time so Streamlit secrets work.
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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import mlflow
    import mlflow.sklearn

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pulserag_knowledge_v5"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
DB_PATH = "pulserag.db"
MODEL_PATH = "pulserag_model.pkl"
TOP_K = 10
RETRIEVAL_SCORE_THRESHOLD = 0.30  # [F1] docs below this are filtered
VERSION = "5.0.0"
EMBED_BATCH_SIZE = 64

_DEVICE = os.getenv("PULSERAG_DEVICE", None)

# [P1] Cross-encoder disabled by default to save ~300MB RAM on Streamlit Cloud.
# Set PULSERAG_RERANK=1 in your environment to enable it (e.g. local Docker/EC2).
_RERANK_ENABLED = os.getenv("PULSERAG_RERANK", "0") == "1"

# [F3] Extended feature set with 3 interaction features
FEATURES = [
    "time_to_read",
    "follow_up_asked",
    "similar_followup",
    "session_query_count",
    "response_length",
    "avg_retrieval_score",
    "query_complexity",
    "hallucination_risk",
    "response_diversity",
    # [F3] Interaction features
    "quality_weighted_length",  # response_length * avg_retrieval_score
    "engagement_depth",  # session_query_count * follow_up_asked
    "risk_under_load",  # hallucination_risk * query_complexity
]

_STREAK_LIMIT = 3
_STREAK_GOOD = 3  # [F9] de-escalation trigger
_RETRAIN_EVERY = 30
_PSI_THRESHOLD = 0.20  # [F12] population stability index drift threshold
PROMPT_SEQUENCE = ["v1_standard", "v2_simplified", "v3_detailed", "v4_socratic"]

# ── Singletons ────────────────────────────────────────────────────────────────
_ml_model: object = None
_ml_metrics: dict = {}
_feature_importance: dict = {}
_opt_threshold: float = 0.5  # [F4] calibrated, not hardcoded
_embed_model: object = None
_cross_encoder: object = None
_chroma_collection: object = None

# ── Online learning (SGDClassifier partial_fit) ───────────────────────────────
_online_model: object = None  # SGDClassifier — updated after every query
_online_metrics: dict = {}  # tracked separately from batch model
_online_buffer: list = []  # rolling buffer of (features, label) tuples
_ONLINE_BUFFER_SIZE: int = 50  # flush to partial_fit every N samples
_online_query_count: int = 0  # total queries seen by online model

# ── Per-session state ─────────────────────────────────────────────────────────
_session_state: dict[str, dict] = {}
_memory: dict[str, deque] = defaultdict(lambda: deque(maxlen=6))
_pending_feedback: dict[str, dict] = {}
_rewrite_cache: dict[str, str] = {}
_bm25_avgdl: float = 80.0

# ── Stats cache ───────────────────────────────────────────────────────────────
_stats_cache: dict = {}
_stats_cache_ts: float = 0.0
_STATS_TTL: float = 3.0

# ── Latency tracking [F11] ────────────────────────────────────────────────────
_latency_log: list[float] = []


def _get_session(session_id: str) -> dict:
    if session_id not in _session_state:
        _session_state[session_id] = {
            "streak": 0,
            "streak_good": 0,  # [F9]
            "prompt_version": "v1_standard",
            "understood": [],
            "total_queries": 0,
            "retrieval_boost": 0.0,  # [F8] active retrieval score boost
            "min_score_override": None,  # [F8] dynamic score threshold
        }
    return _session_state[session_id]


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_DOCS: list[dict] = [
    {
        "id": "ml_001",
        "topic": "machine_learning",
        "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves. The process begins with observations—examples, direct experience, or instruction—so that computers can make better decisions in the future without human intervention for each case.",  # noqa: E501
    },
    {
        "id": "ml_002",
        "topic": "machine_learning",
        "text": "Supervised learning trains models on labeled input-output pairs. Common algorithms include linear regression, logistic regression, decision trees, random forests, support vector machines, and neural networks. Model performance is measured on held-out test data to estimate generalization.",  # noqa: E501
    },
    {
        "id": "ml_003",
        "topic": "machine_learning",
        "text": "Unsupervised learning finds hidden patterns in unlabeled data. K-means clustering partitions data into k groups by minimizing within-cluster variance. DBSCAN finds density-based clusters. PCA and t-SNE compress high-dimensional data for visualization and feature extraction.",  # noqa: E501
    },
    {
        "id": "ml_004",
        "topic": "machine_learning",
        "text": "Reinforcement learning trains agents to maximize cumulative reward by interacting with an environment. Key algorithms: Q-learning, Deep Q-Networks, Actor-Critic, PPO. Used in robotics, games, and recommendation systems.",  # noqa: E501
    },
    {
        "id": "ml_005",
        "topic": "machine_learning",
        "text": "Ensemble methods combine multiple models. Bagging (Random Forest) trains on random subsamples. Boosting (AdaBoost, GBM, XGBoost) trains sequentially. Stacking uses a meta-learner to combine base predictions.",  # noqa: E501
    },
    {
        "id": "ml_006",
        "topic": "machine_learning",
        "text": "Bias-variance tradeoff is fundamental. High bias = underfitting. High variance = overfitting. Regularization (L1 Lasso, L2 Ridge, elastic net) penalizes complexity to balance this tradeoff.",  # noqa: E501
    },
    {
        "id": "sys_006",
        "topic": "machine_learning",
        "text": "Imbalanced classification strategies: SMOTE oversample, random undersample, class_weight='balanced'. Evaluation must use F1, ROC-AUC, or Precision-Recall AUC — never raw accuracy on imbalanced datasets.",  # noqa: E501
    },
    {
        "id": "dl_001",
        "topic": "deep_learning",
        "text": "Deep learning uses neural networks with many layers. Backpropagation computes gradients via the chain rule. Gradient descent updates weights to minimize loss. Excels at vision, NLP, and speech tasks.",  # noqa: E501
    },
    {
        "id": "dl_002",
        "topic": "deep_learning",
        "text": "Convolutional Neural Networks process grid-like data. Convolutional layers detect local patterns. Max pooling reduces spatial dimensions. Notable architectures: AlexNet, VGG, ResNet, EfficientNet.",  # noqa: E501
    },
    {
        "id": "dl_003",
        "topic": "deep_learning",
        "text": "Recurrent Neural Networks process sequential data via hidden state. LSTMs use gating mechanisms. GRUs are simpler. Used in time-series, speech, and text generation.",  # noqa: E501
    },
    {
        "id": "dl_004",
        "topic": "deep_learning",
        "text": "The Transformer uses self-attention to process all tokens in parallel. Multi-head attention attends to different subspaces simultaneously. BERT (encoder-only), GPT (decoder-only), T5 (encoder-decoder).",  # noqa: E501
    },
    {
        "id": "dl_005",
        "topic": "deep_learning",
        "text": "Transfer learning reuses pretrained models. Fine-tuning updates all or some weights. Feature extraction freezes pretrained weights and trains only a new head. Dramatically reduces data and compute required.",  # noqa: E501
    },
    {
        "id": "sys_007",
        "topic": "deep_learning",
        "text": "Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V. Multi-head attention runs several attention operations in parallel and concatenates results.",  # noqa: E501
    },
    {
        "id": "rag_001",
        "topic": "rag",
        "text": "Retrieval-Augmented Generation grounds LLM responses in external knowledge. Pipeline: embed query → search vector DB → inject top-k docs as context → generate conditioned response. Reduces hallucination and enables knowledge updates without retraining.",  # noqa: E501
    },
    {
        "id": "rag_002",
        "topic": "rag",
        "text": "Vector databases store embeddings for ANN search. HNSW enables sub-linear complexity. ChromaDB is open-source and local-first. Pinecone, Weaviate, Milvus are managed alternatives.",  # noqa: E501
    },
    {
        "id": "rag_003",
        "topic": "rag",
        "text": "Chunking strategy determines retrieval quality. Fixed-size, sentence-based, recursive, and semantic chunking are common approaches. Optimal chunk size 256-1024 tokens depends on embedding model context window.",  # noqa: E501
    },
    {
        "id": "rag_004",
        "topic": "rag",
        "text": "Advanced RAG: HyDE generates hypothetical answers for retrieval. Query rewriting uses LLM to reformulate. Multi-query retrieval merges results. Re-ranking with cross-encoders improves precision.",  # noqa: E501
    },
    {
        "id": "rag_005",
        "topic": "rag",
        "text": "Hybrid search combines dense vector search with sparse BM25. Dense retrieval captures semantic meaning. Sparse retrieval handles exact keyword matching. RRF merges ranked lists without score normalization.",  # noqa: E501
    },
    {
        "id": "sys_001",
        "topic": "rag",
        "text": "Agentic RAG extends basic RAG with tool-use and multi-step reasoning. Agents decompose complex queries into sub-questions and iteratively retrieve additional context. LangChain, LlamaIndex, OpenAI Assistants provide frameworks.",  # noqa: E501
    },
    {
        "id": "eval_001",
        "topic": "evaluation",
        "text": "Classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC. Macro averaging treats all classes equally; weighted averaging accounts for class imbalance.",  # noqa: E501
    },
    {
        "id": "eval_002",
        "topic": "evaluation",
        "text": "Cross-validation: K-fold CV splits data into k partitions. Stratified k-fold preserves class proportions — essential for imbalanced datasets. Nested CV avoids optimistic bias.",  # noqa: E501
    },
    {
        "id": "eval_003",
        "topic": "evaluation",
        "text": "RAGAS metrics: context_precision, context_recall, faithfulness, answer_relevancy. LLM-as-Judge uses a strong model to score responses without human annotation.",  # noqa: E501
    },
    {
        "id": "eval_004",
        "topic": "evaluation",
        "text": "Calibration measures if predicted probabilities reflect true likelihoods. Platt scaling and isotonic regression are post-hoc calibration methods. Expected Calibration Error quantifies deviation.",  # noqa: E501
    },
    {
        "id": "sys_003",
        "topic": "evaluation",
        "text": "A/B testing compares system variants by routing different user segments to different versions. Statistical significance via t-test or chi-squared with alpha=0.05. Cohen's d measures practical significance.",  # noqa: E501
    },
    {
        "id": "nlp_001",
        "topic": "nlp",
        "text": "NLP enables computers to understand human language. Core tasks: tokenization, POS tagging, NER, dependency parsing. Higher-level tasks: sentiment analysis, machine translation, summarization. Modern NLP uses pre-trained transformers.",  # noqa: E501
    },
    {
        "id": "nlp_002",
        "topic": "nlp",
        "text": "Word embeddings: Word2Vec trains on co-occurrence. GloVe uses global statistics. FastText handles OOV via subword n-grams. BERT and RoBERTa produce contextual embeddings capturing polysemy.",  # noqa: E501
    },
    {
        "id": "nlp_003",
        "topic": "nlp",
        "text": "LLMs like GPT-4, Llama 3, Claude are transformer-based models trained on next-token prediction. Emergent capabilities: in-context learning, chain-of-thought reasoning, instruction following.",  # noqa: E501
    },
    {
        "id": "sys_009",
        "topic": "nlp",
        "text": "Tokenization: BPE merges frequent character pairs. WordPiece (BERT) and SentencePiece (T5, LLaMA) are variants. One token is approximately 4 characters in English.",  # noqa: E501
    },
    {
        "id": "pipe_001",
        "topic": "ml_pipeline",
        "text": "ML pipelines automate: data ingestion, validation, preprocessing, feature engineering, training, evaluation, deployment. MLflow tracks experiments. DVC versions datasets.",  # noqa: E501
    },
    {
        "id": "pipe_002",
        "topic": "ml_pipeline",
        "text": "MLOps applies DevOps to ML: triggered retraining, automated validation gates, canary deployments, automated rollback. Tools: GitHub Actions, Docker, Kubernetes.",  # noqa: E501
    },
    {
        "id": "sys_002",
        "topic": "ml_pipeline",
        "text": "Online learning updates models incrementally without retraining from scratch. Critical when distributions shift over time. Concept drift detection (ADWIN, DDM) can trigger updates.",  # noqa: E501
    },
    {
        "id": "drift_001",
        "topic": "monitoring",
        "text": "Data drift: KS-test for continuous features, chi-squared for categorical, PSI and Jensen-Shannon divergence for distribution shift detection.",  # noqa: E501
    },
    {
        "id": "drift_002",
        "topic": "monitoring",
        "text": "Production ML monitoring tracks: prediction distribution, input feature statistics, data quality, model latency (p50/p95/p99), throughput, error rates, and business KPIs.",  # noqa: E501
    },
    {
        "id": "drift_003",
        "topic": "monitoring",
        "text": "Shadow deployment runs a new model in parallel without serving predictions to users. Validates new model on real traffic before promotion. A/B testing measures business metrics.",  # noqa: E501
    },
    {
        "id": "llm_001",
        "topic": "llm_prompting",
        "text": "Prompt engineering: zero-shot (task only), few-shot (examples included), chain-of-thought (step-by-step reasoning), ReAct (reasoning + tool use alternating).",  # noqa: E501
    },
    {
        "id": "llm_002",
        "topic": "llm_prompting",
        "text": "Fine-tuning: full fine-tuning updates all weights. LoRA injects small trainable matrices into attention layers, reducing trainable parameters by 10,000x. QLoRA adds 4-bit quantization.",  # noqa: E501
    },
    {
        "id": "llm_003",
        "topic": "llm_prompting",
        "text": "LLM hallucination: plausible but factually incorrect content. Mitigations: RAG grounds responses in retrieved facts, self-consistency samples multiple completions, calibrated uncertainty expressions.",  # noqa: E501
    },
    {
        "id": "sys_008",
        "topic": "llm_prompting",
        "text": "System prompts set persona, constraints, behavioral guidelines for an LLM session. Best practices: specify output format, define uncertainty behavior, set tone. Prompt injection is a production security concern.",  # noqa: E501
    },
    {
        "id": "opt_001",
        "topic": "optimization",
        "text": "Gradient descent: Batch GD uses full dataset. SGD uses one sample. Mini-batch balances speed and stability. Momentum, RMSprop, and Adam are adaptive optimizers adjusting learning rates per parameter.",  # noqa: E501
    },
    {
        "id": "opt_002",
        "topic": "optimization",
        "text": "Hyperparameter optimization: Grid search exhaustive. Random search faster. Bayesian optimization builds a surrogate model using Expected Improvement acquisition function.",  # noqa: E501
    },
    {
        "id": "sys_010",
        "topic": "optimization",
        "text": "Learning rate scheduling: step decay, cosine annealing, warm restarts, warmup. Linear warmup over first N steps is critical for transformers to stabilize early training.",  # noqa: E501
    },
    {
        "id": "feat_001",
        "topic": "feature_engineering",
        "text": "Feature engineering: normalization (0-1), standardization (zero mean, unit variance), log transform for skew. Categorical: one-hot, ordinal, target encoding. Temporal: hour, day, rolling statistics.",  # noqa: E501
    },
    {
        "id": "feat_002",
        "topic": "feature_engineering",
        "text": "Feature selection: filter (correlation, chi-squared, mutual information), wrapper (RFE), embedded (LASSO drives irrelevant weights to zero, tree importances). Reduces dimensionality and overfitting.",  # noqa: E501
    },
    {
        "id": "sys_005",
        "topic": "feature_engineering",
        "text": "Interaction features: (response_length x avg_retrieval_score) = quality-weighted length. (session_query_count x follow_up_asked) = engagement depth. (hallucination_risk x query_complexity) = risk under cognitive load.",  # noqa: E501
    },
    {
        "id": "data_001",
        "topic": "data_engineering",
        "text": "Data pipelines: ETL extracts, transforms, loads. ELT loads raw then transforms in warehouse. Airflow orchestrates DAGs. Kafka handles real-time event streaming.",  # noqa: E501
    },
    {
        "id": "data_002",
        "topic": "data_engineering",
        "text": "Data quality issues: missing values (impute/drop), duplicates (deduplication), schema violations, outliers (IQR, Z-score), data leakage. Great Expectations and dbt automate checks.",  # noqa: E501
    },
    {
        "id": "data_003",
        "topic": "data_engineering",
        "text": "Synthetic data generation: rule-based (define distributions), SMOTE (oversample in feature space), GANs (learn real distribution), LLM-based (prompt to produce realistic examples).",  # noqa: E501
    },
    {
        "id": "sys_004",
        "topic": "monitoring",
        "text": "Model cards and datasheets document intended use, limitations, evaluation results, and ethical considerations. Improve transparency and responsible AI practices.",  # noqa: E501
    },
    {
        "id": "eval_005",
        "topic": "evaluation",
        "text": "Confusion matrices show TP, TN, FP, FN. Type I error = FP. Type II error = FN. All classification metrics derive from these four cells.",  # noqa: E501
    },
    {
        "id": "sys_011",
        "topic": "rag",
        "text": "Document ingestion pipeline for custom knowledge: parse PDF/TXT/CSV → chunk → embed → upsert to ChromaDB. Supports dynamic knowledge base expansion without restarting the system.",  # noqa: E501
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# TOPIC DETECTION
# ─────────────────────────────────────────────────────────────────────────────

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "machine_learning": [
        "machine learning",
        "supervised",
        "unsupervised",
        "reinforcement",
        "classification",
        "regression",
        "clustering",
        "random forest",
        "xgboost",
        "overfitting",
        "underfitting",
        "bias variance",
    ],
    "deep_learning": [
        "neural network",
        "deep learning",
        "cnn",
        "rnn",
        "lstm",
        "transformer",
        "backpropagation",
        "gradient",
        "activation",
        "attention",
    ],
    "rag": [
        "rag",
        "retrieval",
        "vector",
        "embedding",
        "chromadb",
        "qdrant",
        "pinecone",
        "chunk",
        "index",
        "similarity",
        "hybrid search",
    ],
    "evaluation": [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc",
        "auc",
        "calibration",
        "confusion matrix",
        "metric",
        "cross validation",
        "ragas",
    ],
    "feature_engineering": [
        "feature",
        "engineering",
        "normalization",
        "encoding",
        "selection",
        "importance",
        "interaction",
    ],
    "nlp": [
        "nlp",
        "natural language",
        "tokenization",
        "sentiment",
        "bert",
        "gpt",
        "llm",
        "language model",
        "word2vec",
        "embedding",
    ],
    "ml_pipeline": [
        "pipeline",
        "mlops",
        "mlflow",
        "dvc",
        "airflow",
        "cicd",
        "deployment",
        "feature store",
        "online learning",
    ],
    "monitoring": [
        "monitoring",
        "drift",
        "covariate",
        "concept drift",
        "evidently",
        "production",
        "shadow deployment",
    ],
    "llm_prompting": [
        "prompt",
        "prompting",
        "few shot",
        "zero shot",
        "chain of thought",
        "fine tuning",
        "lora",
        "rlhf",
        "hallucination",
    ],
    "optimization": [
        "gradient descent",
        "optimizer",
        "adam",
        "learning rate",
        "loss",
        "hyperparameter",
        "bayesian",
    ],
    "data_engineering": [
        "data pipeline",
        "etl",
        "kafka",
        "airflow",
        "data quality",
        "synthetic data",
        "imbalanced",
    ],
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
    cwords = {"why", "how", "what", "explain", "compare", "difference", "tradeoff"}
    score = min(
        1.0,
        (len(words) / 20) * 0.5
        + (avg_len / 8) * 0.3
        + (0.2 if any(w.lower() in cwords for w in words) else 0),
    )
    return round(score, 3)


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────


def _detect_device() -> str:
    if _DEVICE:
        return _DEVICE
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
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


def get_cross_encoder():
    """[F2][P1] Lazy-load cross-encoder only when PULSERAG_RERANK=1 is set.
    Skipped by default on Streamlit Cloud to save ~300MB RAM."""
    global _cross_encoder
    if not _RERANK_ENABLED:
        return None
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder

            _cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)
        except Exception:
            _cross_encoder = None
    return _cross_encoder


# ─────────────────────────────────────────────────────────────────────────────
# CHROMADB
# ─────────────────────────────────────────────────────────────────────────────


def _corpus_hash() -> str:
    return hashlib.md5("".join(d["id"] for d in KNOWLEDGE_DOCS).encode()).hexdigest()[
        :12
    ]


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
    col = get_collection()
    current_hash = _corpus_hash()
    try:
        meta = col.get(ids=["__corpus_hash__"], include=["documents"])
        stored_hash = meta["documents"][0] if meta["documents"] else ""
    except Exception:
        stored_hash = ""

    if stored_hash == current_hash and col.count() > len(KNOWLEDGE_DOCS):
        return

    texts = [d["text"] for d in KNOWLEDGE_DOCS]
    vecs = embed(texts).tolist()
    col.upsert(
        ids=[d["id"] for d in KNOWLEDGE_DOCS],
        embeddings=vecs,
        documents=texts,
        metadatas=[{"topic": d["topic"], "doc_id": d["id"]} for d in KNOWLEDGE_DOCS],
    )
    col.upsert(
        ids=["__corpus_hash__"],
        documents=[current_hash],
        embeddings=[embed([current_hash])[0].tolist()],
    )


def _recalc_avgdl() -> None:
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
    vecs = embed(texts).tolist()
    col.upsert(
        ids=[d["id"] for d in new],
        embeddings=vecs,
        documents=texts,
        metadatas=[{"topic": d.get("topic", "custom"), "doc_id": d["id"]} for d in new],
    )
    KNOWLEDGE_DOCS.extend(new)
    _recalc_avgdl()
    return len(new)


def dense_retrieve(
    query: str, top_k: int = TOP_K, topic_filter: Optional[str] = None
) -> list[dict]:
    col = get_collection()
    qvec = embed([query]).tolist()
    where = {"topic": topic_filter} if topic_filter else None
    res = col.query(
        query_embeddings=qvec,
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    results = []
    for i in range(len(res["ids"][0])):
        doc_id = res["ids"][0][i]
        if doc_id == "__corpus_hash__":
            continue
        results.append(
            {
                "doc_id": doc_id,
                "text": res["documents"][0][i],
                "score": round(1 - res["distances"][0][i], 4),
                "topic": res["metadatas"][0][i].get("topic", "general"),
            }
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# BM25
# ─────────────────────────────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bm25_score(query: str, text: str, k1: float = 1.5, b: float = 0.75) -> float:
    q_terms = set(_tokenize(query))
    d_terms = _tokenize(text)
    dl = len(d_terms)
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


def bm25_retrieve(
    query: str, top_k: int = TOP_K, topic_filter: Optional[str] = None
) -> list[dict]:
    candidates = [
        d for d in KNOWLEDGE_DOCS if topic_filter is None or d["topic"] == topic_filter
    ]
    scored = [(bm25_score(query, d["text"]), d) for d in candidates]
    scored = [(s, d) for s, d in scored if s > 0]
    scored.sort(key=lambda x: -x[0])
    return [
        {"doc_id": d["id"], "text": d["text"], "score": s, "topic": d["topic"]}
        for s, d in scored[:top_k]
    ]


def _rrf_merge(result_lists: list[list[dict]], top_k: int, k: int = 60) -> list[dict]:
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
        out.append(
            {
                "doc_id": doc_id,
                "text": doc["text"],
                "score": round(rrf_score, 6),
                "dense_score": doc.get("score", 0.0),
                "topic": doc["topic"],
            }
        )
    return out


def rerank_docs(query: str, docs: list[dict], top_k: int = 6) -> list[dict]:
    """[F2] Cross-encoder re-ranking. Falls back to original order if unavailable."""
    ce = get_cross_encoder()
    if ce is None or len(docs) == 0:
        return docs[:top_k]
    try:
        pairs = [(query, d["text"]) for d in docs]
        scores = ce.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: -x[0])
        for score, doc in ranked:
            doc["rerank_score"] = round(float(score), 4)
        return [doc for _, doc in ranked[:top_k]]
    except Exception:
        return docs[:top_k]


def filter_by_score(
    docs: list[dict], threshold: float, session_override: Optional[float] = None
) -> list[dict]:
    """[F1] Remove docs below quality threshold. Session can lower threshold dynamically."""
    eff_threshold = session_override if session_override is not None else threshold
    filtered = [
        d for d in docs if d.get("dense_score", d.get("score", 0)) >= eff_threshold
    ]
    # Always keep at least 2 docs to avoid empty context
    return filtered if len(filtered) >= 2 else docs[:2]


def hybrid_retrieve(
    query: str, top_k: int = TOP_K, topic_filter: Optional[str] = None
) -> list[dict]:
    variants = generate_query_variants(query)
    all_lists: list[list[dict]] = []
    for v in variants:
        dense = dense_retrieve(v, top_k=top_k, topic_filter=topic_filter)
        bm25 = bm25_retrieve(v, top_k=top_k, topic_filter=topic_filter)
        if dense:
            all_lists.append(dense)
        if bm25:
            all_lists.append(bm25)
    if not all_lists:
        return []
    return _rrf_merge(all_lists, top_k=top_k)


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS-STYLE METRICS [F5]
# ─────────────────────────────────────────────────────────────────────────────


def compute_context_precision(query: str, docs: list[dict], top_k: int = 5) -> float:
    """
    [F5] Context precision: fraction of retrieved docs that are relevant to the query.
    Uses embedding cosine similarity > 0.4 as relevance proxy.
    """
    if not docs:
        return 0.0
    q_emb = embed([query])[0]
    scores = []
    for doc in docs[:top_k]:
        d_emb = embed([doc["text"]])[0]
        sim = float(np.dot(q_emb, d_emb))
        scores.append(1 if sim > 0.4 else 0)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def compute_context_recall(response: str, docs: list[dict]) -> float:
    """
    [F5] Context recall: fraction of response content that is grounded in context.
    Uses term overlap as proxy (same logic as hallucination risk, inverted).
    """
    if not docs or not response:
        return 0.0
    resp_terms = set(re.findall(r"\w{4,}", response.lower()))
    ctx_terms = set(re.findall(r"\w{4,}", " ".join(d["text"] for d in docs).lower()))
    if not resp_terms:
        return 0.0
    overlap = len(resp_terms & ctx_terms) / len(resp_terms)
    return round(min(1.0, overlap * 1.3), 4)


def compute_faithfulness(response: str, docs: list[dict]) -> float:
    """
    [F5] Faithfulness: proportion of response claims supported by context.
    Sentence-level: each sentence checked for term overlap with context.
    """
    if not docs or not response:
        return 0.0
    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", response) if len(s.strip()) > 10
    ]
    ctx_terms = set(re.findall(r"\w{4,}", " ".join(d["text"] for d in docs).lower()))
    supported = 0
    for sent in sentences:
        s_terms = set(re.findall(r"\w{4,}", sent.lower()))
        if s_terms and len(s_terms & ctx_terms) / len(s_terms) >= 0.3:
            supported += 1
    return round(supported / len(sentences), 4) if sentences else 0.0


def compute_ragas_metrics(query: str, response: str, docs: list[dict]) -> dict:
    """[F5] All three RAGAS-style metrics in one call."""
    return {
        "context_precision": compute_context_precision(query, docs),
        "context_recall": compute_context_recall(response, docs),
        "faithfulness": compute_faithfulness(response, docs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HALLUCINATION GUARD + DIVERSITY
# ─────────────────────────────────────────────────────────────────────────────


def hallucination_risk(response: str, docs: list[dict]) -> float:
    if not docs:
        return 0.8
    resp_terms = set(re.findall(r"\w{4,}", response.lower()))
    ctx_terms = set(re.findall(r"\w{4,}", " ".join(d["text"] for d in docs).lower()))
    if not resp_terms:
        return 0.5
    overlap = len(resp_terms & ctx_terms) / len(resp_terms)
    return round(max(0.0, min(1.0, 1 - overlap * 1.5)), 3)


def response_diversity_score(response: str) -> float:
    words = re.findall(r"\w+", response.lower())
    if len(words) < 5:
        return 0.0
    return round(len(set(words)) / len(words), 3)


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATION MEMORY
# ─────────────────────────────────────────────────────────────────────────────


def add_to_memory(session_id: str, role: str, content: str) -> None:
    _memory[session_id].append((role, content[:500]))


def get_memory_context(session_id: str) -> str:
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
    if not os.path.exists(DB_PATH):
        return
    try:
        con = sqlite3.connect(DB_PATH)
        rows = con.execute(
            "SELECT query, response FROM interactions WHERE session_id=? ORDER BY id DESC LIMIT 3",
            (session_id,),
        ).fetchall()
        con.close()
        for q, r in reversed(rows):
            _memory[session_id].append(("user", (q or "")[:500]))
            _memory[session_id].append(("assistant", (r or "")[:500]))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# QUERY REWRITER
# ─────────────────────────────────────────────────────────────────────────────


def rewrite_query(query: str) -> str:
    cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()
    if cache_key in _rewrite_cache:
        return _rewrite_cache[cache_key]
    try:
        resp = _get_groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Rewrite the following query to maximize recall from a vector database. "
                        "Make it more specific, use relevant technical terms, expand acronyms, and include synonyms. "
                        "Output only the rewritten query, nothing else.\n\nQuery: "
                        + query
                    ),
                }
            ],
            temperature=0.1,
            max_tokens=120,
        )
        rw = resp.choices[0].message.content.strip()
        result = rw if 3 <= len(rw.split()) <= 50 else query
        _rewrite_cache[cache_key] = result
        return result
    except Exception:
        return query


def generate_query_variants(query: str) -> list[str]:
    cache_key = "variants_" + hashlib.md5(query.lower().strip().encode()).hexdigest()
    if cache_key in _rewrite_cache:
        cached = _rewrite_cache[cache_key]
        return cached if isinstance(cached, list) else [query]
    try:
        resp = _get_groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Generate 3 different search queries for retrieving information about the following question. "
                        "Each query should use different phrasing and terminology. "
                        "Output exactly 3 queries, one per line, no numbering, no extra text.\n\nOriginal: "
                        + query
                    ),
                }
            ],
            temperature=0.3,
            max_tokens=150,
        )
        lines = [
            line.strip()
            for line in resp.choices[0].message.content.strip().split("\n")
            if line.strip()
        ]
        variants = lines[:3] if len(lines) >= 3 else [query]
        if query not in variants:
            variants = [query] + variants[:2]
        _rewrite_cache[cache_key] = variants
        return variants
    except Exception:
        return [query]


# ─────────────────────────────────────────────────────────────────────────────
# FOLLOW-UP DETECTION
# ─────────────────────────────────────────────────────────────────────────────

_prev_query_embed: dict[str, np.ndarray] = {}


def compute_followup_signals(session_id: str, query: str) -> tuple[int, int]:
    follow_up_asked = 0
    similar_followup = 0
    sess = _get_session(session_id)
    query_count = sess.get("total_queries", 0)
    cur_emb = embed([query])[0]

    if session_id in _prev_query_embed and query_count > 0:
        cos_sim = float(np.dot(cur_emb, _prev_query_embed[session_id]))
        if cos_sim > 0.82:
            similar_followup = 1

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
# GROQ CLIENT
# ─────────────────────────────────────────────────────────────────────────────


def _get_groq_client() -> Groq:
    # [P4] Always read at call-time so Streamlit st.secrets injection works correctly
    key = os.getenv("GROQ_API_KEY", "") or GROQ_API_KEY
    return Groq(api_key=key)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

_GROUNDING = (
    "CRITICAL RULES:\n"
    "1. Answer ONLY using information from the CONTEXT below.\n"
    "2. If the context contains the answer, quote or paraphrase it directly.\n"
    "3. If a specific number, metric, or name appears in the context, reproduce it exactly.\n"
    "4. If the context does NOT contain enough information, say: "
    "'The provided context does not contain enough information to answer this question.'\n"
    "5. Never make up citations, numbers, or facts not present in the context.\n"
)

PROMPT_REGISTRY: dict[str, dict] = {
    "v1_standard": {
        "version": "v1_standard",
        "label": "Standard",
        "system": "You are a precise AI assistant. " + _GROUNDING,
        "template": "CONTEXT:\n{context}\n\n{memory}QUESTION: {query}\n\nAnswer directly using only the context above.",
    },
    "v2_simplified": {
        "version": "v2_simplified",
        "label": "Simplified",
        "system": "You are a patient AI tutor who explains clearly. " + _GROUNDING,
        "template": "CONTEXT:\n{context}\n\n{memory}QUESTION: {query}\n\nExplain clearly using the context. Use simple language and bullet points.",  # noqa: E501
    },
    "v3_detailed": {
        "version": "v3_detailed",
        "label": "Detailed Structured",
        "system": "You are an expert instructor who gives structured answers. "
        + _GROUNDING,
        "template": "CONTEXT:\n{context}\n\n{memory}QUESTION: {query}\n\nProvide a structured answer: (1) Direct answer, (2) How it works, (3) Key details from context, (4) Takeaway.",  # noqa: E501
    },
    "v4_socratic": {
        "version": "v4_socratic",
        "label": "Socratic",
        "system": "You are a Socratic tutor. " + _GROUNDING,
        "template": "CONTEXT:\n{context}\n\n{memory}QUESTION: {query}\n\nAnswer precisely using the context, then ask one follow-up question to check understanding.",  # noqa: E501
    },
}


def get_active_prompt(session_id: str) -> dict:
    sess = _get_session(session_id)
    return PROMPT_REGISTRY.get(sess["prompt_version"], PROMPT_REGISTRY["v1_standard"])


def set_active_prompt(session_id: str, version: str) -> None:
    if version in PROMPT_REGISTRY:
        _get_session(session_id)["prompt_version"] = version


# ─────────────────────────────────────────────────────────────────────────────
# A/B TESTING
# ─────────────────────────────────────────────────────────────────────────────

AB_GROUPS = {
    "control": "v1_standard",
    "treatment_a": "v2_simplified",
    "treatment_b": "v3_detailed",
}


def assign_ab_group(session_id: str) -> str:
    h = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
    return list(AB_GROUPS.keys())[h % len(AB_GROUPS)]


def get_ab_prompt_version(session_id: str) -> str:
    return AB_GROUPS[assign_ab_group(session_id)]


# ─────────────────────────────────────────────────────────────────────────────
# RAG PIPELINE
# ─────────────────────────────────────────────────────────────────────────────


def rag_query(
    query: str,
    session_id: str,
    use_ab: bool = True,
    on_step: Optional[Callable[[int], None]] = None,
) -> dict:
    t0 = time.time()
    sess = _get_session(session_id)

    # Step 0: topic + rewrite
    topic = detect_topic(query)
    rw = rewrite_query(query)
    if on_step:
        on_step(0)

    # Step 1: hybrid retrieval
    docs = hybrid_retrieve(rw, top_k=TOP_K, topic_filter=topic)
    if len(docs) < 4:
        broad = hybrid_retrieve(rw, top_k=TOP_K, topic_filter=None)
        seen = {d["doc_id"] for d in docs}
        for d in broad:
            if d["doc_id"] not in seen:
                docs.append(d)
                seen.add(d["doc_id"])
        docs = docs[:TOP_K]
    if on_step:
        on_step(1)

    # [F1] Filter low-quality docs
    score_override = sess.get("min_score_override")
    docs = filter_by_score(docs, RETRIEVAL_SCORE_THRESHOLD, score_override)

    # [F2] Cross-encoder re-ranking
    docs = rerank_docs(query, docs, top_k=6)
    if on_step:
        on_step(2)

    doc_ids = [d["doc_id"] for d in docs]
    scores = [d.get("dense_score", d["score"]) for d in docs]
    avg_score = round(float(np.mean(scores)), 4) if scores else 0.0
    context_parts = [f"[Source {i + 1}]\n{d['text']}" for i, d in enumerate(docs)]
    context = "\n\n---\n\n".join(context_parts)

    # Step 3: memory
    mem_str = (
        (get_memory_context(session_id) + "\n\n")
        if get_memory_context(session_id)
        else ""
    )
    if on_step:
        on_step(3)

    # Prompt selection
    if use_ab:
        ab_group = assign_ab_group(session_id)
        sess_prompt = sess["prompt_version"]
        prompt_cfg = (
            PROMPT_REGISTRY[sess_prompt]
            if sess_prompt != "v1_standard"
            else PROMPT_REGISTRY[AB_GROUPS[ab_group]]
        )
    else:
        ab_group = "feedback_loop"
        prompt_cfg = get_active_prompt(session_id)

    # Step 4: generate
    try:
        user_msg = prompt_cfg["template"].format(
            context=context, query=query, memory=mem_str
        )
        chat = _get_groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": prompt_cfg["system"]},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.15,
            max_tokens=1200,
        )
        response_text = chat.choices[0].message.content.strip()
    except Exception as e:
        # [F15] Graceful degradation
        response_text = f"[System error: Unable to generate response. Please try again. Detail: {str(e)[:120]}]"

    add_to_memory(session_id, "user", query)
    add_to_memory(session_id, "assistant", response_text)
    if on_step:
        on_step(4)

    latency = round(time.time() - t0, 2)
    _latency_log.append(latency)  # [F11]

    return {
        "response": response_text,
        "retrieved_doc_ids": doc_ids,
        "retrieval_scores": scores,
        "avg_retrieval_score": avg_score,
        "response_length": len(response_text),
        "latency_sec": latency,
        "prompt_version": prompt_cfg["version"],
        "ab_group": ab_group,
        "topic_detected": topic or "general",
        "rewritten_query": rw,
        "context": context,
        "hallucination_risk": hallucination_risk(response_text, docs),
        "response_diversity": response_diversity_score(response_text),
        "docs": docs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM-AS-JUDGE [F6] — runs every query
# ─────────────────────────────────────────────────────────────────────────────


def llm_judge(query: str, context: str, response: str) -> Optional[dict]:
    """[F6] Returns None on failure. Now called every query."""
    prompt = (
        "You are an objective AI evaluator. Return ONLY valid JSON, no markdown.\n\n"
        f"Question: {query}\nContext: {context[:400]}\nResponse: {response[:600]}\n\n"
        "Score 1-5 for each:\n  relevance, clarity, completeness, groundedness, conciseness\n\n"
        'Return exactly: {"relevance":X,"clarity":X,"completeness":X,"groundedness":X,"conciseness":X,"overall":X}'
    )
    try:
        resp = _get_groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=120,
        )
        raw = (
            resp.choices[0]
            .message.content.strip()
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
        parsed = json.loads(raw)
        return {
            k: max(1, min(5, int(round(parsed.get(k, 3)))))
            for k in [
                "relevance",
                "clarity",
                "completeness",
                "groundedness",
                "conciseness",
                "overall",
            ]
        }
    except Exception:
        return None


def compute_judge_ml_correlation(
    df_judge: pd.DataFrame, df_interactions: pd.DataFrame
) -> Optional[float]:
    """[F7] Pearson correlation between judge overall score and ML understanding probability."""
    try:
        merged = pd.merge(
            df_judge[["timestamp", "overall"]],
            df_interactions[["timestamp", "prediction"]],
            on="timestamp",
            how="inner",
        )
        if len(merged) < 5:
            return None
        return round(float(merged["overall"].corr(merged["prediction"])), 4)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA
# ─────────────────────────────────────────────────────────────────────────────


def generate_synthetic_data(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    def clip(v, lo=0.01, hi=0.99):
        return float(np.clip(v, lo, hi))

    def sample(understood: int, n_s: int) -> list[dict]:
        rows = []
        for _ in range(n_s):
            if understood:
                ttr = max(0.5, rng.normal(45, 22))
                fu = int(rng.random() < 0.20)
                sfu = 0
                sqc = int(np.clip(rng.normal(2.5, 1.5), 1, 12))
                rlen = int(np.clip(rng.normal(520, 140), 80, 900))
                ars = clip(rng.normal(0.72, 0.12))
                qc = clip(rng.normal(0.32, 0.15), 0.05, 0.95)
                hr = clip(rng.normal(0.22, 0.12), 0.0, 0.95)
                rd = clip(rng.normal(0.70, 0.10))
            else:
                ttr = max(0.5, rng.normal(10, 7))
                fu = int(rng.random() < 0.65)
                sfu = int(fu and rng.random() < 0.70)
                sqc = int(np.clip(rng.normal(6, 3), 1, 20))
                rlen = int(np.clip(rng.normal(180, 100), 30, 600))
                ars = clip(rng.normal(0.45, 0.14))
                qc = clip(rng.normal(0.65, 0.18), 0.05, 0.99)
                hr = clip(rng.normal(0.58, 0.18), 0.0, 0.99)
                rd = clip(rng.normal(0.48, 0.11))
            rows.append(
                {
                    "time_to_read": round(ttr, 2),
                    "follow_up_asked": fu,
                    "similar_followup": sfu,
                    "session_query_count": sqc,
                    "response_length": rlen,
                    "avg_retrieval_score": round(ars, 4),
                    "query_complexity": round(qc, 4),
                    "hallucination_risk": round(hr, 4),
                    "response_diversity": round(rd, 4),
                    # [F3] Interaction features
                    "quality_weighted_length": round(rlen * ars, 2),
                    "engagement_depth": round(sqc * fu, 2),
                    "risk_under_load": round(hr * qc, 4),
                    "understood": understood,
                }
            )
        return rows

    rows = sample(1, n // 2) + sample(0, n // 2)
    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df.to_csv("synthetic_data.csv", index=False)
    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """[F3] Safely add interaction features to any dataframe."""
    df = df.copy()
    if "quality_weighted_length" not in df.columns:
        df["quality_weighted_length"] = (
            df["response_length"] * df["avg_retrieval_score"]
        )
    if "engagement_depth" not in df.columns:
        df["engagement_depth"] = df["session_query_count"] * df["follow_up_asked"]
    if "risk_under_load" not in df.columns:
        df["risk_under_load"] = df["hallucination_risk"] * df["query_complexity"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ML MODEL
# ─────────────────────────────────────────────────────────────────────────────


def _build_candidates() -> dict:
    candidates = {
        "Logistic Regression": Pipeline(
            [
                ("sc", StandardScaler()),
                (
                    "m",
                    LogisticRegression(
                        max_iter=600,
                        C=1.0,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("sc", StandardScaler()),
                (
                    "m",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=8,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("sc", StandardScaler()),
                (
                    "m",
                    GradientBoostingClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=4,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }
    try:
        from xgboost import XGBClassifier

        candidates["XGBoost"] = Pipeline(
            [
                ("sc", StandardScaler()),
                (
                    "m",
                    XGBClassifier(
                        n_estimators=100,
                        max_depth=4,
                        learning_rate=0.1,
                        nthread=-1,
                        eval_metric="auc",
                        random_state=42,
                        verbosity=0,
                    ),
                ),
            ]
        )
    except ImportError:
        pass
    return candidates


def _calibrate_threshold(model, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """[F4] Find optimal threshold via F1 maximization on validation set."""
    probs = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, probs)
    f1_scores = []
    for p, r in zip(precision, recall):
        denom = p + r
        f1_scores.append((2 * p * r / denom) if denom > 0 else 0)
    if not thresholds.size:
        return 0.5
    best_idx = int(np.argmax(f1_scores[:-1]))
    return round(float(thresholds[best_idx]), 4)


def train_model(df: Optional[pd.DataFrame] = None) -> dict:
    global _ml_model, _ml_metrics, _feature_importance, _opt_threshold

    if df is None or df.empty:
        existing = load_interactions()
        df = existing if len(existing) >= 40 else generate_synthetic_data(600)

    df = _add_interaction_features(df)

    for col_, default in [("hallucination_risk", 0.3), ("response_diversity", 0.6)]:
        if col_ not in df.columns:
            df = df.copy()
            df[col_] = default

    df = df[[f for f in FEATURES if f in df.columns] + ["understood"]].dropna()
    X, y = df[[f for f in FEATURES if f in df.columns]], df["understood"].astype(int)

    # 3-way split: train / val (for threshold) / test (for metrics)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15, random_state=42, stratify=y_tv
    )

    candidates = _build_candidates()
    cv_results: dict = {}
    best_name, best_pipe, best_auc = "", None, -1.0

    def _cv_one(name_pipe):
        name, pipe = name_pipe
        scores = cross_val_score(
            pipe, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1
        )
        return name, pipe, float(scores.mean()), float(scores.std())

    with ThreadPoolExecutor(max_workers=len(candidates)) as ex:
        futures = list(ex.map(_cv_one, candidates.items()))

    for name, pipe, mean_auc, std in futures:
        cv_results[name] = {"mean_auc": round(mean_auc, 4), "std": round(std, 4)}
        if mean_auc > best_auc:
            best_auc, best_name, best_pipe = mean_auc, name, pipe

    best_pipe.fit(X_train, y_train)

    # [F4] Calibrate threshold on validation set
    _opt_threshold = _calibrate_threshold(best_pipe, X_val, y_val)

    y_prob = best_pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= _opt_threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred).tolist()

    fi: dict = {}
    try:
        inner = best_pipe.named_steps["m"]
        if hasattr(inner, "feature_importances_"):
            feat_names = [f for f in FEATURES if f in X.columns]
            fi = dict(zip(feat_names, inner.feature_importances_.tolist()))
        elif hasattr(inner, "coef_"):
            feat_names = [f for f in FEATURES if f in X.columns]
            fi = dict(zip(feat_names, np.abs(inner.coef_[0]).tolist()))
    except Exception:
        pass
    _feature_importance = fi

    metrics = {
        "best_model": best_name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "opt_threshold": _opt_threshold,
        "confusion_matrix": cm,
        "cv_results": cv_results,
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "features": [f for f in FEATURES if f in X.columns],
        "trained_at": datetime.now().isoformat(),
        "data_source": "real" if len(load_interactions()) >= 40 else "synthetic",
    }
    _ml_model = best_pipe
    _ml_metrics = metrics
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "model": best_pipe,
                "metrics": metrics,
                "fi": fi,
                "threshold": _opt_threshold,
            },
            f,
        )

    # MLflow experiment tracking
    if _MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment("pulserag")
            with mlflow.start_run(
                run_name=f"{best_name}_{datetime.now().strftime('%H%M%S')}"
            ):
                mlflow.log_param("best_model", best_name)
                mlflow.log_param("train_size", metrics["train_size"])
                mlflow.log_param("data_source", metrics["data_source"])
                mlflow.log_param("n_features", len(metrics["features"]))
                mlflow.log_metric("roc_auc", metrics["roc_auc"])
                mlflow.log_metric("f1_score", metrics["f1_score"])
                mlflow.log_metric("accuracy", metrics["accuracy"])
                mlflow.log_metric("opt_threshold", metrics["opt_threshold"])
                for name, res in cv_results.items():
                    safe = name.replace(" ", "_")
                    mlflow.log_metric(f"cv_auc_{safe}", res["mean_auc"])
                    mlflow.log_metric(f"cv_std_{safe}", res["std"])
                if fi:
                    for feat, imp in fi.items():
                        mlflow.log_metric(f"fi_{feat}", round(float(imp), 6))
                mlflow.sklearn.log_model(best_pipe, "model")
        except Exception:
            pass  # MLflow logging is non-critical

    return metrics


def load_model() -> bool:
    global _ml_model, _ml_metrics, _feature_importance, _opt_threshold
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            _ml_model = data["model"]
            _ml_metrics = data["metrics"]
            _feature_importance = data.get("fi", {})
            _opt_threshold = data.get("threshold", 0.5)
            return True
        except Exception:
            pass
    return False


def build_feature_dict(raw: dict) -> dict:
    """[F3][F14] Pure function: computes all features including interaction features."""
    base = {f: raw.get(f, 0.0) for f in FEATURES[:9]}  # base 9
    base["quality_weighted_length"] = (
        base["response_length"] * base["avg_retrieval_score"]
    )
    base["engagement_depth"] = base["session_query_count"] * base["follow_up_asked"]
    base["risk_under_load"] = base["hallucination_risk"] * base["query_complexity"]
    return base


def predict_understanding(features: dict) -> dict:

    if _ml_model is None:
        if not load_model():
            train_model()
    full_features = build_feature_dict(features)
    # Use the feature list the model was actually trained on
    trained_features = _ml_metrics.get("features", FEATURES)
    x = pd.DataFrame([{f: full_features.get(f, 0.0) for f in trained_features}])
    try:
        prob = float(_ml_model.predict_proba(x)[0][1])
    except ValueError:
        # Stale .pkl trained on a different feature set — delete and retrain
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        train_model()
        trained_features = _ml_metrics.get("features", FEATURES)
        x = pd.DataFrame([{f: full_features.get(f, 0.0) for f in trained_features}])
        prob = float(_ml_model.predict_proba(x)[0][1])
    return {
        "understood": int(prob >= _opt_threshold),
        "probability": round(prob, 4),
        "confidence": round(max(prob, 1 - prob), 4),
        "threshold": _opt_threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ONLINE LEARNING — SGDClassifier with partial_fit
# ─────────────────────────────────────────────────────────────────────────────


def _get_online_model() -> SGDClassifier:
    """Lazy-init the online SGDClassifier. Always returns same instance."""
    global _online_model
    if _online_model is None:
        _online_model = SGDClassifier(
            loss="log_loss",  # probabilistic output
            penalty="l2",
            alpha=0.0001,
            learning_rate="optimal",
            class_weight="balanced",
            warm_start=True,  # enables partial_fit semantics on fit()
            random_state=42,
            max_iter=1,  # one pass per partial_fit call
        )
    return _online_model


def online_update(features: dict, label: int) -> dict:
    """
    Online learning: update SGDClassifier with a single (features, label) sample.
    Buffers samples and calls partial_fit every _ONLINE_BUFFER_SIZE samples
    to amortize overhead. Returns current online model metrics.
    """
    global _online_buffer, _online_query_count, _online_metrics

    full = build_feature_dict(features)
    feat_names = [f for f in FEATURES if f in full]
    x_row = [full.get(f, 0.0) for f in feat_names]

    _online_buffer.append((x_row, label))
    _online_query_count += 1

    model = _get_online_model()

    # Always do a single-sample partial_fit immediately (true online learning)
    try:
        model.partial_fit(
            [x_row],
            [label],
            classes=[0, 1],
        )
    except Exception:
        pass

    # Every _ONLINE_BUFFER_SIZE samples, flush buffer and compute metrics
    if len(_online_buffer) >= _ONLINE_BUFFER_SIZE:
        X_buf = [row for row, _ in _online_buffer]
        y_buf = [lbl for _, lbl in _online_buffer]
        try:
            model.partial_fit(X_buf, y_buf, classes=[0, 1])
            probs = model.predict_proba(X_buf)[:, 1]
            preds = (probs >= 0.5).astype(int)
            _online_metrics = {
                "accuracy": round(float(accuracy_score(y_buf, preds)), 4),
                "f1_score": round(float(f1_score(y_buf, preds, zero_division=0)), 4),
                "roc_auc": (
                    round(float(roc_auc_score(y_buf, probs)), 4)
                    if len(set(y_buf)) > 1
                    else 0.0
                ),
                "samples_seen": _online_query_count,
                "buffer_size": len(_online_buffer),
                "last_updated": datetime.now().isoformat(),
            }
        except Exception:
            pass
        _online_buffer = []  # reset buffer after flush

    return _online_metrics


def online_predict(features: dict) -> Optional[dict]:
    """
    Get prediction from the online model if it has seen enough samples.
    Returns None if the online model is not yet warmed up (< 10 samples).
    """

    if _online_query_count < 10:
        return None
    model = _get_online_model()
    try:
        full = build_feature_dict(features)
        feat_names = [f for f in FEATURES if f in full]
        x = [full.get(f, 0.0) for f in feat_names]
        prob = float(model.predict_proba([x])[0][1])
        return {
            "online_probability": round(prob, 4),
            "online_understood": int(prob >= 0.5),
            "samples_seen": _online_query_count,
        }
    except Exception:
        return None


def get_online_metrics() -> dict:
    """Return current online model metrics for display."""
    return {
        **_online_metrics,
        "samples_seen": _online_query_count,
        "buffer_pending": len(_online_buffer),
        "model_ready": _online_query_count >= 10,
    }


def save_online_model() -> None:
    """Persist online model to disk alongside batch model."""
    if _online_model is not None:
        try:
            with open("pulserag_online_model.pkl", "wb") as f:
                pickle.dump(
                    {
                        "model": _online_model,
                        "metrics": _online_metrics,
                        "query_count": _online_query_count,
                        "buffer": _online_buffer,
                    },
                    f,
                )
        except Exception:
            pass


def load_online_model() -> bool:
    """Load persisted online model from disk."""
    global _online_model, _online_metrics, _online_query_count, _online_buffer
    path = "pulserag_online_model.pkl"
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        _online_model = data["model"]
        _online_metrics = data.get("metrics", {})
        _online_query_count = data.get("query_count", 0)
        _online_buffer = data.get("buffer", [])
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# LATENCY MONITORING [F11]
# ─────────────────────────────────────────────────────────────────────────────


def get_latency_percentiles() -> dict:
    if not _latency_log:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "count": 0}
    arr = np.array(_latency_log)
    return {
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3),
        "mean": round(float(arr.mean()), 3),
        "count": len(arr),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DRIFT DETECTION [F12]
# ─────────────────────────────────────────────────────────────────────────────


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    eps = 1e-6
    mn = min(expected.min(), actual.min())
    mx = max(expected.max(), actual.max())
    bin_edges = np.linspace(mn, mx, bins + 1)
    exp_hist = np.histogram(expected, bins=bin_edges)[0].astype(float) + eps
    act_hist = np.histogram(actual, bins=bin_edges)[0].astype(float) + eps
    exp_pct = exp_hist / exp_hist.sum()
    act_pct = act_hist / act_hist.sum()
    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(psi, 4)


def check_drift_and_retrain() -> dict:
    """[F12] Automatically detect drift via PSI and trigger retrain if needed."""
    df = load_interactions()
    if len(df) < 60:
        return {"drift_detected": False, "reason": "insufficient data", "psi": None}

    mid = len(df) // 2
    ref = df.iloc[:mid]
    curr = df.iloc[mid:]
    key_feat = "avg_retrieval_score"

    if key_feat not in df.columns:
        return {"drift_detected": False, "reason": "feature missing", "psi": None}

    psi = compute_psi(ref[key_feat].values, curr[key_feat].values)
    if psi > _PSI_THRESHOLD:
        train_model(df)
        return {
            "drift_detected": True,
            "psi": psi,
            "action": "retrained",
            "threshold": _PSI_THRESHOLD,
        }
    return {"drift_detected": False, "psi": psi, "threshold": _PSI_THRESHOLD}


# ─────────────────────────────────────────────────────────────────────────────
# FEEDBACK LOOP [F8][F9][F10]
# ─────────────────────────────────────────────────────────────────────────────


def apply_feedback(
    prediction: dict,
    retrieval_scores: list,
    session_id: str,
    h_risk: float,
    retrieval_docs: Optional[list[dict]] = None,
) -> dict:
    sess = _get_session(session_id)
    sess["total_queries"] += 1

    # [F10] Resolve pending metric_after
    if session_id in _pending_feedback:
        pending = _pending_feedback.pop(session_id)
        _update_feedback_metric_after(pending["log_id"], prediction["probability"])

    old_prompt = sess["prompt_version"]
    action = reason = ""
    understood = prediction["understood"]
    sess["understood"].append(understood)

    # Track streaks in both directions
    if understood:
        sess["streak"] = max(0, sess["streak"] - 1)
        sess["streak_good"] = sess.get("streak_good", 0) + 1
    else:
        sess["streak"] = sess["streak"] + 1
        sess["streak_good"] = 0

    # Rule 1: confusion streak → escalate
    if sess["streak"] >= _STREAK_LIMIT:
        try:
            cur_idx = PROMPT_SEQUENCE.index(sess["prompt_version"])
        except ValueError:
            cur_idx = 0
        nxt = PROMPT_SEQUENCE[min(cur_idx + 1, len(PROMPT_SEQUENCE) - 1)]
        if nxt != sess["prompt_version"]:
            set_active_prompt(session_id, nxt)
            action = f"prompt_escalate → {nxt}"
            reason = f"{sess['streak']} consecutive not-understood"
        sess["streak"] = 0

    # [F9] De-escalation: if 3 consecutive understood, step back toward v1
    if not action and sess.get("streak_good", 0) >= _STREAK_GOOD:
        try:
            cur_idx = PROMPT_SEQUENCE.index(sess["prompt_version"])
        except ValueError:
            cur_idx = 0
        if cur_idx > 0:
            prev = PROMPT_SEQUENCE[cur_idx - 1]
            set_active_prompt(session_id, prev)
            action = f"prompt_deescalate → {prev}"
            reason = f"{sess['streak_good']} consecutive understood"
        sess["streak_good"] = 0

    avg_score = float(np.mean(retrieval_scores)) if retrieval_scores else 0.5

    # Rule 2: [F8] Low retrieval → actively lower score threshold to broaden search
    if avg_score < 0.50 and not understood and not action:
        sess["min_score_override"] = max(0.15, RETRIEVAL_SCORE_THRESHOLD - 0.10)
        action = "retrieval_threshold_lowered"
        reason = f"low avg retrieval ({avg_score:.3f}) — threshold lowered to {sess['min_score_override']:.2f}"

    # Rule 3: [F8] High hallucination → force detailed structured prompt
    if h_risk > 0.65 and not action:
        if sess["prompt_version"] != "v3_detailed":
            set_active_prompt(session_id, "v3_detailed")
            action = "hallucination_guard → v3_detailed"
            reason = f"high hallucination risk ({h_risk:.3f}) — switched to structured grounded prompt"
        else:
            action = "hallucination_risk_flag"
            reason = f"high hal risk ({h_risk:.3f}) — already on v3"

    # Rule 4: last 5 all not-understood → socratic
    sess_hist = sess["understood"]
    if len(sess_hist) >= 5 and sum(sess_hist[-5:]) == 0 and not action:
        set_active_prompt(session_id, "v4_socratic")
        action = "session_reset → v4_socratic"
        reason = "last 5 all not-understood"

    # Rule 5: [F8] Retrieval diversity — broaden query by removing topic filter
    if retrieval_docs and not action:
        topics = [d.get("topic", "general") for d in retrieval_docs]
        if len(set(topics)) == 1 and not understood:
            sess["min_score_override"] = max(0.10, RETRIEVAL_SCORE_THRESHOLD - 0.15)
            action = "retrieval_diversity_broadened"
            reason = f"all docs from single topic '{topics[0]}' — broadening retrieval"

    # Rule 6: periodic retrain
    retrain = False
    total_q = sum(s.get("total_queries", 0) for s in _session_state.values())
    if total_q % _RETRAIN_EVERY == 0:
        df = load_interactions()
        if len(df) >= 30:
            train_model(df)
            retrain = True
            if not action:
                action = "ml_retrain"
                reason = f"periodic retrain after {total_q} queries"

    log_id = _insert_returning_id(
        "feedback_log",
        {
            "timestamp": datetime.now().isoformat(),
            "action": action or "none",
            "reason": reason,
            "old_prompt": old_prompt,
            "new_prompt": sess["prompt_version"],
            "metric_before": prediction["probability"],
            "metric_after": None,
            "session_id": session_id,
        },
    )
    if action:
        _pending_feedback[session_id] = {
            "log_id": log_id,
            "metric_before": prediction["probability"],
        }

    return {
        "action": action or "none",
        "reason": reason,
        "new_prompt": sess["prompt_version"],
        "streak": sess["streak"],
        "streak_good": sess.get("streak_good", 0),
        "retrained": retrain,
    }


def _update_feedback_metric_after(log_id: int, metric_after: float) -> None:
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "UPDATE feedback_log SET metric_after=? WHERE id=?", (metric_after, log_id)
        )
        con.commit()
        con.close()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# SQLITE
# ─────────────────────────────────────────────────────────────────────────────


def init_db() -> None:
    con = sqlite3.connect(DB_PATH)
    # Auto-migrate: add missing columns to existing tables without dropping data
    try:
        existing = [
            r[1] for r in con.execute("PRAGMA table_info(judge_log)").fetchall()
        ]
        for col in ["context_precision", "context_recall", "faithfulness"]:
            if col not in existing:
                con.execute(f"ALTER TABLE judge_log ADD COLUMN {col} REAL")
        existing_i = [
            r[1] for r in con.execute("PRAGMA table_info(interactions)").fetchall()
        ]
        for col in [
            "quality_weighted_length",
            "engagement_depth",
            "risk_under_load",
            "context_precision",
            "context_recall",
            "faithfulness",
            "opt_threshold",
        ]:
            if col not in existing_i:
                con.execute(f"ALTER TABLE interactions ADD COLUMN {col} REAL")
        con.commit()
    except Exception:
        pass
    con.executescript("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, query TEXT, rewritten_query TEXT, response TEXT,
            retrieved_docs TEXT, retrieval_scores TEXT, topic_detected TEXT,
            time_to_read REAL, follow_up_asked INTEGER, similar_followup INTEGER,
            session_query_count INTEGER, response_length INTEGER,
            avg_retrieval_score REAL, query_complexity REAL,
            hallucination_risk REAL, response_diversity REAL,
            quality_weighted_length REAL, engagement_depth REAL, risk_under_load REAL,
            context_precision REAL, context_recall REAL, faithfulness REAL,
            understood INTEGER, prediction REAL, confidence REAL,
            opt_threshold REAL,
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
            groundedness REAL, conciseness REAL, overall REAL,
            context_precision REAL, context_recall REAL, faithfulness REAL
        );
    """)
    con.commit()
    con.close()


def _insert(table: str, row: dict) -> None:
    con = sqlite3.connect(DB_PATH)
    cols = ", ".join(row.keys())
    marks = ", ".join("?" for _ in row)
    con.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values()))
    con.commit()
    con.close()


def _insert_returning_id(table: str, row: dict) -> int:
    con = sqlite3.connect(DB_PATH)
    cols = ", ".join(row.keys())
    marks = ", ".join("?" for _ in row)
    cur = con.execute(
        f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values())
    )
    row_id = cur.lastrowid
    con.commit()
    con.close()
    return row_id


def save_interaction(row):
    _insert("interactions", row)


def save_ab_record(row):
    _insert("ab_tests", row)


def save_judge_log(row: Optional[dict]) -> None:
    if row is not None:
        _insert("judge_log", row)


def _load_table(table: str) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {table}", con)
    con.close()
    return df


def load_interactions() -> pd.DataFrame:
    return _load_table("interactions")


def load_feedback_log() -> pd.DataFrame:
    return _load_table("feedback_log")


def load_ab_results() -> pd.DataFrame:
    return _load_table("ab_tests")


def load_judge_log() -> pd.DataFrame:
    return _load_table("judge_log")


def get_ab_summary() -> pd.DataFrame:
    df = load_ab_results()
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby("group_name")
        .agg(
            n=("understood", "count"),
            understood_rate=("understood", "mean"),
            avg_prediction=("prediction", "mean"),
            avg_retrieval=("retrieval_score", "mean"),
        )
        .round(3)
        .reset_index()
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────────────────────────────────────


def smart_chunk(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
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
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= max_chars:
                        current = (current + " " + sent).strip() if current else sent
                    else:
                        if current:
                            chunks.append(current)
                        current = (
                            (current[-overlap:] + " " + sent).strip()
                            if current
                            else sent
                        )
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# MONITORING [F13]
# ─────────────────────────────────────────────────────────────────────────────


def generate_monitoring_report() -> str:
    """[F13] Never uses synthetic data fallback — real data only."""
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError:
        return "error: pip install evidently"
    df = load_interactions()
    if len(df) < 40:
        return "error: need at least 40 real interactions to generate a meaningful drift report — keep chatting and try again"  # noqa: E501
    needed = [
        f
        for f in [
            "time_to_read",
            "avg_retrieval_score",
            "hallucination_risk",
            "response_diversity",
            "query_complexity",
        ]
        if f in df.columns
    ] + ["understood"]
    df = df[needed].dropna()
    if len(df) < 20:
        return "error: not enough complete rows after filtering"
    mid = len(df) // 2
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df.iloc[:mid].copy(), current_data=df.iloc[mid:].copy())
    path = "monitoring_report.html"
    report.save_html(path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────


def get_system_stats() -> dict:
    global _stats_cache, _stats_cache_ts
    now = time.time()
    if _stats_cache and (now - _stats_cache_ts) < _STATS_TTL:
        return _stats_cache

    df = load_interactions()
    fb = load_feedback_log()
    jdf = load_judge_log()
    total_q = sum(s.get("total_queries", 0) for s in _session_state.values())
    versions = [s.get("prompt_version", "v1_standard") for s in _session_state.values()]
    active = (
        max(versions, key=lambda v: PROMPT_SEQUENCE.index(v))
        if versions
        else "v1_standard"
    )

    # [F7] Judge vs ML correlation
    judge_ml_corr = None
    if not jdf.empty and not df.empty:
        judge_ml_corr = compute_judge_ml_correlation(jdf, df)

    # [F11] Latency percentiles
    latency_pct = get_latency_percentiles()

    _stats_cache = {
        "total_interactions": int(len(df)),
        "understood_rate": (
            round(float(df["understood"].mean()), 3) if not df.empty else 0.0
        ),
        "avg_retrieval_score": (
            round(float(df["avg_retrieval_score"].mean()), 3) if not df.empty else 0.0
        ),
        "avg_hallucination_risk": (
            round(float(df["hallucination_risk"].mean()), 3)
            if not df.empty and "hallucination_risk" in df.columns
            else 0.0
        ),
        "avg_diversity": (
            round(float(df["response_diversity"].mean()), 3)
            if not df.empty and "response_diversity" in df.columns
            else 0.0
        ),
        "active_prompt": active,
        "feedback_actions": int(len(fb)),
        "streak": max((s.get("streak", 0) for s in _session_state.values()), default=0),
        "total_queries": total_q,
        "model_metrics": _ml_metrics,
        "feature_importance": _feature_importance,
        "opt_threshold": _opt_threshold,
        "corpus_size": len(KNOWLEDGE_DOCS),
        "version": VERSION,
        "judge_ml_correlation": judge_ml_corr,
        "latency": latency_pct,
        "data_source": _ml_metrics.get("data_source", "synthetic"),
        "online_model": get_online_metrics(),
        # RAGAS averages
        "avg_context_precision": (
            round(float(df["context_precision"].mean()), 3)
            if not df.empty and "context_precision" in df.columns
            else None
        ),
        "avg_context_recall": (
            round(float(df["context_recall"].mean()), 3)
            if not df.empty and "context_recall" in df.columns
            else None
        ),
        "avg_faithfulness": (
            round(float(df["faithfulness"].mean()), 3)
            if not df.empty and "faithfulness" in df.columns
            else None
        ),
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
    rag = rag_query(query, session_id, on_step=on_step)

    base_features = {
        "time_to_read": time_to_read,
        "follow_up_asked": follow_up_asked,
        "similar_followup": similar_followup,
        "session_query_count": session_query_count,
        "response_length": rag["response_length"],
        "avg_retrieval_score": rag["avg_retrieval_score"],
        "query_complexity": query_complexity,
        "hallucination_risk": rag["hallucination_risk"],
        "response_diversity": rag["response_diversity"],
    }
    # [F3] Add interaction features
    features = build_feature_dict(base_features)
    prediction = predict_understanding(features)
    if on_step:
        on_step(5)

    feedback = apply_feedback(
        prediction,
        rag["retrieval_scores"],
        session_id,
        rag["hallucination_risk"],
        retrieval_docs=rag["docs"],
    )
    if on_step:
        on_step(6)

    # [F5] RAGAS metrics
    ragas = compute_ragas_metrics(query, rag["response"], rag["docs"])

    # [F6] LLM Judge every query
    judge_scores: Optional[dict] = llm_judge(query, rag["context"], rag["response"])

    # [F7] Save judge log with RAGAS fields
    if judge_scores is not None:
        save_judge_log(
            {
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "query": query,
                "relevance": judge_scores["relevance"],
                "clarity": judge_scores["clarity"],
                "completeness": judge_scores["completeness"],
                "groundedness": judge_scores["groundedness"],
                "conciseness": judge_scores["conciseness"],
                "overall": judge_scores["overall"],
                "context_precision": ragas["context_precision"],
                "context_recall": ragas["context_recall"],
                "faithfulness": ragas["faithfulness"],
            }
        )

    save_ab_record(
        {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "group_name": rag["ab_group"],
            "prompt_version": rag["prompt_version"],
            "understood": prediction["understood"],
            "prediction": prediction["probability"],
            "retrieval_score": rag["avg_retrieval_score"],
        }
    )

    save_interaction(
        {
            "session_id": session_id,
            "query": query,
            "rewritten_query": rag["rewritten_query"],
            "response": rag["response"],
            "retrieved_docs": json.dumps(rag["retrieved_doc_ids"]),
            "retrieval_scores": json.dumps(rag["retrieval_scores"]),
            "topic_detected": rag["topic_detected"],
            "time_to_read": time_to_read,
            "follow_up_asked": follow_up_asked,
            "similar_followup": similar_followup,
            "session_query_count": session_query_count,
            "response_length": rag["response_length"],
            "avg_retrieval_score": rag["avg_retrieval_score"],
            "query_complexity": query_complexity,
            "hallucination_risk": rag["hallucination_risk"],
            "response_diversity": rag["response_diversity"],
            "quality_weighted_length": features["quality_weighted_length"],
            "engagement_depth": features["engagement_depth"],
            "risk_under_load": features["risk_under_load"],
            "context_precision": ragas["context_precision"],
            "context_recall": ragas["context_recall"],
            "faithfulness": ragas["faithfulness"],
            "understood": prediction["understood"],
            "prediction": prediction["probability"],
            "confidence": prediction["confidence"],
            "opt_threshold": prediction["threshold"],
            "prompt_version": rag["prompt_version"],
            "ab_group": rag["ab_group"],
            "timestamp": datetime.now().isoformat(),
        }
    )

    # [F12] Check drift periodically
    total_q = sum(s.get("total_queries", 0) for s in _session_state.values())
    if total_q % 20 == 0:
        check_drift_and_retrain()

    # Online learning: update SGD model with this query's label
    online_update(features, prediction["understood"])
    online_pred = online_predict(features)
    # Persist online model every 10 queries
    if total_q % 10 == 0:
        save_online_model()

    global _stats_cache_ts
    _stats_cache_ts = 0.0

    return {
        "response": rag["response"],
        "rewritten_query": rag["rewritten_query"],
        "topic_detected": rag["topic_detected"],
        "retrieved_doc_ids": rag["retrieved_doc_ids"],
        "retrieval_scores": rag["retrieval_scores"],
        "avg_retrieval_score": rag["avg_retrieval_score"],
        "response_length": rag["response_length"],
        "hallucination_risk": rag["hallucination_risk"],
        "response_diversity": rag["response_diversity"],
        "latency_sec": rag["latency_sec"],
        "prompt_version": rag["prompt_version"],
        "ab_group": rag["ab_group"],
        "prediction": prediction,
        "online_prediction": online_pred,
        "feedback": feedback,
        "judge_scores": judge_scores or {},
        "ragas": ragas,
        "features_used": features,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────


def bootstrap() -> None:
    # [P2] Sequential bootstrap — parallel loading caused OOM on Streamlit Cloud
    # because embedding model + ChromaDB HNSW index + ML training all competed
    # for RAM simultaneously, crashing the health check before the app started.
    _recalc_avgdl()

    # Step 1: DB (fast, no RAM cost)
    init_db()

    # Step 2: Embedding model + ChromaDB index (biggest RAM spike — do alone)
    get_embed_model()
    setup_collection()

    # Step 3: ML model (train only if no saved model exists)
    if not load_model():
        # [P3] 300 samples at bootstrap (was 600) — sufficient for a good initial model
        # and saves ~2s + RAM during cold start. CI uses full 600 via generate_synthetic_data(600).
        df = generate_synthetic_data(300)
        train_model(df)

    # Step 4: Restore online model if it was persisted
    load_online_model()


if __name__ == "__main__":
    bootstrap()
    print(f"PulseRAG v{VERSION} backend ready.")
    print(json.dumps(get_system_stats(), indent=2, default=str))
