"""
=============================================================================
Indecimal AI Systems Challenge — backend.py
=============================================================================
All logic lives here:
  • Knowledge base (20 ML/AI documents)
  • Qdrant vector store  (in-memory fallback if no cloud URL given)
  • HuggingFace sentence-transformer embeddings
  • Groq LLM  (llama3-70b-8192)
  • RAG pipeline
  • LLM-as-Judge evaluation
  • Synthetic data generation
  • ML understanding predictor  (LR / RF / GBM — best picked by CV AUC)
  • Prompt registry + feedback loop
  • SQLite interaction & feedback logging
  • Evidently monitoring report
  • Unified  run_pipeline()  entry-point called by app.py
=============================================================================
"""

from __future__ import annotations

import json
import os
import pickle
import random
import sqlite3
import time
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME CONFIG  (overridden by app.py after reading st.secrets)
# ─────────────────────────────────────────────────────────────────────────────

GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
QDRANT_URL      = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", "")

COLLECTION_NAME  = "indecimal_knowledge"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL       = "llama3-70b-8192"
DB_PATH          = "interactions.db"
MODEL_PATH       = "understanding_model.pkl"
TOP_K            = 5

# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE  — 20 ML / AI documents used as the RAG corpus
# ─────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_DOCS: list[dict] = [
    {
        "id": "ml_001", "topic": "machine_learning",
        "text": (
            "Machine learning is a subset of artificial intelligence that enables systems to "
            "learn and improve from experience without being explicitly programmed. It focuses "
            "on developing computer programs that can access data and use it to learn for "
            "themselves. The process begins with observations such as examples, direct "
            "experience, or instruction, so computers can make better decisions in the future."
        ),
    },
    {
        "id": "ml_002", "topic": "machine_learning",
        "text": (
            "Supervised learning is a type of machine learning where the algorithm is trained "
            "on labeled data. The model learns a mapping from inputs to outputs based on "
            "example input-output pairs. Common supervised learning algorithms include linear "
            "regression, logistic regression, decision trees, random forests, support vector "
            "machines, and neural networks. The labeled training set guides the model toward "
            "correct predictions on unseen data."
        ),
    },
    {
        "id": "ml_003", "topic": "machine_learning",
        "text": (
            "Unsupervised learning involves training models on data without labeled responses. "
            "The algorithm must discover hidden patterns or intrinsic structures in input data. "
            "Clustering algorithms like k-means and hierarchical clustering, and dimensionality "
            "reduction methods like PCA and t-SNE, are common unsupervised techniques. "
            "Autoencoders are neural-network-based unsupervised models used for representation "
            "learning and anomaly detection."
        ),
    },
    {
        "id": "ml_004", "topic": "machine_learning",
        "text": (
            "Reinforcement learning is a machine learning paradigm where an agent learns by "
            "interacting with an environment to maximize cumulative reward. The agent takes "
            "actions, receives feedback as rewards or penalties, and updates its policy. "
            "Q-learning, Deep Q-Networks (DQN), and Proximal Policy Optimization (PPO) are "
            "key RL algorithms. RL has achieved superhuman performance in games and robotics."
        ),
    },
    {
        "id": "dl_001", "topic": "deep_learning",
        "text": (
            "Deep learning is a subset of machine learning based on artificial neural networks "
            "with many hidden layers. These deep networks learn representations at multiple "
            "levels of abstraction automatically from raw data. Deep learning has revolutionized "
            "computer vision, natural language processing, and speech recognition, achieving "
            "state-of-the-art accuracy on benchmark tasks that previously required hand-crafted "
            "features and domain expertise."
        ),
    },
    {
        "id": "dl_002", "topic": "deep_learning",
        "text": (
            "Convolutional Neural Networks (CNNs) are specialized for processing grid-like data "
            "such as images. They use convolutional layers to automatically learn spatial "
            "hierarchies of features, followed by pooling layers for spatial reduction, and "
            "fully connected layers for classification. CNNs introduced by AlexNet in 2012 "
            "sparked the deep-learning revolution in computer vision. ResNet, VGG, and "
            "EfficientNet are widely used CNN architectures."
        ),
    },
    {
        "id": "dl_003", "topic": "deep_learning",
        "text": (
            "The Transformer architecture, introduced in 'Attention Is All You Need' (2017), "
            "revolutionized NLP by replacing recurrence with self-attention. Transformers "
            "process all tokens in parallel, enabling much faster training. The encoder-decoder "
            "structure with multi-head attention allows the model to attend to different parts "
            "of the input simultaneously. BERT, GPT, T5, and LLaMA are all transformer-based "
            "models that achieve state-of-the-art results across NLP tasks."
        ),
    },
    {
        "id": "rag_001", "topic": "rag",
        "text": (
            "Retrieval-Augmented Generation (RAG) combines retrieval systems with generative "
            "language models. Instead of relying solely on parametric knowledge in model weights, "
            "RAG retrieves relevant documents from an external knowledge base and uses them as "
            "context for generation. This reduces hallucination, allows access to up-to-date "
            "information, and makes the model's knowledge easily updateable without retraining."
        ),
    },
    {
        "id": "rag_002", "topic": "rag",
        "text": (
            "Vector databases are essential components of RAG systems. They store document "
            "embeddings and enable fast approximate nearest-neighbor search using algorithms "
            "like HNSW (Hierarchical Navigable Small World). Qdrant, Pinecone, Weaviate, and "
            "Chroma are popular vector databases. They support payload filtering, hybrid search "
            "combining dense and sparse vectors, and collection-level configuration for distance "
            "metrics such as cosine, dot product, and Euclidean distance."
        ),
    },
    {
        "id": "rag_003", "topic": "rag",
        "text": (
            "Chunking strategy critically affects RAG performance. Documents split into "
            "chunks that are too small lose context; chunks that are too large reduce retrieval "
            "precision. Common strategies include fixed-size chunking with overlap, "
            "sentence-boundary chunking, and recursive character splitting. Optimal chunk size "
            "typically ranges from 256 to 1024 tokens depending on the embedding model and "
            "downstream generation requirements."
        ),
    },
    {
        "id": "eval_001", "topic": "evaluation",
        "text": (
            "Model evaluation metrics measure how well a machine learning model performs. "
            "For classification tasks, key metrics are: accuracy (fraction of correct "
            "predictions), precision (positive predictive value), recall (true positive rate), "
            "F1-score (harmonic mean of precision and recall), and ROC-AUC (area under the "
            "receiver operating characteristic curve). Choosing the right metric depends on "
            "class imbalance and the relative cost of false positives vs false negatives."
        ),
    },
    {
        "id": "eval_002", "topic": "evaluation",
        "text": (
            "Cross-validation is a robust technique to estimate model generalization. K-fold "
            "cross-validation divides data into k equal subsets, trains on k-1 folds, and "
            "validates on the remaining fold, repeating k times and averaging results. Stratified "
            "k-fold preserves class proportions in each fold. This gives a more reliable "
            "performance estimate than a single train-test split, especially when data is limited."
        ),
    },
    {
        "id": "eval_003", "topic": "evaluation",
        "text": (
            "Overfitting occurs when a model memorizes training data including noise, leading "
            "to poor generalization. Signs include high training accuracy with low validation "
            "accuracy. Mitigation strategies include regularization (L1/L2 penalties), dropout "
            "in neural networks, early stopping, data augmentation, cross-validation for "
            "hyperparameter tuning, and ensemble methods that average over multiple models."
        ),
    },
    {
        "id": "feat_001", "topic": "feature_engineering",
        "text": (
            "Feature engineering uses domain knowledge to create informative features from raw "
            "data. Techniques include one-hot encoding for categorical variables, normalization "
            "and standardization for numerical features, polynomial features for non-linear "
            "relationships, target encoding, feature crosses, and temporal features for "
            "time-series data. Good feature engineering often has more impact on model "
            "performance than algorithm choice."
        ),
    },
    {
        "id": "feat_002", "topic": "feature_engineering",
        "text": (
            "Feature selection reduces model complexity by keeping only the most informative "
            "features. Filter methods use statistical tests like correlation, chi-squared, or "
            "mutual information. Wrapper methods like recursive feature elimination (RFE) "
            "use model performance as feedback. Embedded methods like LASSO regularization "
            "and tree-based feature importance perform selection during training. Removing "
            "irrelevant features reduces overfitting and improves interpretability."
        ),
    },
    {
        "id": "nlp_001", "topic": "nlp",
        "text": (
            "Natural Language Processing enables computers to understand, interpret, and "
            "generate human language. Core NLP tasks include tokenization, part-of-speech "
            "tagging, named entity recognition (NER), dependency parsing, sentiment analysis, "
            "machine translation, question answering, and text summarization. Modern NLP is "
            "dominated by pre-trained transformer models fine-tuned on task-specific datasets."
        ),
    },
    {
        "id": "nlp_002", "topic": "nlp",
        "text": (
            "Word embeddings are dense vector representations of words that encode semantic "
            "relationships. Word2Vec, GloVe, and FastText are classical static embedding "
            "methods where each word has one vector regardless of context. Modern contextual "
            "embeddings from BERT and RoBERTa produce different vectors for the same word "
            "depending on surrounding context, significantly improving performance on NLP tasks "
            "like NER, question answering, and text classification."
        ),
    },
    {
        "id": "pipe_001", "topic": "ml_pipeline",
        "text": (
            "An ML pipeline is an end-to-end automated sequence from data ingestion to model "
            "deployment. Components typically include data collection, validation, preprocessing, "
            "feature engineering, model training, evaluation, and deployment. MLflow tracks "
            "experiments, parameters, and artifacts. Apache Airflow orchestrates pipeline "
            "DAGs. Kubeflow runs pipelines on Kubernetes. Well-designed pipelines ensure "
            "reproducibility, auditability, and easy retraining."
        ),
    },
    {
        "id": "drift_001", "topic": "monitoring",
        "text": (
            "Data drift occurs when the statistical distribution of model input features "
            "changes over time, degrading model performance. Concept drift refers to changes "
            "in the relationship between inputs and outputs. Detection methods include the "
            "Kolmogorov-Smirnov test, Population Stability Index (PSI), and Jensen-Shannon "
            "divergence. Monitoring systems compare reference data distributions against "
            "current production data and trigger alerts when drift exceeds thresholds."
        ),
    },
    {
        "id": "drift_002", "topic": "monitoring",
        "text": (
            "Production ML monitoring tracks prediction distributions, input feature "
            "distributions, data quality metrics, model latency, throughput, and business "
            "KPIs. Alerts fire when metrics deviate beyond defined thresholds. Tools like "
            "Evidently AI generate detailed drift reports comparing reference vs current data. "
            "WhyLogs provides lightweight logging for data profiles. Arize and Fiddler offer "
            "full-featured ML observability platforms with root-cause analysis capabilities."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE  — SQLite for interaction logs & feedback records
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id          TEXT,
            query               TEXT,
            response            TEXT,
            retrieved_docs      TEXT,
            retrieval_scores    TEXT,
            time_to_read        REAL,
            follow_up_asked     INTEGER,
            similar_followup    INTEGER,
            session_query_count INTEGER,
            response_length     INTEGER,
            avg_retrieval_score REAL,
            query_complexity    REAL,
            understood          INTEGER,
            prediction          REAL,
            prompt_version      TEXT,
            timestamp           TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedback_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT,
            action          TEXT,
            reason          TEXT,
            old_prompt      TEXT,
            new_prompt      TEXT,
            metric_before   REAL,
            metric_after    REAL
        )
    """)
    conn.commit()
    conn.close()


def save_interaction(record: dict) -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO interactions (
            session_id, query, response, retrieved_docs, retrieval_scores,
            time_to_read, follow_up_asked, similar_followup,
            session_query_count, response_length, avg_retrieval_score,
            query_complexity, understood, prediction, prompt_version, timestamp
        ) VALUES (
            :session_id, :query, :response, :retrieved_docs, :retrieval_scores,
            :time_to_read, :follow_up_asked, :similar_followup,
            :session_query_count, :response_length, :avg_retrieval_score,
            :query_complexity, :understood, :prediction, :prompt_version, :timestamp
        )
    """, record)
    conn.commit()
    conn.close()


def save_feedback_log(entry: dict) -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO feedback_log (
            timestamp, action, reason, old_prompt, new_prompt,
            metric_before, metric_after
        ) VALUES (
            :timestamp, :action, :reason, :old_prompt, :new_prompt,
            :metric_before, :metric_after
        )
    """, entry)
    conn.commit()
    conn.close()


def load_interactions() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM interactions", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def load_feedback_log() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM feedback_log", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING MODEL  (cached in module scope)
# ─────────────────────────────────────────────────────────────────────────────

_embed_model: Optional[SentenceTransformer] = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def embed(texts: list[str]) -> list[list[float]]:
    return get_embed_model().encode(texts, normalize_embeddings=True).tolist()

# ─────────────────────────────────────────────────────────────────────────────
# QDRANT  — vector store with in-memory fallback
# ─────────────────────────────────────────────────────────────────────────────

_qdrant_client: Optional[QdrantClient] = None
_collection_ready: bool = False


def get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        if QDRANT_URL and QDRANT_API_KEY:
            _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            # In-memory — no cloud account required
            _qdrant_client = QdrantClient(":memory:")
    return _qdrant_client


def setup_collection() -> None:
    global _collection_ready
    if _collection_ready:
        return
    client = get_qdrant()
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    # Always upsert so in-memory is populated on each start
    vectors = embed([d["text"] for d in KNOWLEDGE_DOCS])
    points = [
        PointStruct(
            id=i,
            vector=vectors[i],
            payload={
                "doc_id": KNOWLEDGE_DOCS[i]["id"],
                "topic":  KNOWLEDGE_DOCS[i]["topic"],
                "text":   KNOWLEDGE_DOCS[i]["text"],
            },
        )
        for i in range(len(KNOWLEDGE_DOCS))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    _collection_ready = True


def retrieve(query: str, top_k: int = TOP_K, topic_filter: Optional[str] = None):
    client = get_qdrant()
    q_vec = embed([query])[0]
    filt = None
    if topic_filter:
        filt = Filter(
            must=[FieldCondition(key="topic", match=MatchValue(value=topic_filter))]
        )
    return client.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_vec,
        limit=top_k,
        query_filter=filt,
        with_payload=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT REGISTRY  — three versions, feedback loop switches between them
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_REGISTRY: dict[str, dict] = {
    "v1_standard": {
        "version": "v1_standard",
        "label":   "Standard",
        "system": (
            "You are a knowledgeable AI tutor. Use the provided context to answer "
            "the user's question accurately and concisely. If the context does not "
            "fully cover the question, say so honestly."
        ),
        "template": (
            "Context:\n{context}\n\n"
            "Question: {query}\n\n"
            "Provide a clear, accurate answer based on the context above."
        ),
    },
    "v2_simplified": {
        "version": "v2_simplified",
        "label":   "Simplified",
        "system": (
            "You are a patient AI tutor. Your goal is to make complex topics easy "
            "to understand. Use simple language, break concepts down step by step, "
            "and include analogies where helpful. Avoid jargon."
        ),
        "template": (
            "Context:\n{context}\n\n"
            "Question: {query}\n\n"
            "Explain this in simple, beginner-friendly terms. Use bullet points or "
            "numbered steps if that helps. Include a real-world analogy if possible."
        ),
    },
    "v3_detailed": {
        "version": "v3_detailed",
        "label":   "Detailed",
        "system": (
            "You are an expert AI instructor. Provide comprehensive, well-structured "
            "explanations with concrete examples. Always structure your response with "
            "clear sections: Definition → How it works → Example → Key takeaway."
        ),
        "template": (
            "Context:\n{context}\n\n"
            "Question: {query}\n\n"
            "Provide a detailed, structured explanation covering: what it is, how it "
            "works, a concrete example, and the key takeaway."
        ),
    },
}

_active_prompt_version: str = "v1_standard"


def get_active_prompt() -> dict:
    return PROMPT_REGISTRY[_active_prompt_version]


def set_active_prompt(version: str) -> None:
    global _active_prompt_version
    if version in PROMPT_REGISTRY:
        _active_prompt_version = version

# ─────────────────────────────────────────────────────────────────────────────
# RAG PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def rag_query(query: str, top_k: int = TOP_K) -> dict:
    """
    1. Embed query  →  2. Retrieve from Qdrant  →  3. Build context
    →  4. Call Groq LLM with active prompt  →  return result dict
    """
    t0 = time.time()

    # Retrieve
    results = retrieve(query, top_k=top_k)
    doc_ids  = [r.payload["doc_id"] for r in results]
    docs_text = [r.payload["text"]  for r in results]
    scores    = [round(float(r.score), 4) for r in results]
    avg_score = round(float(np.mean(scores)), 4) if scores else 0.0

    # Context — join retrieved chunks
    context = "\n\n---\n\n".join(docs_text)

    # LLM
    prompt_cfg = get_active_prompt()
    user_msg   = prompt_cfg["template"].format(context=context, query=query)

    client = Groq(api_key=GROQ_API_KEY)
    chat   = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": prompt_cfg["system"]},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=700,
    )
    response_text = chat.choices[0].message.content.strip()

    return {
        "response":            response_text,
        "retrieved_doc_ids":   doc_ids,
        "retrieval_scores":    scores,
        "avg_retrieval_score": avg_score,
        "response_length":     len(response_text),
        "latency_sec":         round(time.time() - t0, 2),
        "prompt_version":      prompt_cfg["version"],
        "context":             context,
    }

# ─────────────────────────────────────────────────────────────────────────────
# LLM-AS-JUDGE  — evaluates RAG response quality
# ─────────────────────────────────────────────────────────────────────────────

def llm_judge(query: str, context: str, response: str) -> dict:
    """Prompt Groq to score the response on relevance / clarity / completeness."""
    prompt = f"""You are an objective AI evaluator. Score the response below on three criteria.
Return ONLY a valid JSON object with no extra text or markdown fences.

Question: {query}
Retrieved context (first 400 chars): {context[:400]}
AI Response (first 500 chars): {response[:500]}

Score each criterion 1 (poor) to 5 (excellent):
  - relevance:    Does the response directly answer the question using the context?
  - clarity:      Is the response clear and easy to understand?
  - completeness: Does the response cover the key aspects of the question?
  - overall:      Overall quality score.

Return exactly: {{"relevance": X, "clarity": X, "completeness": X, "overall": X}}"""

    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=80,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return {"relevance": 3, "clarity": 3, "completeness": 3, "overall": 3, "error": str(e)}

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = [
    "time_to_read",
    "follow_up_asked",
    "similar_followup",
    "session_query_count",
    "response_length",
    "avg_retrieval_score",
    "query_complexity",
]


def generate_synthetic_data(n: int = 400) -> pd.DataFrame:
    """
    Simulate user interaction records with realistic behavioral patterns.
    Label  1 = understood,  0 = not understood.
    """
    random.seed(42)
    np.random.seed(42)
    rows = []
    for _ in range(n):
        understood = random.randint(0, 1)
        if understood:
            time_to_read      = random.uniform(15, 100)
            follow_up_asked   = random.choices([0, 1], weights=[0.72, 0.28])[0]
            similar_followup  = 0 if not follow_up_asked else random.choices([0, 1], weights=[0.85, 0.15])[0]
            session_qcount    = random.randint(1, 4)
            response_length   = random.randint(250, 700)
            avg_ret_score     = random.uniform(0.65, 0.97)
            query_complexity  = random.uniform(0.1, 0.55)
        else:
            time_to_read      = random.uniform(1, 18)
            follow_up_asked   = random.choices([0, 1], weights=[0.28, 0.72])[0]
            similar_followup  = 0 if not follow_up_asked else random.choices([0, 1], weights=[0.28, 0.72])[0]
            session_qcount    = random.randint(3, 12)
            response_length   = random.randint(40, 260)
            avg_ret_score     = random.uniform(0.25, 0.62)
            query_complexity  = random.uniform(0.5, 1.0)

        # Add small Gaussian noise to prevent perfect separation
        time_to_read  = max(0.5, time_to_read  + random.gauss(0, 2))
        avg_ret_score = max(0.1, min(1.0, avg_ret_score + random.gauss(0, 0.03)))

        rows.append({
            "time_to_read":        round(time_to_read, 2),
            "follow_up_asked":     follow_up_asked,
            "similar_followup":    similar_followup,
            "session_query_count": session_qcount,
            "response_length":     response_length,
            "avg_retrieval_score": round(avg_ret_score, 4),
            "query_complexity":    round(query_complexity, 4),
            "understood":          understood,
        })

    df = pd.DataFrame(rows)
    df.to_csv("synthetic_data.csv", index=False)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# ML MODEL  — Understanding Predictor
# ─────────────────────────────────────────────────────────────────────────────

_ml_model   = None
_ml_metrics: dict = {}


def train_model(df: Optional[pd.DataFrame] = None) -> dict:
    """
    Train LR, RF, GBM — pick best by 5-fold CV ROC-AUC.
    Saves model to disk and updates module-level cache.
    """
    global _ml_model, _ml_metrics

    if df is None or df.empty:
        existing = load_interactions()
        df = existing if len(existing) >= 30 else generate_synthetic_data(400)

    if "query_complexity" not in df.columns:
        df = df.copy()
        df["query_complexity"] = 0.5

    df = df[FEATURES + ["understood"]].dropna()
    X, y = df[FEATURES], df["understood"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": Pipeline([
            ("sc", StandardScaler()),
            ("m",  LogisticRegression(max_iter=500, C=1.0, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("sc", StandardScaler()),
            ("m",  RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)),
        ]),
        "Gradient Boosting": Pipeline([
            ("sc", StandardScaler()),
            ("m",  GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                               max_depth=4, random_state=42)),
        ]),
    }

    cv_results: dict[str, float] = {}
    best_name, best_pipe, best_auc = "", None, -1.0
    for name, pipe in candidates.items():
        auc = float(cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc").mean())
        cv_results[name] = round(auc, 4)
        if auc > best_auc:
            best_auc, best_name, best_pipe = auc, name, pipe

    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)
    y_prob = best_pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "best_model":            best_name,
        "accuracy":              round(accuracy_score(y_test, y_pred), 4),
        "f1_score":              round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":               round(roc_auc_score(y_test, y_prob), 4),
        "cv_results":            cv_results,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "train_size":            int(len(X_train)),
        "test_size":             int(len(X_test)),
        "trained_at":            datetime.now().isoformat(),
    }

    _ml_model   = best_pipe
    _ml_metrics = metrics

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": best_pipe, "metrics": metrics}, f)

    return metrics


def load_model() -> bool:
    global _ml_model, _ml_metrics
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        _ml_model   = data["model"]
        _ml_metrics = data["metrics"]
        return True
    return False


def predict_understanding(features: dict) -> dict:
    global _ml_model
    if _ml_model is None:
        if not load_model():
            train_model()
    x   = pd.DataFrame([{f: features.get(f, 0.0) for f in FEATURES}])
    prob = float(_ml_model.predict_proba(x)[0][1])
    return {
        "understood":  int(prob >= 0.5),
        "probability": round(prob, 4),
        "confidence":  round(max(prob, 1 - prob), 4),
    }

# ─────────────────────────────────────────────────────────────────────────────
# FEEDBACK LOOP
# ─────────────────────────────────────────────────────────────────────────────

_streak          = 0          # consecutive not-understood count
_total_queries   = 0
_STREAK_LIMIT    = 3          # trigger prompt switch after this many
_RETRAIN_EVERY   = 40         # retrain ML model every N interactions


def apply_feedback(prediction: dict, retrieval_scores: list) -> dict:
    global _streak, _total_queries, _active_prompt_version

    _total_queries += 1
    old_prompt  = _active_prompt_version
    action      = "none"
    reason      = ""

    if prediction["understood"]:
        _streak = max(0, _streak - 1)
    else:
        _streak += 1

    # Rule 1: understanding streak threshold → switch to more helpful prompt
    if _streak >= _STREAK_LIMIT:
        if _active_prompt_version == "v1_standard":
            set_active_prompt("v2_simplified")
            action = "prompt_switch → v2_simplified"
            reason = f"User not understanding for {_streak} queries in a row"
        elif _active_prompt_version == "v2_simplified":
            set_active_prompt("v3_detailed")
            action = "prompt_switch → v3_detailed"
            reason = "Still struggling after simplified prompt; escalating to structured detail"
        else:
            action = "max_prompt_reached"
            reason = "Already on most detailed prompt version"
        _streak = 0

    # Rule 2: low retrieval quality signal
    avg_score = float(np.mean(retrieval_scores)) if retrieval_scores else 0.5
    if avg_score < 0.55 and not prediction["understood"] and action == "none":
        action = "retrieval_quality_flag"
        reason = f"Low avg retrieval score ({avg_score:.3f}); broader retrieval recommended"

    # Rule 3: periodic ML retrain
    retrain = False
    if _total_queries % _RETRAIN_EVERY == 0:
        df = load_interactions()
        if len(df) >= 30:
            train_model(df)
            retrain = True
            if action == "none":
                action = "ml_model_retrain"
                reason = f"Periodic retrain at {_total_queries} interactions using real data"

    new_prompt = _active_prompt_version
    save_feedback_log({
        "timestamp":    datetime.now().isoformat(),
        "action":       action,
        "reason":       reason,
        "old_prompt":   old_prompt,
        "new_prompt":   new_prompt,
        "metric_before": prediction["probability"],
        "metric_after":  prediction["probability"],
    })

    return {
        "action":      action,
        "reason":      reason,
        "new_prompt":  new_prompt,
        "streak":      _streak,
        "retrained":   retrain,
    }

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING  — Evidently drift report
# ─────────────────────────────────────────────────────────────────────────────

def generate_monitoring_report() -> str:
    """Returns path to generated HTML report, or an error string."""
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError:
        return "error: evidently not installed"

    df = load_interactions()
    if len(df) < 40:
        df = generate_synthetic_data(200)

    df = df[FEATURES + ["understood"]].dropna()
    if len(df) < 20:
        return "error: not enough data for drift report"

    mid = len(df) // 2
    ref = df.iloc[:mid].copy()
    cur = df.iloc[mid:].copy()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    path = "monitoring_report.html"
    report.save_html(path)
    return path


def get_system_stats() -> dict:
    df = load_interactions()
    fb = load_feedback_log()
    return {
        "total_interactions":  int(len(df)),
        "understood_rate":     round(float(df["understood"].mean()), 3) if not df.empty else 0.0,
        "avg_retrieval_score": round(float(df["avg_retrieval_score"].mean()), 3) if not df.empty else 0.0,
        "active_prompt":       _active_prompt_version,
        "feedback_actions":    int(len(fb)),
        "streak":              _streak,
        "model_metrics":       _ml_metrics,
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY-POINT  — called by app.py for every user query
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    query:               str,
    session_id:          str,
    session_query_count: int,
    time_to_read:        float = 10.0,
    follow_up_asked:     int   = 0,
    similar_followup:    int   = 0,
    query_complexity:    float = 0.5,
) -> dict:
    """
    Full adaptive pipeline:
      Query → RAG → LLM Response → Feature Extraction
      → ML Prediction → LLM Judge (every 3rd query)
      → Feedback Loop → DB Logging → Return result dict
    """
    # ── RAG ───────────────────────────────────────────────────────────────────
    rag = rag_query(query)

    # ── Feature vector ────────────────────────────────────────────────────────
    features = {
        "time_to_read":        time_to_read,
        "follow_up_asked":     follow_up_asked,
        "similar_followup":    similar_followup,
        "session_query_count": session_query_count,
        "response_length":     rag["response_length"],
        "avg_retrieval_score": rag["avg_retrieval_score"],
        "query_complexity":    query_complexity,
    }

    # ── ML prediction ─────────────────────────────────────────────────────────
    prediction = predict_understanding(features)

    # ── Feedback loop ─────────────────────────────────────────────────────────
    feedback = apply_feedback(prediction, rag["retrieval_scores"])

    # ── LLM judge (every 3rd query to save API tokens) ────────────────────────
    judge_scores: dict = {}
    if session_query_count % 3 == 0:
        judge_scores = llm_judge(query, rag["context"], rag["response"])

    # ── Log to DB ─────────────────────────────────────────────────────────────
    save_interaction({
        "session_id":          session_id,
        "query":               query,
        "response":            rag["response"],
        "retrieved_docs":      json.dumps(rag["retrieved_doc_ids"]),
        "retrieval_scores":    json.dumps(rag["retrieval_scores"]),
        "time_to_read":        time_to_read,
        "follow_up_asked":     follow_up_asked,
        "similar_followup":    similar_followup,
        "session_query_count": session_query_count,
        "response_length":     rag["response_length"],
        "avg_retrieval_score": rag["avg_retrieval_score"],
        "query_complexity":    query_complexity,
        "understood":          prediction["understood"],
        "prediction":          prediction["probability"],
        "prompt_version":      rag["prompt_version"],
        "timestamp":           datetime.now().isoformat(),
    })

    return {
        "response":             rag["response"],
        "retrieved_doc_ids":    rag["retrieved_doc_ids"],
        "retrieval_scores":     rag["retrieval_scores"],
        "avg_retrieval_score":  rag["avg_retrieval_score"],
        "latency_sec":          rag["latency_sec"],
        "prompt_version":       rag["prompt_version"],
        "prediction":           prediction,
        "feedback":             feedback,
        "judge_scores":         judge_scores,
        "features_used":        features,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP  — called once at app startup
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap() -> None:
    """Initialise DB → embed knowledge base into Qdrant → train/load ML model."""
    init_db()
    setup_collection()     # embeds all 20 docs into Qdrant (or in-memory)
    if not load_model():
        df = generate_synthetic_data(400)
        train_model(df)


if __name__ == "__main__":
    bootstrap()
    print("Backend ready.")
    import json
    print(json.dumps(get_system_stats(), indent=2, default=str))
