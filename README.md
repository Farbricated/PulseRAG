# Indecimal AI Systems Challenge
## Adaptive & Evaluative AI Pipeline

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Groq API key
#    Edit .streamlit/secrets.toml  →  GROQ_API_KEY = "gsk_..."

# 3. Run
streamlit run app.py
```

Open **http://localhost:8501**

> **Qdrant is optional.** Leave `QDRANT_URL` blank and the system automatically
> uses an in-memory Qdrant instance — perfect for local development and demos.
> For persistence across restarts, create a free cluster at cloud.qdrant.io.

---

## Free API Keys

| Service | URL | Notes |
|---|---|---|
| Groq | https://console.groq.com | Required — free, very fast |
| Qdrant Cloud | https://cloud.qdrant.io | Optional — free 1 GB cluster |

---

## Files

```
indecimal_final/
├── backend.py          ← All logic: RAG, Qdrant, Groq, ML, Feedback, Monitoring
├── app.py              ← Streamlit UI — 5 tabs
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── secrets.toml   ← Your API keys (gitignored)
└── README.md
```

---

## Full Pipeline

```
User Query
  → Embed (all-MiniLM-L6-v2)
  → Retrieve top-5 (Qdrant cosine similarity)
  → Build context
  → Generate response (Groq llama3-70b)
  → Extract behavioral features
  → Predict understanding (ML: LR / RF / GBM — best by CV AUC)
  → LLM-as-Judge evaluation (every 3rd query)
  → Feedback loop (prompt switch / retrieval flag / model retrain)
  → Log to SQLite
  → Evidently drift monitoring on demand
  → Next query uses improved prompt + retrained model
```

---

## Challenge Coverage

| Requirement | Implementation |
|---|---|
| Data Layer | `generate_synthetic_data()` — 400 samples with realistic behavioral signals |
| RAG Pipeline | `rag_query()` — Qdrant retrieval + context construction + Groq generation |
| ML Model | `train_model()` — LR vs RF vs GBM, 5-fold CV, best by ROC-AUC |
| Evaluation | `llm_judge()` + sklearn metrics (accuracy, F1, ROC-AUC) |
| Feedback Mechanism | `apply_feedback()` — prompt switching + retrieval flags + periodic retrain |
| Monitoring | `generate_monitoring_report()` — Evidently DataDriftPreset |
| Qdrant | Cosine similarity, payload filtering, in-memory fallback |

---

## Evaluation Metrics — Why These?

**ML Model:**
- **Accuracy** — baseline correctness
- **F1-Score** — robust to class imbalance (understood vs not)
- **ROC-AUC** — threshold-independent, best for probabilistic outputs
- **5-fold CV** — reliable estimate with limited labeled data

**RAG/LLM:**
- **Avg retrieval cosine score** — direct measure of context relevance
- **LLM-as-Judge** — relevance / clarity / completeness scored 1–5
- **Understanding rate** — ultimate downstream user-outcome proxy

---

## AWS Deployment (see Architecture tab in app)

- **EC2 t2.micro** — FastAPI backend (free 12 months)
- **S3** — model artifacts, logs, Evidently reports (free 5 GB)
- **Streamlit Cloud** — frontend (free forever, deploys from GitHub)
- **Qdrant Cloud** — vector DB (free 1 GB)
- **Groq API** — LLM (free tier)
