# PulseRAG v5
### Adaptive & Evaluative AI Pipeline — Indecimal AI Systems Challenge

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER QUERY                                          │
└────────────────────────────────┬───────────────────────────────────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │  [1] TOPIC ROUTER                    │
              │  keyword → topic filter              │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │  [2] QUERY REWRITER + VARIANTS       │
              │  LLM reformulation + 3 variants      │
              │  cached → no duplicate API calls     │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │  [3] EMBED                           │
              │  all-MiniLM-L6-v2 · 384-dim          │
              │  GPU-aware (CUDA/MPS/CPU)            │
              └──────────────────┬──────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌────────▼────────┐   ┌──────────▼──────────┐   ┌──────▼──────────┐
│  Dense Retrieve  │   │  BM25 Retrieve      │   │  (per variant)  │
│  ChromaDB HNSW  │   │  real avgdl corpus  │   │  3 query forms  │
└────────┬────────┘   └──────────┬──────────┘   └──────┬──────────┘
         └───────────────────────┼───────────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │  [4] RRF FUSION                      │
              │  Reciprocal Rank Fusion · top-10     │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │  [5] SCORE FILTER                    │
              │  Remove docs below threshold (0.30)  │
              │  Session can lower threshold dynamically│
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │  [6] CROSS-ENCODER RERANK            │
              │  ms-marco-MiniLM-L-6-v2 · top-6     │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │  [7] MEMORY INJECT                   │
              │  Last 3 conversation turns           │
              │  SQLite-persisted across restarts    │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │  [8] A/B PROMPT SELECT               │
              │  MD5(session_id) % 3 → group         │
              │  v1_standard / v2_simplified / v3_detailed│
              │  Feedback escalation overrides A/B   │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │  [9] GENERATE                        │
              │  Groq llama-3.1-8b-instant           │
              │  Graceful error handling             │
              └──────────────────┬──────────────────┘
                                 │
         ┌───────────────────────┼────────────────────────────┐
         │                       │                            │
┌────────▼────────┐   ┌──────────▼──────────┐   ┌───────────▼──────────┐
│  [10] HAL GUARD  │   │  [11] RAGAS METRICS  │   │  [12] FEATURE EXTRACT│
│  term overlap   │   │  context_precision   │   │  9 base + 3 inter-   │
│  risk 0-1       │   │  context_recall      │   │  action features     │
└────────┬────────┘   │  faithfulness        │   └───────────┬──────────┘
         │            └──────────┬──────────┘               │
         └───────────────────────┼────────────────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │  [13] ML PREDICT                     │
              │  LR / RF / GBM / XGBoost            │
              │  5-fold CV AUC selection             │
              │  Calibrated threshold (PR curve)     │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │  [14] LLM-AS-JUDGE                   │
              │  Every query (not every 3rd)         │
              │  5 dims: relevance/clarity/complete/ │
              │  groundedness/conciseness            │
              │  None on failure (no fake 3s)        │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │  [15] FEEDBACK LOOP (6 rules)        │
              │  Bi-directional escalation/deescalation│
              │  Rules 2,3,5 actively change behavior │
              │  metric_after resolved on next query  │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │  [16] PSI DRIFT CHECK                │
              │  Population Stability Index          │
              │  Auto-retrain if PSI > 0.20          │
              │  Every 20 queries                    │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │  [17] LOG (SQLite)                   │
              │  interactions · feedback_log         │
              │  ab_tests · judge_log                │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │  [18] IMPROVE                        │
              │  Next query uses updated state       │
              │  Prompt/threshold/retrieval updated  │
              └─────────────────────────────────────┘
```

---

## What is PulseRAG?

PulseRAG is a production-oriented adaptive AI tutoring system that:

1. **Retrieves** relevant documents using hybrid BM25 + dense search with RRF fusion, cross-encoder re-ranking, and quality filtering
2. **Generates** responses via Groq (llama-3.1-8b-instant) using 4 adaptive prompt strategies
3. **Evaluates** response quality using RAGAS-style metrics (context precision, recall, faithfulness) and LLM-as-Judge (5 dimensions)
4. **Predicts** whether the user understood the response using an ML model with 12 features (9 base + 3 interaction)
5. **Adapts** bi-directionally via a 6-rule feedback loop that actively modifies retrieval strategy and prompt selection
6. **Monitors** data drift via PSI with automatic model retraining

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# 3. (Optional) Generate sample upload files
python generate_upload_data.py

# 4. Run
streamlit run app.py
```

Free Groq key → https://console.groq.com

---

## Files

```
pulserag_v5/
├── backend.py                  ← All logic (1 file, fully modular)
├── app.py                      ← Streamlit UI — 5 tabs
├── generate_upload_data.py     ← Creates 3 sample upload files
├── requirements.txt
└── README.md

Runtime-generated (gitignored):
├── chroma_db/                  ← ChromaDB persistent HNSW index
├── pulserag.db                 ← SQLite: 4 tables
├── pulserag_model.pkl          ← Trained ML model + calibrated threshold
├── synthetic_data.csv          ← Generated training data
└── monitoring_report.html      ← Evidently drift report (on demand)
```

---

## ML Pipeline Detail

### 12 Features

| Feature | Type | Signal |
|---|---|---|
| time_to_read | base | Real seconds between response render and next Send |
| follow_up_asked | base | Last assistant msg ended with ? AND user replied |
| similar_followup | base | Embedding cosine sim > 0.82 to previous query |
| session_query_count | base | Number of queries in session |
| response_length | base | Character count of generated response |
| avg_retrieval_score | base | Mean cosine similarity of retrieved docs |
| query_complexity | base | Word count + complexity word heuristic |
| hallucination_risk | base | 1 - (response/context term overlap) |
| response_diversity | base | Type-token ratio of response |
| quality_weighted_length | **interaction** | response_length × avg_retrieval_score |
| engagement_depth | **interaction** | session_query_count × follow_up_asked |
| risk_under_load | **interaction** | hallucination_risk × query_complexity |

### Model Selection

4 candidates trained in parallel via `ThreadPoolExecutor`:
- Logistic Regression (L2, balanced class weight)
- Random Forest (150 trees, max_depth=8)
- Gradient Boosting (100 estimators, lr=0.1)
- XGBoost (if available)

Selection via 5-fold CV ROC-AUC. Best model then has its classification threshold calibrated using precision-recall curve optimization on a held-out validation split (3-way: train/val/test).

### RAGAS Metrics (in-pipeline, no external service)

- **context_precision**: fraction of retrieved docs with embedding similarity > 0.4 to query
- **context_recall**: fraction of response content grounded in retrieved context (term overlap proxy)
- **faithfulness**: sentence-level check — fraction of response sentences with ≥30% term overlap with context

---

## Feedback Loop — 6 Rules (all actively change behavior)

| # | Trigger | Active Behavior |
|---|---|---|
| 1 | 3 consecutive not-understood | Escalate prompt: v1→v2→v3→v4 |
| 1b | 3 consecutive understood | De-escalate prompt: v4→v3→v2→v1 |
| 2 | Avg retrieval < 0.50 + confusion | Lower score threshold → broaden retrieval |
| 3 | Hallucination risk > 0.65 | Switch to v3_detailed grounded prompt |
| 4 | Last 5 session all wrong | Switch to v4_socratic |
| 5 | All docs from single topic | Lower threshold → diversify retrieval |
| 6 | Every 30 interactions | Retrain ML model on real interaction data |

---

## Monitoring

- **Evidently DataDriftPreset**: on-demand, real data only (min 40 interactions required)
- **PSI drift detection**: automated, runs every 20 queries, triggers retraining at PSI > 0.20
- **Latency tracking**: p50/p95/p99 across all queries, displayed in Analytics tab
- **Judge vs ML correlation**: Pearson correlation between judge overall score and ML probability

---

## A/B Testing

Sessions deterministically assigned via MD5(session_id) % 3:

| Group | Prompt | Strategy |
|---|---|---|
| control | v1_standard | Concise, grounded answer |
| treatment_a | v2_simplified | Simple language + bullet points |
| treatment_b | v3_detailed | 4-section structured format |

Feedback escalation layered on top of A/B assignment — once escalated, the escalated prompt overrides the A/B assignment for that session.

---

## Evaluation Design

### Why these metrics?

**ROC-AUC** — threshold-agnostic measure of discriminative ability. Primary selection criterion for model comparison.

**Calibrated threshold** — raw 0.5 is rarely optimal for imbalanced behavioral signals. We optimize the threshold on a held-out validation set by maximizing F1 on the precision-recall curve.

**Context precision** — validates that retrieved documents are relevant, not just voluminous. High retrieval with low precision bloats context with noise.

**Context recall** — validates that the response is grounded in what was retrieved. Low recall indicates hallucination risk.

**Faithfulness** — sentence-level grounding check. More granular than term overlap at document level.

**LLM Judge** — covers dimensions orthogonal to ML model: response clarity, completeness, and conciseness. Runs every query and is stored alongside RAGAS for cross-correlation analysis.

---

## Stack

| Component | Tool | Why |
|---|---|---|
| LLM | Groq llama-3.1-8b-instant | Free tier, ~500 tok/s |
| Vector DB | ChromaDB HNSW | Persistent, cosine similarity, local-first |
| Embed | all-MiniLM-L6-v2 | 384-dim, fast, GPU-aware |
| Re-rank | ms-marco-MiniLM-L-6-v2 | Cross-encoder, precision improvement |
| ML | sklearn + XGBoost | 4 candidates, parallel CV, PR-calibrated |
| Drift | PSI + Evidently | Quantitative + visual monitoring |
| DB | SQLite 4 tables | Zero-config, persistent interaction log |
| UI | Streamlit 5 tabs | Rapid deployment, real-time updates |

---

## Decisions & Trade-offs

**ChromaDB over Qdrant**: ChromaDB requires no infrastructure setup, enabling faster local development. In production, Qdrant provides better horizontal scaling and filtering. The retrieval interface is abstracted so swapping is a backend change only.

**Groq (llama-3.1-8b-instant) over GPT-4**: Free tier, ~500 tokens/second inference. Sufficient quality for tutoring responses. GPT-4 would improve judge score quality but adds cost and latency.

**In-pipeline RAGAS over RAGAS library**: The RAGAS library requires ground truth annotations. Our proxy metrics (embedding similarity, term overlap) run without external dependencies and provide directional signals for retrieval quality.

**Synthetic training data**: Real behavioral data requires active users. Overlapping Gaussian distributions (target AUC ~0.78) ensure the ML model sees a realistic classification problem, not a trivially separable one. Model automatically retrains on real data once 30+ interactions are logged.

**SQLite over PostgreSQL**: Zero infrastructure. Scales to thousands of interactions per session. In production with concurrent users, PostgreSQL or DynamoDB would be appropriate.