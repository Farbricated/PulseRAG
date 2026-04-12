# Grokker v3
### Adaptive & Evaluative AI Pipeline

---

## What is Grokker?

Grokker is an adaptive AI tutoring system that:
1. **Answers** questions using a RAG pipeline (ChromaDB + Groq Llama3-70b)
2. **Predicts** whether the user understood the answer (ML model)
3. **Evaluates** response quality (LLM-as-Judge, 5 dimensions)
4. **Adapts** its teaching style automatically (6-rule feedback loop)

*"To grok" means to deeply understand something.*

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Enter your free Groq API key when prompted → https://console.groq.com

You can also set it as an environment variable to skip the prompt:
```bash
export GROQ_API_KEY=gsk_...
streamlit run app.py
```

---

## Files

```
grokker_v3/
├── backend.py          ← All logic: ChromaDB, RAG, ML, Feedback, A/B, Monitoring
├── app.py              ← Streamlit UI — 6 tabs
├── requirements.txt
└── README.md
```

Runtime-generated (add to .gitignore):
```
chroma_db/              ← ChromaDB persistent vector index
grokker.db              ← SQLite: interactions, feedback_log, ab_tests, judge_log
grokker_model.pkl       ← Trained ML model
synthetic_data.csv      ← Generated training data
monitoring_report.html  ← Evidently drift report
```

---

## Full Pipeline

```
User Query
  → Topic Router        (keyword → topic filter)
  → Query Rewriter      (LLM reformulates short/ambiguous queries)
  → Embed               (all-MiniLM-L6-v2 · 384-dim)
  → Hybrid Retrieve     (ChromaDB cosine + BM25 → RRF fusion · top-6)
  → Memory Inject       (last 3 conversation turns)
  → A/B Prompt Select   (MD5 hash → group → v1/v2/v3/v4)
  → Generate            (Groq llama3-70b-8192)
  → Hallucination Guard (response-context term overlap → risk score)
  → Feature Extract     (9 behavioral signals)
  → ML Predict          (LR / RF / GBM / XGBoost → best by 5-fold CV AUC)
  → LLM Judge           (relevance / clarity / completeness / groundedness / conciseness)
  → Feedback Loop       (6 rules → prompt escalation / flags / retrain)
  → A/B Log             (group + outcome → ab_tests table)
  → SQLite Log          (full interaction record)
  → Evidently Monitor   (on-demand DataDriftPreset report)
```

---

## 9 Behavioral Features

| Feature | Why it matters |
|---|---|
| time_to_read | Longer reading → higher engagement → understanding |
| follow_up_asked | Follow-up questions signal confusion |
| similar_followup | Same question rephrased → stuck signal |
| session_query_count | High count → persistence or frustration |
| response_length | Very short responses may be incomplete |
| avg_retrieval_score | Higher cosine similarity → better context |
| query_complexity | Complex questions are harder to understand |
| hallucination_risk | Response unsupported by context → confusion |
| response_diversity | Lexical richness correlates with answer quality |

---

## Feedback Loop — 6 Rules

| Rule | Trigger | Action |
|---|---|---|
| 1 | 3 consecutive not-understood | Prompt escalation: v1→v2→v3→v4 |
| 2 | Avg retrieval score < 0.50 + confusion | Log retrieval quality flag |
| 3 | Hallucination risk > 0.65 | Log hallucination risk flag |
| 4 | Last 5 responses in session all wrong | Switch to Socratic prompt mode |
| 5 | All retrieved docs from single topic | Log retrieval diversity flag |
| 6 | Every 30 interactions | Retrain ML model on real data |

---

## A/B Testing

Sessions are deterministically assigned via MD5(session_id) % 3:

| Group | Prompt | Strategy |
|---|---|---|
| control | v1_standard | Concise, accurate answer |
| treatment_a | v2_simplified | Simple language + analogies |
| treatment_b | v3_detailed | 4-section structured format |

---

## Improvements over original V3

- Cleaner `_load_table()` helper removes duplicated DB read logic
- `rewrite_query()` safety-checks the rewrite length before using it
- `app.py` uses a centralized `ADAPT_MAP` dict instead of scattered if/else
- `app.py` uses a centralized `PROMPT_LABELS` dict in one place
- `apply_feedback()` streak logic consolidated to a single line
- All DB writes use a single shared `_insert()` helper
- `load_model()` wrapped in try/except to gracefully handle corrupted pickle
- `llm_judge()` clamps scores to 1-5 to prevent bad JSON from crashing the UI
- Monitoring button added directly in the Architecture tab
- Sidebar `st.caption()` used correctly (no raw HTML needed)
- `requirements.txt` kept minimal and pinned at working lower bounds