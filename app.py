"""
=============================================================================
Indecimal AI Systems Challenge — app.py
=============================================================================
Streamlit frontend.  Five tabs:
  1. 💬 Chat          — RAG Q&A with live signal panel
  2. 📊 Dashboard     — interaction trends, retrieval stats
  3. 🤖 ML Model      — train / evaluate / manual predict
  4. 🔁 Feedback Log  — every adaptation the loop has made
  5. 🏗 Architecture  — full system diagram + metric justifications
=============================================================================
"""

from __future__ import annotations

import os
import time
import uuid
import json
import math
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indecimal AI Pipeline",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

/* ── base ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0b0d11 !important;
    color: #dde1e8;
}
[data-testid="stSidebar"] {
    background: #0f1117 !important;
    border-right: 1px solid #1e222e;
}
[data-testid="metric-container"] {
    background: #13171f;
    border: 1px solid #1e2535;
    border-radius: 8px;
    padding: 14px 18px;
}

/* ── chat bubbles ── */
.user-bubble {
    background: #111827;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 10px 0 4px 0;
    font-size: 0.93rem;
    line-height: 1.55;
}
.ai-bubble {
    background: #0f1a10;
    border-left: 3px solid #22c55e;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 4px 0 10px 0;
    font-size: 0.93rem;
    line-height: 1.65;
    white-space: pre-wrap;
}

/* ── signal cards ── */
.mono-card {
    background: #13171f;
    border: 1px solid #1e2535;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 5px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.8;
    color: #8b9ab0;
}
.understood-yes {
    background: #071510;
    border: 1px solid #22c55e;
    border-radius: 6px;
    padding: 9px 14px;
    color: #22c55e;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.understood-no {
    background: #150707;
    border: 1px solid #ef4444;
    border-radius: 6px;
    padding: 9px 14px;
    color: #ef4444;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.feedback-pill {
    background: #15100a;
    border: 1px solid #f59e0b;
    border-radius: 6px;
    padding: 8px 14px;
    color: #f59e0b;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    margin-top: 8px;
}
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 2.5px;
    color: #3b4a60;
    text-transform: uppercase;
    margin: 14px 0 6px 0;
}
.pipeline-row {
    background: #0d1016;
    border: 1px solid #181d28;
    border-radius: 5px;
    padding: 7px 12px;
    margin: 3px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.77rem;
    color: #6b7a92;
}

/* ── buttons ── */
.stButton > button {
    background: #1d4ed8 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    padding: 8px 18px !important;
}
.stButton > button:hover {
    background: #2563eb !important;
}

/* ── inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    background: #0f1117 !important;
    border: 1px solid #1e2535 !important;
    border-radius: 6px !important;
    color: #dde1e8 !important;
}
.stSlider { accent-color: #3b82f6; }

/* ── tabs ── */
[data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #1e222e; }
[data-baseweb="tab"]      { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: #3b4a60 !important; }
[aria-selected="true"]    { color: #3b82f6 !important; border-bottom: 2px solid #3b82f6 !important; }

/* ── misc ── */
#MainMenu, footer, header { visibility: hidden; }
.stDivider { border-color: #1e222e; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# BACKEND INIT  (cached — runs once per process)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⚙ Bootstrapping pipeline…")
def _load_backend(groq_key: str, qdrant_url: str, qdrant_key: str):
    """Import backend, inject keys, run bootstrap. Cached by key values."""
    import backend as b
    b.GROQ_API_KEY   = groq_key
    b.QDRANT_URL     = qdrant_url
    b.QDRANT_API_KEY = qdrant_key
    b.bootstrap()
    return b

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

def _ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

_ss("session_id",        str(uuid.uuid4())[:8])
_ss("messages",          [])
_ss("query_count",       0)
_ss("pipeline_results",  [])
_ss("backend",           None)
_ss("ready",             False)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:1.25rem;
                font-weight:600;color:#e2e8f0;letter-spacing:-0.5px;">
        ⚙ INDECIMAL
    </div>
    <div style="font-size:0.7rem;color:#3b4a60;letter-spacing:2px;
                text-transform:uppercase;margin-top:2px;">
        Adaptive AI Pipeline
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── API Keys ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Configuration</div>', unsafe_allow_html=True)

    # Read from st.secrets if available (Streamlit Cloud deployment)
    default_groq   = st.secrets.get("GROQ_API_KEY",   os.getenv("GROQ_API_KEY",   ""))
    default_qurl   = st.secrets.get("QDRANT_URL",     os.getenv("QDRANT_URL",     ""))
    default_qkey   = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY", ""))

    groq_key   = st.text_input("Groq API Key *", type="password", value=default_groq,
                                help="Get free key → console.groq.com")
    qdrant_url = st.text_input("Qdrant URL (optional)", value=default_qurl,
                                placeholder="https://xxx.qdrant.io:6333",
                                help="Leave blank → in-memory Qdrant (no persistence)")
    qdrant_key = st.text_input("Qdrant API Key (optional)", type="password", value=default_qkey)

    if st.button("🚀  Initialize Pipeline", use_container_width=True):
        if not groq_key.strip():
            st.error("Groq API key is required.")
        else:
            with st.spinner("Loading embeddings + building index…"):
                try:
                    b = _load_backend(groq_key.strip(), qdrant_url.strip(), qdrant_key.strip())
                    st.session_state.backend = b
                    st.session_state.ready   = True
                    st.success("Pipeline ready ✓")
                except Exception as exc:
                    st.error(f"Init failed: {exc}")

    st.divider()

    # ── Status ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Pipeline Status</div>', unsafe_allow_html=True)
    if st.session_state.ready:
        b     = st.session_state.backend
        stats = b.get_system_stats()
        best  = stats["model_metrics"].get("best_model", "–")
        st.markdown(f"""
        <div class="pipeline-row">✅  RAG → Qdrant + Groq</div>
        <div class="pipeline-row">✅  ML Model → {best}</div>
        <div class="pipeline-row">✅  Feedback Loop → active</div>
        <div class="pipeline-row">📌  Prompt → {stats['active_prompt']}</div>
        <div class="pipeline-row">📊  Interactions → {stats['total_interactions']}</div>
        <div class="pipeline-row">🔁  Streak → {stats['streak']}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="pipeline-row">⏸  Awaiting init</div>', unsafe_allow_html=True)

    st.divider()

    # ── Session ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Session</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="mono-card">
    ID &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {st.session_state.session_id}<br>
    Queries : {st.session_state.query_count}
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄  New Session", use_container_width=True):
        st.session_state.session_id       = str(uuid.uuid4())[:8]
        st.session_state.messages         = []
        st.session_state.query_count      = 0
        st.session_state.pipeline_results = []
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_chat, tab_dash, tab_model, tab_fb, tab_arch = st.tabs([
    "💬  Chat",
    "📊  Dashboard",
    "🤖  ML Model",
    "🔁  Feedback Log",
    "🏗  Architecture",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    if not st.session_state.ready:
        st.info("👈  Enter your **Groq API key** in the sidebar and click **Initialize Pipeline**.")
        st.stop()

    b = st.session_state.backend

    col_chat, col_panel = st.columns([3, 1], gap="large")

    # ── Left: conversation ──────────────────────────────────────────────────
    with col_chat:
        st.markdown("### AI Tutor — RAG Powered")
        st.markdown(
            '<div style="font-size:0.72rem;color:#3b4a60;letter-spacing:1.5px;'
            'text-transform:uppercase;margin-bottom:12px;">'
            'Topics: Machine Learning · Deep Learning · RAG · NLP · Evaluation · Monitoring'
            '</div>',
            unsafe_allow_html=True,
        )

        # Chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="user-bubble">🧑&nbsp; {msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="ai-bubble">🤖&nbsp; {msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )

        # Input form
        with st.form("chat_form", clear_on_submit=True):
            user_query = st.text_area(
                "Question",
                height=90,
                placeholder=(
                    "Ask anything about ML/AI…\n"
                    "e.g. How does RAG reduce hallucinations?\n"
                    "     What is the difference between precision and recall?"
                ),
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Send  →", use_container_width=False)

        if submitted and user_query.strip():
            q = user_query.strip()
            st.session_state.query_count += 1

            # Rough query complexity: word count + question-word density
            words      = q.lower().split()
            q_words    = {"why", "how", "explain", "compare", "difference",
                          "relationship", "mechanism", "contrast", "describe"}
            complexity = min(1.0, len(words) / 30 + 0.25 * sum(1 for w in words if w in q_words))

            with st.spinner("Retrieving → Generating → Predicting…"):
                result = b.run_pipeline(
                    query               = q,
                    session_id          = st.session_state.session_id,
                    session_query_count = st.session_state.query_count,
                    time_to_read        = 10.0,   # updated by 👍/👎
                    follow_up_asked     = 0,
                    similar_followup    = 0,
                    query_complexity    = round(complexity, 3),
                )

            st.session_state.messages.append({"role": "user",      "content": q})
            st.session_state.messages.append({"role": "assistant",  "content": result["response"]})
            st.session_state.pipeline_results.append({"query": q, "result": result})
            st.rerun()

    # ── Right: live signals ─────────────────────────────────────────────────
    with col_panel:
        st.markdown("### Live Signals")

        if not st.session_state.pipeline_results:
            st.markdown(
                '<div class="mono-card" style="color:#3b4a60;">Ask a question<br>to see signals.</div>',
                unsafe_allow_html=True,
            )
        else:
            last   = st.session_state.pipeline_results[-1]["result"]
            pred   = last["prediction"]
            fb     = last["feedback"]
            judge  = last.get("judge_scores", {})

            # Understanding badge
            if pred["understood"]:
                st.markdown(
                    f'<div class="understood-yes">✓ UNDERSTOOD<br>'
                    f'p = {pred["probability"]:.3f} | conf = {pred["confidence"]:.3f}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="understood-no">✗ NOT UNDERSTOOD<br>'
                    f'p = {pred["probability"]:.3f} | conf = {pred["confidence"]:.3f}</div>',
                    unsafe_allow_html=True,
                )

            # Retrieval stats
            st.markdown('<div class="section-label">Retrieval</div>', unsafe_allow_html=True)
            scores_str = str([round(s, 3) for s in last["retrieval_scores"]])
            st.markdown(f"""
            <div class="mono-card">
            avg score : {last['avg_retrieval_score']:.4f}<br>
            top scores: {scores_str}<br>
            latency   : {last['latency_sec']}s<br>
            prompt    : {last['prompt_version']}
            </div>
            """, unsafe_allow_html=True)

            # Feedback action
            if fb["action"] not in ("none", ""):
                st.markdown(
                    f'<div class="feedback-pill">⚡ {fb["action"]}<br>'
                    f'<span style="opacity:0.7">{fb["reason"][:70]}</span></div>',
                    unsafe_allow_html=True,
                )

            # LLM judge scores
            if judge and "error" not in judge:
                st.markdown('<div class="section-label">LLM Judge</div>', unsafe_allow_html=True)
                for k in ["relevance", "clarity", "completeness", "overall"]:
                    v = judge.get(k, 3)
                    st.progress(v / 5, text=f"{k}: {v}/5")

            # Explicit feedback
            st.markdown('<div class="section-label">Did this help?</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("👍", use_container_width=True, key="thumbs_up"):
                    b.save_interaction({
                        "session_id":          st.session_state.session_id,
                        "query":               "__feedback_positive__",
                        "response":            "",
                        "retrieved_docs":      "[]",
                        "retrieval_scores":    "[]",
                        "time_to_read":        35.0,
                        "follow_up_asked":     0,
                        "similar_followup":    0,
                        "session_query_count": st.session_state.query_count,
                        "response_length":     0,
                        "avg_retrieval_score": 0.82,
                        "query_complexity":    0.3,
                        "understood":          1,
                        "prediction":          0.88,
                        "prompt_version":      last["prompt_version"],
                        "timestamp":           datetime.now().isoformat(),
                    })
                    st.success("👍 Logged")
            with c2:
                if st.button("👎", use_container_width=True, key="thumbs_down"):
                    b.save_interaction({
                        "session_id":          st.session_state.session_id,
                        "query":               "__feedback_negative__",
                        "response":            "",
                        "retrieved_docs":      "[]",
                        "retrieval_scores":    "[]",
                        "time_to_read":        2.5,
                        "follow_up_asked":     1,
                        "similar_followup":    1,
                        "session_query_count": st.session_state.query_count,
                        "response_length":     0,
                        "avg_retrieval_score": 0.38,
                        "query_complexity":    0.8,
                        "understood":          0,
                        "prediction":          0.18,
                        "prompt_version":      last["prompt_version"],
                        "timestamp":           datetime.now().isoformat(),
                    })
                    st.warning("👎 Logged — system adapts!")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    if not st.session_state.ready:
        st.info("Initialize the pipeline first.")
        st.stop()

    b     = st.session_state.backend
    stats = b.get_system_stats()
    df    = b.load_interactions()

    st.markdown("### System Dashboard")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Interactions",        stats["total_interactions"])
    c2.metric("Understanding Rate",  f"{stats['understood_rate']*100:.1f}%")
    c3.metric("Avg Retrieval Score", f"{stats['avg_retrieval_score']:.3f}")
    c4.metric("Feedback Actions",    stats["feedback_actions"])
    c5.metric("Active Prompt",       stats["active_prompt"])

    st.divider()

    if df.empty:
        st.info("Chat with the tutor to populate the dashboard.")
    else:
        row1_l, row1_r = st.columns(2, gap="large")

        with row1_l:
            st.markdown("#### Understanding Rate (rolling-5 avg)")
            tmp = df.copy()
            tmp["idx"]    = range(len(tmp))
            tmp["rolling"] = tmp["understood"].rolling(5, min_periods=1).mean()
            st.line_chart(tmp.set_index("idx")[["rolling"]], color="#22c55e")

        with row1_r:
            st.markdown("#### Retrieval Score Over Time")
            if "avg_retrieval_score" in df.columns:
                tmp2 = df.copy()
                tmp2["idx"] = range(len(tmp2))
                st.line_chart(tmp2.set_index("idx")[["avg_retrieval_score"]], color="#3b82f6")

        row2_l, row2_r = st.columns(2, gap="large")

        with row2_l:
            st.markdown("#### Prompt Version Usage")
            if "prompt_version" in df.columns:
                pv = df["prompt_version"].value_counts().rename_axis("version").reset_index(name="count")
                st.bar_chart(pv.set_index("version"))

        with row2_r:
            st.markdown("#### Understanding by Prompt Version")
            if "prompt_version" in df.columns and "understood" in df.columns:
                grp = df.groupby("prompt_version")["understood"].mean().reset_index()
                grp.columns = ["version", "understood_rate"]
                st.bar_chart(grp.set_index("version"))

        st.divider()
        st.markdown("#### Recent Interactions (last 20)")
        show_cols = [c for c in ["timestamp", "query", "understood", "prediction",
                                  "avg_retrieval_score", "prompt_version"] if c in df.columns]
        disp = df[show_cols].tail(20).iloc[::-1].reset_index(drop=True)
        # Truncate long queries
        if "query" in disp.columns:
            disp["query"] = disp["query"].str[:60]
        st.dataframe(disp, use_container_width=True)

        st.divider()
        if st.button("📈  Generate Evidently Drift Report"):
            with st.spinner("Running drift analysis…"):
                path = b.generate_monitoring_report()
            if path.endswith(".html") and os.path.exists(path):
                with open(path) as f:
                    st.components.v1.html(f.read(), height=650, scrolling=True)
            else:
                st.warning(f"Could not generate report: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    if not st.session_state.ready:
        st.info("Initialize the pipeline first.")
        st.stop()

    b = st.session_state.backend
    st.markdown("### Understanding Predictor — ML Model")

    col_left, col_right = st.columns([1, 2], gap="large")

    # ── Left: controls ──────────────────────────────────────────────────────
    with col_left:
        st.markdown("#### Train / Retrain")
        n_syn = st.slider("Synthetic samples", 100, 1000, 400, 50)
        use_real = st.checkbox("Include real interaction data", value=True)

        if st.button("🔁  Generate & Train", use_container_width=True):
            with st.spinner("Generating data + training 3 models…"):
                df_syn = b.generate_synthetic_data(n_syn)
                if use_real:
                    df_real = b.load_interactions()
                    if not df_real.empty:
                        df_syn = pd.concat([df_syn, df_real], ignore_index=True)
                metrics = b.train_model(df_syn)
            st.success(f"Best model: **{metrics['best_model']}**")
            st.json({k: v for k, v in metrics.items()
                     if k not in ("classification_report", "cv_results")})

        st.divider()
        st.markdown("#### Manual Prediction Simulator")
        st.caption("Set feature values and see what the model predicts.")

        f_time  = st.slider("Time to read (s)",      0.0, 120.0, 25.0, 1.0)
        f_fu    = st.selectbox("Follow-up asked?",   ["No (0)", "Yes (1)"])
        f_sfu   = st.selectbox("Similar follow-up?", ["No (0)", "Yes (1)"])
        f_sqc   = st.slider("Session query count",   1, 15, 2)
        f_rl    = st.slider("Response length (chars)", 50, 900, 400)
        f_rs    = st.slider("Avg retrieval score",   0.0, 1.0, 0.75, 0.01)
        f_qc    = st.slider("Query complexity",      0.0, 1.0, 0.40, 0.01)

        if st.button("🔮  Predict Understanding", use_container_width=True):
            res = b.predict_understanding({
                "time_to_read":        f_time,
                "follow_up_asked":     1 if "Yes" in f_fu  else 0,
                "similar_followup":    1 if "Yes" in f_sfu else 0,
                "session_query_count": f_sqc,
                "response_length":     f_rl,
                "avg_retrieval_score": f_rs,
                "query_complexity":    f_qc,
            })
            css   = "understood-yes" if res["understood"] else "understood-no"
            label = "✓ UNDERSTOOD" if res["understood"] else "✗ NOT UNDERSTOOD"
            st.markdown(
                f'<div class="{css}">{label}<br>'
                f'probability = {res["probability"]:.4f}<br>'
                f'confidence  = {res["confidence"]:.4f}</div>',
                unsafe_allow_html=True,
            )

    # ── Right: metrics ──────────────────────────────────────────────────────
    with col_right:
        st.markdown("#### Current Model Metrics")
        m = b._ml_metrics
        if not m:
            st.info("Train the model to see metrics.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{m.get('accuracy', 0):.4f}")
            m2.metric("F1-Score", f"{m.get('f1_score', 0):.4f}")
            m3.metric("ROC-AUC",  f"{m.get('roc_auc',  0):.4f}")

            st.markdown("#### Cross-Validation ROC-AUC (5-fold) — All Candidates")
            cv = m.get("cv_results", {})
            if cv:
                cv_df = pd.DataFrame({"Model": list(cv.keys()), "CV ROC-AUC": list(cv.values())})
                st.dataframe(cv_df, use_container_width=True, hide_index=True)
                st.bar_chart(cv_df.set_index("Model"), color="#3b82f6")

            st.markdown("#### Classification Report")
            cr = m.get("classification_report", {})
            if cr:
                cr_df = pd.DataFrame(cr).T.round(3)
                st.dataframe(cr_df, use_container_width=True)

            st.markdown("#### Feature Rationale")
            st.markdown("""
| Feature | Behavioral signal it captures |
|---|---|
| `time_to_read` | Longer → more engaged → likely understood |
| `follow_up_asked` | Follow-up = curiosity OR confusion |
| `similar_followup` | Re-asking same thing → clear confusion |
| `session_query_count` | High count = struggling to get answers |
| `response_length` | Very short responses may lack context |
| `avg_retrieval_score` | Higher cosine sim = more relevant context |
| `query_complexity` | Complex questions need more care in evaluation |
""")

            trained_at = m.get("trained_at", "–")
            st.caption(f"Trained on {m.get('train_size','?')} samples | "
                       f"Tested on {m.get('test_size','?')} samples | "
                       f"At {trained_at[:19] if trained_at != '–' else '–'}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FEEDBACK LOG
# ══════════════════════════════════════════════════════════════════════════════
with tab_fb:
    if not st.session_state.ready:
        st.info("Initialize the pipeline first.")
        st.stop()

    b = st.session_state.backend
    st.markdown("### Feedback Loop Log")
    st.caption("Every system adaptation triggered by the feedback mechanism is recorded here.")

    fb_df = b.load_feedback_log()

    if fb_df.empty:
        st.markdown("""
        <div class="mono-card" style="line-height:2.2;">
        No adaptations yet.<br><br>
        The loop will trigger when:<br>
        &nbsp;&nbsp;• 3 consecutive <span style="color:#ef4444">not-understood</span>
          predictions → prompt switch<br>
        &nbsp;&nbsp;• Low retrieval score + confusion → retrieval quality flag<br>
        &nbsp;&nbsp;• Every 40 interactions → ML model retrain on real data<br>
        &nbsp;&nbsp;• 👎 explicit thumbs-down → counts toward streak
        </div>
        """, unsafe_allow_html=True)
    else:
        fa1, fa2, fa3, fa4 = st.columns(4)
        fa1.metric("Total Adaptations",   len(fb_df))
        fa2.metric("Prompt Switches",
                   len(fb_df[fb_df["action"].str.contains("prompt_switch", na=False)]))
        fa3.metric("Retrieval Flags",
                   len(fb_df[fb_df["action"].str.contains("retrieval", na=False)]))
        fa4.metric("ML Retrains",
                   len(fb_df[fb_df["action"].str.contains("retrain", na=False)]))

        st.divider()
        st.markdown("#### Adaptation History")
        st.dataframe(fb_df.iloc[::-1].reset_index(drop=True), use_container_width=True)

        st.divider()
        st.markdown("#### Action Type Distribution")
        ac = fb_df["action"].value_counts()
        st.bar_chart(ac, color="#f59e0b")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
with tab_arch:
    st.markdown("### System Architecture & Design")

    # Full pipeline flow
    st.markdown("""
    <div class="mono-card" style="font-size:0.77rem;line-height:2.1;">
    <b style="color:#dde1e8;letter-spacing:1px;">PIPELINE FLOW</b><br><br>
    User Query<br>
    &nbsp;&nbsp;→ [1] <b style="color:#3b82f6;">EMBED</b>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; sentence-transformers / all-MiniLM-L6-v2  (384-dim)<br>
    &nbsp;&nbsp;→ [2] <b style="color:#3b82f6;">RETRIEVE</b>
        &nbsp;&nbsp;&nbsp; Qdrant cosine similarity — top-5 chunks<br>
    &nbsp;&nbsp;→ [3] <b style="color:#3b82f6;">CONTEXT</b>
        &nbsp;&nbsp;&nbsp;&nbsp; Concatenate retrieved docs with separators<br>
    &nbsp;&nbsp;→ [4] <b style="color:#3b82f6;">GENERATE</b>
        &nbsp;&nbsp;&nbsp; Groq API (llama3-70b-8192) + active prompt version<br>
    &nbsp;&nbsp;→ [5] <b style="color:#22c55e;">FEATURES</b>
        &nbsp;&nbsp;&nbsp; Extract behavioral signals from interaction<br>
    &nbsp;&nbsp;→ [6] <b style="color:#22c55e;">ML PREDICT</b>
        &nbsp;&nbsp; Understanding probability (binary classifier)<br>
    &nbsp;&nbsp;→ [7] <b style="color:#22c55e;">EVALUATE</b>
        &nbsp;&nbsp;&nbsp;&nbsp; LLM-as-Judge scores (relevance/clarity/completeness)<br>
    &nbsp;&nbsp;→ [8] <b style="color:#f59e0b;">FEEDBACK</b>
        &nbsp;&nbsp;&nbsp;&nbsp; Streak check → prompt switch / retrieval flag / retrain<br>
    &nbsp;&nbsp;→ [9] <b style="color:#f59e0b;">LOG</b>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; SQLite interaction + feedback log<br>
    &nbsp;&nbsp;→ [10] <b style="color:#f59e0b;">MONITOR</b>
        &nbsp;&nbsp;&nbsp;&nbsp; Evidently drift report on demand<br>
    &nbsp;&nbsp;→ [11] <b style="color:#a78bfa;">IMPROVE</b>
        &nbsp;&nbsp;&nbsp;&nbsp; Next query uses updated prompt + retrained model
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("#### Component Stack")
        st.markdown("""
| Layer | Tool | Why chosen |
|---|---|---|
| LLM | Groq (Llama3-70b) | Free tier, ultra-fast inference |
| Vector DB | Qdrant (cloud or memory) | Native Python client, payload filtering |
| Embeddings | all-MiniLM-L6-v2 | Fast, 384-dim, free, strong quality |
| ML Model | Scikit-learn pipeline | LR / RF / GBM compared by CV AUC |
| Monitoring | Evidently AI | Open-source drift reports |
| Logging | SQLite | Zero-config, file-based |
| Frontend | Streamlit | Rapid iteration, built-in widgets |
| Secrets | st.secrets | Works locally and on Streamlit Cloud |
""")

    with col_b:
        st.markdown("#### Evaluation Metrics — Justification")
        st.markdown("""
**ML Model**
- **Accuracy** — overall correctness baseline
- **F1-Score** — handles class imbalance between understood/not
- **ROC-AUC** — threshold-independent discrimination measure
- **5-fold CV** — guards against overfitting on small datasets

**RAG / LLM**
- **Avg retrieval cosine score** — proxy for context relevance
- **LLM-as-Judge (1–5)** — relevance, clarity, completeness
- **Understanding rate** — ultimate user-outcome proxy metric

**Monitoring**
- **Evidently DataDriftPreset** — KS-test on all feature distributions
- **Prompt version trend** — tracks which version the feedback loop prefers
- **Streak counter** — real-time measure of consecutive confusion
""")

    st.divider()
    st.markdown("#### Feedback Loop Rules")
    st.markdown("""
| Trigger | Condition | System Action |
|---|---|---|
| Streak | 3 consecutive `not_understood` | Switch: v1_standard → v2_simplified → v3_detailed |
| Retrieval quality | avg_score < 0.55 & not understood | Log `retrieval_quality_flag` for analysis |
| Periodic retrain | Every 40 interactions | Retrain ML model on accumulated real data |
| Explicit 👎 | User clicks thumbs-down | Logs `understood=0`, increments streak |
| Explicit 👍 | User clicks thumbs-up | Logs `understood=1`, decrements streak |
""")

    st.divider()
    st.markdown("#### AWS Deployment Architecture (Production)")
    st.code("""
Internet
    │
    ├── Streamlit Community Cloud  ←  Frontend (free forever)
    │         │
    │         ↓  REST calls
    ├── AWS API Gateway  →  EC2 t2.micro (Ubuntu 22.04)
    │                            ├── FastAPI :8000  (RAG + ML inference)
    │                            └── boto3  →  S3 Bucket
    │                                           ├── /models   (ML artifacts)
    │                                           ├── /logs     (interaction CSVs)
    │                                           └── /reports  (Evidently HTML)
    │
    ├── Qdrant Cloud  (free 1 GB cluster)
    └── Groq API      (free tier)

Service justifications:
  EC2 t2.micro  — persistent server needed for FastAPI (Lambda too cold-start heavy)
  S3            — durable object store for model artifacts and logs
  API Gateway   — clean separation, SSL termination, rate limiting
  Streamlit Cloud — zero-config frontend deployment, CI/CD via GitHub push
""", language="text")
