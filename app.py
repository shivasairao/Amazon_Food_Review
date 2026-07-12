import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================================
# CONFIG
# =====================================================================
MODEL_PATH = "gru_sentiment_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 200

# Real numbers from the training run — update if you retrain.
MODEL_STATS = {
    "architecture": "Embedding(64) → GRU(64) → Dropout(0.3) → Dense(32) → Dense(1)",
    "dataset_size": "363,903 reviews",
    "positive_pct": 84,
    "negative_pct": 16,
    "vocab_size": "10,000 words",
    "seq_len": "200 tokens",
}

st.set_page_config(
    page_title="Sentiment Lab · Amazon Fine Food Reviews",
    page_icon="🌿",
    layout="wide",
)

# =====================================================================
# STYLE — "grocery-label" editorial design system
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --bg: #EEF1E6;
    --surface: #FBFBF6;
    --ink: #1E281F;
    --ink-soft: #55604F;
    --rule: #CBD3BE;
    --mustard: #C1841F;
    --mustard-deep: #9C6A16;
    --moss: #4C7A57;
    --moss-bg: #E4EEE3;
    --tomato: #B14430;
    --tomato-bg: #F4E3DE;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background: var(--bg); }
.block-container { max-width: 980px; padding-top: 2.5rem; }

/* ---------- Header ---------- */
.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--mustard-deep);
    border: 1px solid var(--mustard-deep);
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 100px;
    margin-bottom: 0.9rem;
}
.headline {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.6rem;
    line-height: 1.08;
    color: var(--ink);
    margin: 0 0 0.5rem 0;
}
.subhead {
    font-size: 1.02rem;
    color: var(--ink-soft);
    max-width: 46rem;
    margin-bottom: 2rem;
}

/* ---------- Cards ---------- */
.card {
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: 14px;
    padding: 1.6rem 1.7rem;
}
.card + .card { margin-top: 1.2rem; }
.card-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink-soft);
    margin-bottom: 0.9rem;
}

/* ---------- Inputs ---------- */
.stTextArea textarea {
    border-radius: 10px !important;
    border: 1.5px solid var(--rule) !important;
    background: var(--surface) !important;
    font-size: 1rem !important;
    color: var(--ink) !important;
}
.stTextArea textarea:focus {
    border-color: var(--mustard) !important;
    box-shadow: 0 0 0 1px var(--mustard) !important;
}

.stButton > button {
    background: var(--ink) !important;
    color: var(--surface) !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.4rem !important;
    transition: transform 0.12s ease, background 0.12s ease;
}
.stButton > button:hover { background: var(--mustard-deep) !important; transform: translateY(-1px); }

/* sample chips */
div[data-testid="stHorizontalBlock"] .stButton > button {
    background: var(--surface) !important;
    color: var(--ink-soft) !important;
    border: 1px solid var(--rule) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    padding: 0.35rem 0.8rem !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    border-color: var(--mustard) !important;
    color: var(--mustard-deep) !important;
    background: var(--surface) !important;
    transform: none;
}

/* ---------- Verdict badge ---------- */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    padding: 0.35rem 0.9rem;
    border-radius: 100px;
}
.badge.positive { background: var(--moss-bg); color: var(--moss); }
.badge.negative { background: var(--tomato-bg); color: var(--tomato); }

.confidence-number {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.4rem;
    color: var(--ink);
    line-height: 1;
}
.confidence-caption {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--ink-soft);
    letter-spacing: 0.05em;
}

/* ---------- Model Facts panel (nutrition-label pastiche) ---------- */
.facts-panel {
    background: var(--surface);
    border: 2px solid var(--ink);
    border-radius: 4px;
    padding: 1rem 1.1rem 1.2rem 1.1rem;
    font-family: 'IBM Plex Mono', monospace;
}
.facts-title {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 1.5rem;
    border-bottom: 8px solid var(--ink);
    padding-bottom: 0.3rem;
    margin-bottom: 0.4rem;
}
.facts-sub {
    font-size: 0.78rem;
    border-bottom: 1px solid var(--ink);
    padding-bottom: 0.4rem;
    margin-bottom: 0.4rem;
    color: var(--ink-soft);
}
.facts-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    padding: 0.28rem 0;
    border-bottom: 1px solid var(--rule);
}
.facts-row.big { font-weight: 600; font-size: 0.95rem; border-bottom: 4px solid var(--ink); }
.facts-row .val { color: var(--ink); font-weight: 600; }

/* misc */
hr.rule { border: none; border-top: 1px solid var(--rule); margin: 1.6rem 0; }
.footnote { font-size: 0.78rem; color: var(--ink-soft); }
[data-testid="stFileUploader"] section { background: var(--surface); border-radius: 10px; border: 1.5px dashed var(--rule); }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# MODEL LOADING + INFERENCE
# =====================================================================
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_sentiment(model, tokenizer, review_text):
    cleaned = clean_text(review_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob = float(model.predict(padded, verbose=0)[0][0])
    label = "Positive" if prob > 0.5 else "Negative"
    return label, prob


# =====================================================================
# VISUAL COMPONENTS
# =====================================================================
def gauge_svg(prob_positive: float) -> str:
    """Semicircular dial from Negative (left) to Positive (right)."""
    import math
    cx, cy, r = 110, 105, 88
    needle_angle_deg = 180 - (prob_positive * 180)  # 180=left(neg) .. 0=right(pos)
    rad = math.radians(needle_angle_deg)
    nx = cx + (r - 14) * math.cos(rad)
    ny = cy - (r - 14) * math.sin(rad)

    def arc_path(a0, a1, radius):
        r0, r1 = math.radians(a0), math.radians(a1)
        x0, y0 = cx + radius * math.cos(r0), cy - radius * math.sin(r0)
        x1, y1 = cx + radius * math.cos(r1), cy - radius * math.sin(r1)
        return f"M {x0:.1f} {y0:.1f} A {radius} {radius} 0 0 0 {x1:.1f} {y1:.1f}"

    track_w = 16
    seg_neg = arc_path(180, 120, r)
    seg_mid = arc_path(120, 60, r)
    seg_pos = arc_path(60, 0, r)

    return f"""
    <svg viewBox="0 0 220 130" width="220" height="130" xmlns="http://www.w3.org/2000/svg">
        <path d="{seg_neg}" fill="none" stroke="#B14430" stroke-width="{track_w}" stroke-linecap="round" opacity="0.85"/>
        <path d="{seg_mid}" fill="none" stroke="#C1841F" stroke-width="{track_w}" stroke-linecap="round" opacity="0.85"/>
        <path d="{seg_pos}" fill="none" stroke="#4C7A57" stroke-width="{track_w}" stroke-linecap="round" opacity="0.85"/>
        <circle cx="{cx}" cy="{cy}" r="6" fill="#1E281F"/>
        <line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#1E281F" stroke-width="3" stroke-linecap="round"/>
        <text x="18" y="122" font-family="IBM Plex Mono, monospace" font-size="10" fill="#55604F">NEG</text>
        <text x="188" y="122" font-family="IBM Plex Mono, monospace" font-size="10" fill="#55604F">POS</text>
    </svg>
    """


def render_facts_panel():
    st.markdown(f"""
    <div class="facts-panel">
        <div class="facts-title">Model Facts</div>
        <div class="facts-sub">Serving size: 1 review &nbsp;·&nbsp; Servings per dataset: {MODEL_STATS['dataset_size']}</div>
        <div class="facts-row big"><span>Test Accuracy</span><span class="val">92.2%</span></div>
        <div class="facts-row"><span>Negative-class recall</span><span class="val">92%</span></div>
        <div class="facts-row"><span>Class balance</span><span class="val">{MODEL_STATS['positive_pct']}% pos / {MODEL_STATS['negative_pct']}% neg</span></div>
        <div class="facts-row"><span>Vocabulary</span><span class="val">{MODEL_STATS['vocab_size']}</span></div>
        <div class="facts-row"><span>Max sequence length</span><span class="val">{MODEL_STATS['seq_len']}</span></div>
        <div class="facts-row"><span>Architecture</span><span class="val">GRU</span></div>
        <p class="footnote" style="margin-top:0.6rem;">
        Not a % Daily Value. Trained on Amazon Fine Food Reviews; Score ≥ 4 → Positive, Score ≤ 2 → Negative, Score = 3 discarded.
        </p>
    </div>
    """, unsafe_allow_html=True)


SAMPLE_REVIEWS = {
    "🍯 Glowing": "This honey is unbelievable — rich, floral, and my kids ask for it on everything now. Reordering a case immediately.",
    "🥫 Disappointed": "Arrived dented and the seal was broken. Tasted metallic and I ended up throwing half the jar away.",
    "🫐 Mixed": "Flavor is fine but nothing special for the price, and shipping took almost two weeks.",
}


# =====================================================================
# LAYOUT
# =====================================================================
st.markdown('<span class="eyebrow">Sentiment Lab · GRU Model</span>', unsafe_allow_html=True)
st.markdown('<div class="headline">Read the room before<br>the review goes live.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subhead">A recurrent neural network trained on 364k Amazon Fine Food Reviews '
    'reads a review the way a customer would — catching tone, not just keywords.</div>',
    unsafe_allow_html=True,
)

try:
    model, tokenizer = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    st.markdown(f"""
    <div class="card">
        <div class="card-label">Model not found</div>
        <p style="color: var(--ink-soft); margin:0;">
        Couldn't load <code>{MODEL_PATH}</code> / <code>{TOKENIZER_PATH}</code>. Run <code>python gru.py</code>
        on <code>Reviews.csv</code> first to produce these files, then restart the app.
        </p>
        <p class="footnote" style="margin-top:0.6rem;">Details: {e}</p>
    </div>
    """, unsafe_allow_html=True)

main_col, side_col = st.columns([2, 1], gap="large")

with main_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Try a sample, or paste your own</div>', unsafe_allow_html=True)

    chip_cols = st.columns(len(SAMPLE_REVIEWS))
    for col, (label, text) in zip(chip_cols, SAMPLE_REVIEWS.items()):
        if col.button(label, key=f"chip_{label}"):
            st.session_state["review_input"] = text

    review = st.text_area(
        "Review text",
        height=140,
        placeholder="Type or paste a product review here...",
        key="review_input",
        label_visibility="collapsed",
    )
    analyze = st.button("Analyze review", disabled=not artifacts_loaded)
    st.markdown('</div>', unsafe_allow_html=True)

    if analyze:
        if not review or not review.strip():
            st.warning("Please enter a review first.")
        else:
            label, prob = predict_sentiment(model, tokenizer, review)
            confidence = prob if label == "Positive" else 1 - prob
            badge_class = "positive" if label == "Positive" else "negative"
            dot = "●"

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Verdict</div>', unsafe_allow_html=True)
            result_col1, result_col2 = st.columns([1, 1.3])
            with result_col1:
                st.markdown(gauge_svg(prob), unsafe_allow_html=True)
            with result_col2:
                st.markdown(f'<span class="badge {badge_class}">{dot} {label}</span>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence-number" style="margin-top:0.6rem;">{confidence:.1%}</div>', unsafe_allow_html=True)
                st.markdown('<div class="confidence-caption">MODEL CONFIDENCE</div>', unsafe_allow_html=True)
                st.markdown(f'<p class="footnote" style="margin-top:0.8rem;">Raw output P(positive) = {prob:.4f}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

with side_col:
    render_facts_panel()

st.markdown('<hr class="rule">', unsafe_allow_html=True)

# ---------------- Batch prediction ----------------
st.markdown('<div class="card-label" style="margin-bottom:0.6rem;">Batch prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.write("Upload a CSV with a `Text` column to score a whole case lot of reviews at once.")
uploaded = st.file_uploader("Choose a CSV file", type="csv", label_visibility="collapsed")

if uploaded is not None and artifacts_loaded:
    batch_df = pd.read_csv(uploaded)
    if "Text" not in batch_df.columns:
        st.error("CSV must contain a 'Text' column.")
    else:
        with st.spinner("Scoring reviews..."):
            cleaned = batch_df["Text"].apply(clean_text)
            seqs = tokenizer.texts_to_sequences(cleaned)
            padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
            probs = model.predict(padded, verbose=0).reshape(-1)
            batch_df["sentiment"] = np.where(probs > 0.5, "Positive", "Negative")
            batch_df["confidence"] = np.where(probs > 0.5, probs, 1 - probs)

        pos_n = int((batch_df["sentiment"] == "Positive").sum())
        neg_n = int((batch_df["sentiment"] == "Negative").sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Reviews scored", len(batch_df))
        m2.metric("Positive", f"{pos_n} ({pos_n/len(batch_df):.0%})")
        m3.metric("Negative", f"{neg_n} ({neg_n/len(batch_df):.0%})")

        st.dataframe(
            batch_df[["Text", "sentiment", "confidence"]].head(200),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download full results as CSV",
            batch_df.to_csv(index=False).encode("utf-8"),
            "sentiment_predictions.csv",
            "text/csv",
        )
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<p class="footnote" style="margin-top:1.5rem;">'
    'Model: GRU · Embedding(64) → GRU(64) → Dropout(0.3) → Dense(32) → Dense(1, sigmoid). '
    'Trained on Amazon Fine Food Reviews. Not affiliated with Amazon.'
    '</p>',
    unsafe_allow_html=True,
)