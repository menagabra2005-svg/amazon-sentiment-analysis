import re

import pickle
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered",
)

# ── Load model & vectoriser ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load and cache the trained model and TF-IDF vectoriser from disk."""
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_artifacts()

# ── Text cleaning (must match the notebook preprocessing) ──────────────────────
def clean_text(text: str) -> str:
    """Lowercase and remove non-alphabetic characters from input text."""
    text = text.lower()
    text = re.sub(r"[^a-z]", " ", text)
    return text

# ── Prediction ─────────────────────────────────────────────────────────────────
def predict_sentiment(text: str) -> tuple[str, float]:
    """
    Predict the sentiment of a review string.

    Parameters
    ----------
    text : str
        Raw user input.

    Returns
    -------
    label : str
        'Positive' or 'Negative'.
    confidence : float
        Model confidence score (0–100 %).
    """
    cleaned    = clean_text(text)
    vector     = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vector)[0]
    confidence = model.predict_proba(vector)[0].max() * 100
    label      = "Positive" if prediction == 1 else "Negative"
    return label, confidence

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("💬 Amazon Review Sentiment Analyser")
st.markdown(
    "Enter any customer review below and the model will classify it as "
    "**Positive** or **Negative** with a confidence score."
)
st.divider()

user_input = st.text_area(
    label="Customer Review",
    placeholder="e.g. The product quality was excellent and delivery was fast!",
    height=150,
)

if st.button("Analyse Sentiment", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter some text before clicking Analyse.")
    else:
        label, confidence = predict_sentiment(user_input)

        if label == "Positive":
            st.success(f"✅ Sentiment: **{label}**")
        else:
            st.error(f"❌ Sentiment: **{label}**")

        st.metric(label="Model Confidence", value=f"{confidence:.1f} %")

st.divider()
st.caption("Model: Logistic Regression · Features: TF-IDF (5 000 terms) · Trained on Amazon Reviews")
