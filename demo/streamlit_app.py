"""Streamlit demo: type a Malay (or mixed Malay-English) sentence, get its sentiment.

Usage:  streamlit run demo/streamlit_app.py
Loads the fine-tuned model from the Hugging Face Hub (override with MODEL_PATH).
"""
import os
import sys
from pathlib import Path

import streamlit as st
from transformers import pipeline

sys.path.append(str(Path(__file__).parent.parent))
from prepare import LABELS, clean  # noqa: E402  same cleaning as training

MODEL_PATH = os.environ.get("MODEL_PATH", "szal31/malay-tweet-sentiment-bert")
EXAMPLES = [
    "Alhamdulillah, servis kat kedai ni laju gila, staff pun peramah!",
    "Dah 2 jam tunggu LRT rosak lagi. Penat betul la macam ni.",
    "Esok mesyuarat pukul 10 pagi kat bilik 3.",
    "Makanan sedap tapi harga mahal sikit la for the portion.",
]


@st.cache_resource(show_spinner="Loading the model (first visit takes about a minute)...")
def classifier():
    return pipeline("text-classification", model=MODEL_PATH, top_k=None)


def predict(text):
    return {r["label"]: r["score"] for r in classifier()(clean(text))[0]}


st.set_page_config(page_title="Malay Tweet Sentiment", page_icon="💬")
st.title("Malay Tweet Sentiment")
st.caption("A Malaysian BERT model fine-tuned on 1,397 human-labelled Malaysian tweets. "
           "Trained on casual Twitter Malay, so formal or very long text may be less accurate.")

st.write("Try an example:")
for i, example in enumerate(EXAMPLES):
    if st.button(example, key=f"ex{i}"):
        st.session_state.text = example
text = st.text_area("Tweet or sentence (Malay, or Malay mixed with English)", key="text", height=100)

if text.strip():
    scores = predict(text)
    label = max(scores, key=scores.get)
    st.subheader(f"{label} ({scores[label]:.0%})")
    for name in LABELS:
        st.progress(scores[name], text=f"{name}: {scores[name]:.0%}")

st.markdown("[Code, results and error analysis on GitHub]"
            "(https://github.com/szal31/Malay-Tweet-Sentiment-Analysis-Using-BERT)")
