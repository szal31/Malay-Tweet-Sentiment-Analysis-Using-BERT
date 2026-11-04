"""Streamlit demo: type a Malay (or mixed Malay-English) sentence, get its sentiment.

Usage:  streamlit run demo/streamlit_app.py
Loads the fine-tuned model from the Hugging Face Hub (override with MODEL_PATH).
"""
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from transformers import pipeline

sys.path.append(str(Path(__file__).parent.parent))
from prepare import LABELS, clean  # noqa: E402  same cleaning as training

MODEL_PATH = os.environ.get("MODEL_PATH", "szal31/malay-tweet-sentiment-bert")
REPO = "https://github.com/szal31/Malay-Tweet-Sentiment-Analysis-Using-BERT"
EXAMPLES = [
    "Alhamdulillah, servis kat kedai ni laju gila, staff pun peramah!",
    "Dah 2 jam tunggu LRT rosak lagi. Penat betul la macam ni.",
    "Esok mesyuarat pukul 10 pagi kat bilik 3.",
    "Makanan sedap tapi harga mahal sikit la for the portion.",
]
# colour always paired with the label text, so it's never the only signal
STYLE = {"Negative": ("red", ":material/sentiment_dissatisfied:", "#DC2626"),
         "Neutral": ("gray", ":material/sentiment_neutral:", "#64748B"),
         "Positive": ("green", ":material/sentiment_satisfied:", "#16A34A")}

st.set_page_config(page_title="Malay Tweet Sentiment", page_icon=":material/sentiment_satisfied:")


@st.cache_resource(show_spinner="Loading the model (first visit takes about a minute)...")
def classifier():
    return pipeline("text-classification", model=MODEL_PATH, top_k=None)


def predict(text):
    return {r["label"]: r["score"] for r in classifier()(clean(text))[0]}


def use_example():
    if st.session_state.example:
        st.session_state.text = st.session_state.example


st.caption(":material/model_training: Fine-tuned Malaysian BERT · 1,397 human-labelled Malaysian tweets")
st.title("Malay tweet sentiment")
st.markdown("Type a tweet in **Malay or Manglish** and the model tells you whether it is **negative, neutral or "
            "positive**, and how sure it is.")

st.pills("Try an example", EXAMPLES, key="example", on_change=use_example)
text = st.text_area("Your text", key="text", height=110, max_chars=500,
                    placeholder="e.g. Best gila nasi lemak kat sini, mesti datang lagi!")

if text.strip():
    scores = predict(text)
    label = max(scores, key=scores.get)
    colour, icon, _ = STYLE[label]
    with st.container(border=True):
        st.markdown(f"### :{colour}[{icon} {label}]")
        st.markdown(f"The model is **{scores[label]:.0%}** sure.")
        chart = pd.DataFrame({"Sentiment": LABELS, "Probability": [scores[n] for n in LABELS]})
        bars = alt.Chart(chart).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("Sentiment:N", sort=LABELS, title=None),
            x=alt.X("Probability:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.Color("Sentiment:N", scale=alt.Scale(domain=LABELS, range=[STYLE[n][2] for n in LABELS]),
                            legend=None),
            tooltip=["Sentiment", alt.Tooltip("Probability:Q", format=".0%")])
        labels = bars.mark_text(align="left", dx=4).encode(text=alt.Text("Probability:Q", format=".0%"))
        st.altair_chart((bars + labels).properties(height=120), width="stretch")
        if scores[label] < 0.5:
            st.caption("Low confidence: the sentence may mix feelings, or use words the model rarely saw.")
else:
    st.info("Pick an example above, or type your own sentence.", icon=":material/edit_note:")

with st.expander("About the model"):
    m1, m2, m3 = st.columns(3)
    m1.metric("Test macro-F1", "0.63", "+0.05 vs baseline", delta_color="normal", border=True)
    m2.metric("Test accuracy", "66%", border=True)
    m3.metric("Test tweets", "300", border=True)
    st.markdown("Fine-tuned from `mesolitica/bert-base-standard-bahasa-cased` on CPU. It is trained on casual "
                "Twitter Malay, so formal or very long text may be less accurate, and sarcasm often fools it.")

st.caption(f"[Code, results and error analysis ↗]({REPO}) · "
           "[Model on Hugging Face ↗](https://huggingface.co/szal31/malay-tweet-sentiment-bert)")
