"""Gradio demo: type a Malay (or mixed Malay-English) sentence, get its sentiment.

Usage:  python app.py
Set MODEL_PATH to a local folder or a Hugging Face model id (default: models/malay-bert).
"""
import os

import gradio as gr
from transformers import pipeline

from prepare import clean

MODEL_PATH = os.environ.get("MODEL_PATH", "models/malay-bert")
classifier = pipeline("text-classification", model=MODEL_PATH, top_k=None)

EXAMPLES = [
    "Alhamdulillah, servis kat kedai ni laju gila, staff pun peramah!",
    "Dah 2 jam tunggu LRT rosak lagi. Penat betul la macam ni.",
    "Esok mesyuarat pukul 10 pagi kat bilik 3.",
    "Makanan sedap tapi harga mahal sikit la for the portion.",
]


def predict(text):
    if not text.strip():
        return {}
    return {r["label"]: r["score"] for r in classifier(clean(text))[0]}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Tweet or sentence (Malay, or Malay mixed with English)", lines=3),
    outputs=gr.Label(label="Sentiment", num_top_classes=3),
    examples=EXAMPLES,
    title="Malay Tweet Sentiment",
    description="A Malaysian BERT model fine-tuned on 1,700 human-labelled Malaysian tweets. "
                "Predicts Negative, Neutral or Positive. Trained on casual Twitter Malay, so formal or "
                "very long text may be less accurate.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
