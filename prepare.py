"""Clean the Malaysian tweet sentiment data and make one fixed train / validation / test split.

Every model (baseline and transformers) uses the same split, so their scores are comparable.
Data: kornwtp/ms-bahasa-twitter-sentiment on Hugging Face (Apache-2.0), 2,008 human-labelled tweets.

Usage:  python prepare.py
"""
import html
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
LABELS = ["Negative", "Neutral", "Positive"]


def clean(text):
    """Same cleaning at training and prediction time: decode &amp; etc., mask links and usernames."""
    text = html.unescape(text)
    text = re.sub(r"https?://\S+", "URL", text)
    text = re.sub(r"@\w+", "@USER", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    d = pd.read_parquet(ROOT / "data" / "tweets.parquet").dropna(subset=["labels"])
    d = d.rename(columns={"texts": "text", "labels": "label"})
    d["text"] = d["text"].map(clean)
    d = d[d.text.str.len() > 0]
    # masking links/usernames creates exact duplicates; drop texts whose copies disagree, keep one of the rest
    d = d[d.groupby("text").label.transform("nunique") == 1].drop_duplicates("text")

    # 70 / 15 / 15, stratified so every split keeps the same label mix
    train, rest = train_test_split(d, test_size=0.30, stratify=d.label, random_state=42)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest.label, random_state=42)
    out = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])

    # Self-checks: known labels only, no text in two splits.
    assert set(out.label) == set(LABELS)
    assert not out.duplicated("text").any()
    out[["text", "label", "split"]].to_csv(ROOT / "data" / "splits.csv", index=False)
    print(out.groupby(["split", "label"]).size().unstack().loc[["train", "val", "test"]])


if __name__ == "__main__":
    main()
