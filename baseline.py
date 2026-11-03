"""Baseline: TF-IDF (word + character n-grams) and logistic regression.

Character n-grams help with Malay slang and spelling variants ("tak", "x", "takde").
The regularisation strength C is chosen on the validation split; the test split is scored once.

Usage:  python baseline.py
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline, make_union

ROOT = Path(__file__).parent


def model(c):
    features = make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                          TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True))
    return make_pipeline(features, LogisticRegression(C=c, max_iter=2000, class_weight="balanced"))


def main():
    d = pd.read_csv(ROOT / "data" / "splits.csv")
    train, val, test = (d[d.split == s] for s in ("train", "val", "test"))

    scores = {c: f1_score(val.label, model(c).fit(train.text, train.label).predict(val.text), average="macro")
              for c in (0.1, 0.3, 1, 3, 10, 30)}
    best_c = max(scores, key=scores.get)

    final = model(best_c).fit(pd.concat([train.text, val.text]), pd.concat([train.label, val.label]))
    pred = final.predict(test.text)
    result = {"model": "TF-IDF + logistic regression", "params": {"C": best_c},
              "val_macro_f1": scores[best_c], "test_macro_f1": f1_score(test.label, pred, average="macro"),
              "test_accuracy": accuracy_score(test.label, pred)}
    (ROOT / "output").mkdir(exist_ok=True)
    (ROOT / "output" / "baseline.json").write_text(json.dumps(result, indent=2))
    print("Validation macro-F1 by C:", {c: round(s, 3) for c, s in scores.items()})
    print(classification_report(test.label, pred, digits=3))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
