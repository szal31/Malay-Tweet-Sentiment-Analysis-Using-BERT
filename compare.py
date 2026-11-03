"""Compare the three models on the test split and look at where the best one goes wrong.

Writes output/1_model_comparison.png, output/2_confusion_matrix.png and output/errors.csv.

Usage:  python compare.py      (after baseline.py and finetune.py)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

from prepare import LABELS

ROOT = Path(__file__).parent
OUT = ROOT / "output"
RUNS = {"baseline": "TF-IDF + logistic regression", "malay-bert": "Malaysian BERT (fine-tuned)",
        "xlm-roberta": "XLM-RoBERTa (fine-tuned)"}
INK, MUTED, SURFACE, BLUE, PALE = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#b9cde8"


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def main():
    results = {k: json.loads((OUT / f"{k}.json").read_text()) for k in RUNS if (OUT / f"{k}.json").exists()}
    table = pd.DataFrame({RUNS[k]: {"Validation macro-F1": r["val_macro_f1"], "Test macro-F1": r["test_macro_f1"],
                                    "Test accuracy": r["test_accuracy"]} for k, r in results.items()}).T
    best = max((k for k in results if k != "baseline"), key=lambda k: results[k]["val_macro_f1"])  # chosen on val

    fig, ax = plt.subplots(figsize=(8, 2.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax, "Test macro-F1 by model (300 held-out tweets, higher is better)")
    s = table["Test macro-F1"].iloc[::-1]
    bars = ax.barh(s.index, s.values, height=0.6, color=[BLUE if i == RUNS[best] else PALE for i in s.index])
    for b, v in zip(bars, s.values):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9, color=MUTED)
    ax.set_xlim(0, 1)
    ax.xaxis.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "1_model_comparison.png", facecolor=SURFACE)

    preds = pd.read_csv(OUT / f"test_predictions_{best}.csv")
    cm = confusion_matrix(preds.label, preds.pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax, f"Confusion matrix: {RUNS[best]}")
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Predicted", color=MUTED, fontsize=9)
    ax.set_ylabel("Actual", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "2_confusion_matrix.png", facecolor=SURFACE)

    errors = preds[preds.label != preds.pred]
    errors.to_csv(OUT / "errors.csv", index=False)
    # Self-check: the confusion matrix covers every test tweet.
    assert cm.sum() == len(preds)

    print(table.round(3).to_string(), "\n")
    print(f"Best on validation: {RUNS[best]}. Test errors: {len(errors)} of {len(preds)}")
    print("Most common confusions:")
    print(errors.groupby(["label", "pred"]).size().sort_values(ascending=False).head(4).to_string())


if __name__ == "__main__":
    main()
