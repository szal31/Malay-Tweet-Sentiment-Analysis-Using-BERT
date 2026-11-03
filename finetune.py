"""Fine-tune a pretrained transformer for 3-class Malay tweet sentiment, on CPU.

Plain PyTorch loop: AdamW, linear warmup/decay, early stopping on validation macro-F1.
The best epoch's weights are scored once on the test split and saved to models/<name>/.

Usage:  python finetune.py malay-bert      (mesolitica/bert-base-standard-bahasa-cased)
        python finetune.py xlm-roberta    (FacebookAI/xlm-roberta-base)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from prepare import LABELS

ROOT = Path(__file__).parent
MODELS = {"malay-bert": "mesolitica/bert-base-standard-bahasa-cased", "xlm-roberta": "FacebookAI/xlm-roberta-base"}
EPOCHS, PATIENCE, BATCH, LR, MAX_LEN = 5, 2, 8, 2e-5, 96  # sized for a 12 GB laptop without a GPU


def batches(tokenizer, frame, shuffle):
    ids = frame.label.map(LABELS.index).tolist()
    pairs = list(zip(frame.text.tolist(), ids))

    def collate(chunk):
        texts, labels = zip(*chunk)
        enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels)
        return enc
    return DataLoader(pairs, batch_size=BATCH, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    return np.concatenate([model(**{k: v for k, v in b.items() if k != "labels"}).logits.argmax(-1).numpy()
                           for b in loader])


def main(name):
    torch.manual_seed(42)
    d = pd.read_csv(ROOT / "data" / "splits.csv")
    train, val, test = (d[d.split == s] for s in ("train", "val", "test"))
    tokenizer = AutoTokenizer.from_pretrained(MODELS[name])
    model = AutoModelForSequenceClassification.from_pretrained(
        MODELS[name], num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)})

    # Freeze the word-embedding table: XLM-R's 250k-word table alone needs ~3 GB to train, and with only
    # 1,400 tweets it would barely change anyway. The transformer layers and classifier still learn.
    model.get_input_embeddings().weight.requires_grad_(False)

    train_dl, val_dl, test_dl = batches(tokenizer, train, True), batches(tokenizer, val, False), batches(tokenizer, test, False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    steps = EPOCHS * len(train_dl)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    y_val = val.label.map(LABELS.index).to_numpy()

    save_dir = ROOT / "models" / name
    best_f1, history, waited, start = -1.0, [], 0, time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for b in train_dl:
            loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(), scheduler.step(), optimizer.zero_grad()
        f1 = f1_score(y_val, predict(model, val_dl), average="macro")
        history.append(round(float(f1), 4))
        print(f"epoch {epoch}: val macro-F1 {f1:.3f}  ({(time.time() - start) / 60:.1f} min)", flush=True)
        if f1 > best_f1:
            best_f1, waited = f1, 0
            model.save_pretrained(save_dir)  # best epoch kept on disk, not as a second copy in RAM
            tokenizer.save_pretrained(save_dir)
        else:
            waited += 1
            if waited >= PATIENCE:  # early stopping: validation stopped improving
                break

    del model, optimizer
    model = AutoModelForSequenceClassification.from_pretrained(save_dir)
    y_test = test.label.map(LABELS.index).to_numpy()
    pred = predict(model, test_dl)
    result = {"model": MODELS[name], "params": {"epochs_run": len(history), "best_epoch": int(np.argmax(history)) + 1,
                                                "lr": LR, "batch": BATCH, "max_len": MAX_LEN,
                                                "frozen_embeddings": True},
              "val_macro_f1_by_epoch": history, "val_macro_f1": float(best_f1),
              "test_macro_f1": float(f1_score(y_test, pred, average="macro")),
              "test_accuracy": float(accuracy_score(y_test, pred)), "train_minutes": round((time.time() - start) / 60, 1)}
    pd.DataFrame({"text": test.text.values, "label": test.label.values, "pred": [LABELS[i] for i in pred]}) \
        .to_csv(ROOT / "output" / f"test_predictions_{name}.csv", index=False)
    (ROOT / "output" / f"{name}.json").write_text(json.dumps(result, indent=2))
    print(classification_report(y_test, pred, target_names=LABELS, digits=3))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    assert len(sys.argv) == 2 and sys.argv[1] in MODELS, f"usage: python finetune.py {{{'|'.join(MODELS)}}}"
    main(sys.argv[1])
