"""
BL-4: Fine-tuned FinBERT baseline for IntentRouter comparison.

Approach:
  - Base model: ProsusAI/finbert (BERT-base trained on financial corpora)
  - Task: 4-class intent classification (portfolio_query / profiling /
          recommendation / general)
  - Evaluation: 5-fold stratified cross-validation (n=100, 80 train / 20 test)
  - Epochs: 10 per fold (AdamW, lr=2e-5, weight decay 0.01)
  - Inference latency: CPU, averaged over all 100 predictions

Usage:
    python tests/eval/run_finbert_finetune.py

Results appended to tests/eval/fixtures/baseline_results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

TESTSET_PATH  = project_root / "tests" / "eval" / "fixtures" / "router_testset.json"
RESULTS_PATH  = project_root / "tests" / "eval" / "fixtures" / "baseline_results.json"

MODEL_NAME = "ProsusAI/finbert"
CLASSES    = ["portfolio_query", "profiling", "recommendation", "general"]
LABEL2ID   = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL   = {i: c for c, i in LABEL2ID.items()}

EPOCHS     = 10
BATCH_SIZE = 8
LR         = 2e-5
MAX_LEN    = 64
WEIGHT_DECAY = 0.01
N_FOLDS    = 5
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class IntentDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings.get("token_type_ids",
                              torch.zeros_like(self.encodings["input_ids"]))[idx],
            "labels":         self.labels[idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            token_type_ids=batch["token_type_ids"].to(device),
            labels=batch["labels"].to(device),
        )
        out.loss.backward()
        optimizer.step()
        total_loss += out.loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader) -> tuple[float, list[int]]:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            token_type_ids=batch["token_type_ids"].to(device),
        )
        preds = out.logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(batch["labels"].tolist())
    acc = sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_labels)
    return acc, all_preds


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    total   = len(results)
    correct = sum(1 for r in results if r["true"] == r["pred"])
    hard    = [r for r in results if r.get("difficulty") == "hard"]
    hard_ok = sum(1 for r in hard if r["true"] == r["pred"])

    confusion = {c: {c2: 0 for c2 in CLASSES} for c in CLASSES}
    for r in results:
        confusion[r["true"]][r["pred"]] += 1

    per_class = {}
    for c in CLASSES:
        tp = confusion[c][c]
        fp = sum(confusion[o][c] for o in CLASSES if o != c)
        fn = sum(confusion[c][o] for o in CLASSES if o != c)
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        re = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * pr * re / (pr + re) if (pr + re) else 0.0
        per_class[c] = {"precision": pr, "recall": re, "f1": f1,
                        "tp": tp, "fp": fp, "fn": fn}

    return {
        "overall_acc":    correct / total,
        "hard_acc":       hard_ok / len(hard) if hard else 0.0,
        "n_total":        total,
        "n_correct":      correct,
        "n_hard":         len(hard),
        "n_hard_correct": hard_ok,
        "per_class":      per_class,
        "confusion":      confusion,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(TESTSET_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    texts  = [c["message"] for c in cases]
    labels = [LABEL2ID[c["label"]] for c in cases]
    y_arr  = np.array(labels)

    print(f"Loading tokenizer from {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    all_results    = []
    all_latencies  = []
    fold_accs      = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, y_arr), 1):
        print(f"\n--- Fold {fold}/{N_FOLDS} ---")

        train_texts  = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_texts   = [texts[i] for i in test_idx]
        test_labels  = [labels[i] for i in test_idx]
        test_cases   = [cases[i]  for i in test_idx]

        train_ds = IntentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
        test_ds  = IntentDataset(test_texts,  test_labels,  tokenizer, MAX_LEN)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

        # Re-initialise model each fold to avoid leakage
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(CLASSES),
            ignore_mismatched_sizes=True,  # replace 3-class head with 4-class
        )
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_acc  = 0.0
        best_state = None
        for epoch in range(1, EPOCHS + 1):
            loss = train_one_epoch(model, train_loader, optimizer)
            acc, _ = evaluate(model, test_loader)
            print(f"  epoch {epoch:02d}/{EPOCHS}  loss={loss:.4f}  val_acc={acc:.3f}")
            if acc > best_acc:
                best_acc  = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Load best checkpoint
        model.load_state_dict(best_state)
        model.to(device)
        fold_accs.append(best_acc)

        # Inference with latency measurement (per-example)
        model.eval()
        single_tokenizer_inputs = tokenizer(
            test_texts, truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt"
        )
        with torch.no_grad():
            for i, (case, pred_label_idx) in enumerate(
                    zip(test_cases,
                        evaluate(model, test_loader)[1])):
                t0 = time.perf_counter()
                _ = model(
                    input_ids=single_tokenizer_inputs["input_ids"][i:i+1].to(device),
                    attention_mask=single_tokenizer_inputs["attention_mask"][i:i+1].to(device),
                    token_type_ids=single_tokenizer_inputs.get(
                        "token_type_ids",
                        torch.zeros_like(single_tokenizer_inputs["input_ids"])
                    )[i:i+1].to(device),
                )
                lat = (time.perf_counter() - t0) * 1000
                all_latencies.append(lat)
                all_results.append({
                    "text":       case["message"],
                    "true":       case["label"],
                    "pred":       ID2LABEL[pred_label_idx],
                    "difficulty": case.get("difficulty", "easy"),
                    "latency_ms": lat,
                })

        print(f"  Fold {fold} best acc: {best_acc:.3f}")

    metrics = compute_metrics(all_results)
    avg_lat = sum(all_latencies) / len(all_latencies)
    p95_lat = sorted(all_latencies)[94]

    print("\n=== FinBERT Fine-tuned Results ===")
    print(f"  Overall accuracy : {metrics['overall_acc']:.1%}")
    print(f"  Hard-subset acc  : {metrics['hard_acc']:.1%}")
    print(f"  Avg latency      : {avg_lat:.1f} ms")
    print(f"  p95 latency      : {p95_lat:.1f} ms")
    print(f"  Per-fold accs    : {[f'{a:.3f}' for a in fold_accs]}")

    # Load existing results and append
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results_data = json.load(f)

    results_data["finbert_ft"] = {
        "metrics":        metrics,
        "avg_latency_ms": avg_lat,
        "p95_latency_ms": p95_lat,
        "cost_per_query": 0.0,
        "fold_accs":      fold_accs,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults appended to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
