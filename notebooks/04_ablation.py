# =============================================================
# Notebook 04: Ablation Studies & Hyperparameter Sweeps
# MedPrompt MLOps Project — CPU / Local version
#
# EXPERIMENTS:
#   A. Hidden dropout sweep  : 0.1, 0.2, 0.3, 0.4
#   B. Training data size    : 25%, 50%, 75%, 100%
#   C. Learning rate sweep   : 1e-5, 3e-5, 5e-5, 1e-4
#   D. Max sequence length   : 64, 128, 256
#
# HOW TO RUN:
#   python notebooks/04_ablation.py
#
# OUTPUT:
#   MLflow experiment: medprompt_ablation
#   ⚠️  Screenshot the comparison charts — worth 20% of grade!
# =============================================================

import sys, os, json, time
sys.path.insert(0, ".")

import torch
import mlflow
import numpy as np
import pandas as pd

from src.model   import MedPromptModel, load_tokenizer
from src.utils   import build_dataloader, load_label_map, load_class_weights
from src.metrics import compute_classification_metrics, compute_rouge, log_metrics_to_mlflow

# ── Shared setup ──────────────────────────────────────────────
DEVICE       = "cpu"
LABEL_MAP    = "./outputs/label_map.json"
WEIGHTS_PATH = "./outputs/class_weights.json"
MODEL_ID     = "distilbert-base-uncased"
ABLATION_STEPS = 80   # steps per run (truncated for speed)

label_map     = load_label_map(LABEL_MAP)
class_weights = load_class_weights(WEIGHTS_PATH, device=DEVICE)
id_to_label   = {int(k): v for k, v in label_map["id_to_specialty"].items()}
num_classes   = len(label_map["specialty_to_id"])
class_names   = [id_to_label[i] for i in sorted(id_to_label.keys())]

df_train = pd.read_csv("./outputs/df_train.csv")
df_val   = pd.read_csv("./outputs/df_val.csv")
tokenizer= load_tokenizer(MODEL_ID)

mlflow.set_tracking_uri("sqlite:///outputs/mlflow.db")
mlflow.set_experiment("medprompt_ablation")

def quick_train_eval(model, train_dl, val_dl, lr=3e-5, steps=ABLATION_STEPS):
    """Run a short training loop and return val metrics."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for step, batch in enumerate(train_dl):
        if step >= steps:
            break
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        cls  = batch["cls_labels"].to(DEVICE)
        tok  = batch["token_labels"].to(DEVICE)
        opt.zero_grad()
        out  = model(ids, mask, cls_labels=cls, token_labels=tok)
        out["total_loss"].backward()
        opt.step()
        if step % 20 == 0:
            mlflow.log_metric("train/loss", out["total_loss"].item(), step=step)

    model.eval()
    preds, true = [], []
    sums, refs  = [], []
    with torch.no_grad():
        for batch in val_dl:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            out  = model(ids, mask)
            preds.extend(out["logits"].argmax(-1).cpu().tolist())
            true.extend(batch["cls_labels"].tolist())
            if len(sums) < 16:
                sums.extend(model.extract_summary(ids, mask, tokenizer))
                refs.extend(batch["reference_summary"])

    cls_m  = compute_classification_metrics(preds, true, class_names)
    rouge  = compute_rouge(sums, refs)
    return {**cls_m, **rouge}

# ══════════════════════════════════════════════════════════════
# EXPERIMENT A: Dropout sweep
# WHY: MTSamples is small (~3,300 train rows). Dropout is the
#      main regulariser. We find the optimal dropout rate.
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("EXPERIMENT A: Dropout Sweep  [0.1, 0.2, 0.3, 0.4]")
print("=" * 60)

DROPOUTS   = [0.1, 0.2, 0.3, 0.4]
drop_results = {}

for dp in DROPOUTS:
    print(f"\n── dropout={dp} ──")
    with mlflow.start_run(run_name=f"ablation_A_dropout_{dp}"):
        mlflow.log_param("experiment", "A_dropout_sweep")
        mlflow.log_param("dropout",    dp)

        # Patch dropout in the classification head
        model = MedPromptModel(MODEL_ID, num_classes, class_weights)
        model.cls_head.drop1 = torch.nn.Dropout(dp)
        model.cls_head.drop2 = torch.nn.Dropout(dp)
        model.to(DEVICE)

        train_dl = build_dataloader(df_train, tokenizer, label_map, batch_size=16, shuffle=True)
        val_dl   = build_dataloader(df_val,   tokenizer, label_map, batch_size=16, shuffle=False)

        metrics = quick_train_eval(model, train_dl, val_dl)
        log_metrics_to_mlflow(metrics, prefix="val")
        drop_results[dp] = metrics
        print(f"  dropout={dp}  macro_f1={metrics['macro_f1']:.4f}  acc={metrics['accuracy']:.4f}")

print("\n[EXPERIMENT A SUMMARY]")
print(f"{'Dropout':<10} {'Macro F1':<12} {'Accuracy'}")
for dp, m in drop_results.items():
    print(f"{dp:<10} {m['macro_f1']:<12.4f} {m['accuracy']:.4f}")

# ══════════════════════════════════════════════════════════════
# EXPERIMENT B: Training data size sweep (learning curves)
# WHY: Shows if the model is data-hungry or already saturating.
#      Required by rubric Section B (principled data treatment).
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT B: Data Size Sweep  [25%, 50%, 75%, 100%]")
print("=" * 60)

FRACTIONS    = [0.25, 0.50, 0.75, 1.00]
size_results = {}

for frac in FRACTIONS:
    n  = max(50, int(len(df_train) * frac))
    df_sub = df_train.sample(n=n, random_state=42)
    print(f"\n── fraction={frac*100:.0f}%  ({n} samples) ──")

    with mlflow.start_run(run_name=f"ablation_B_data_{int(frac*100)}pct"):
        mlflow.log_param("experiment",    "B_data_size_sweep")
        mlflow.log_param("data_fraction", frac)
        mlflow.log_param("n_samples",     n)

        model    = MedPromptModel(MODEL_ID, num_classes, class_weights).to(DEVICE)
        train_dl = build_dataloader(df_sub,  tokenizer, label_map, batch_size=16, shuffle=True)
        val_dl   = build_dataloader(df_val,  tokenizer, label_map, batch_size=16, shuffle=False)

        metrics = quick_train_eval(model, train_dl, val_dl)
        log_metrics_to_mlflow(metrics, prefix="val")
        size_results[frac] = metrics
        print(f"  frac={frac:.2f}  macro_f1={metrics['macro_f1']:.4f}  acc={metrics['accuracy']:.4f}")

print("\n[EXPERIMENT B SUMMARY — Learning Curve]")
print(f"{'Fraction':<10} {'Samples':<10} {'Macro F1':<12} {'Accuracy'}")
for frac, m in size_results.items():
    n = int(len(df_train) * frac)
    print(f"{frac*100:.0f}%{'':<7} {n:<10} {m['macro_f1']:<12.4f} {m['accuracy']:.4f}")

# ══════════════════════════════════════════════════════════════
# EXPERIMENT C: Learning rate sweep
# WHY: LR is the most sensitive hyperparameter. Too high → diverge,
#      too low → slow convergence. We find the optimal LR.
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT C: Learning Rate Sweep  [1e-5, 3e-5, 5e-5, 1e-4]")
print("=" * 60)

LRS        = [1e-5, 3e-5, 5e-5, 1e-4]
lr_results = {}

for lr in LRS:
    print(f"\n── lr={lr} ──")
    with mlflow.start_run(run_name=f"ablation_C_lr_{lr}"):
        mlflow.log_param("experiment", "C_lr_sweep")
        mlflow.log_param("lr",         lr)

        model    = MedPromptModel(MODEL_ID, num_classes, class_weights).to(DEVICE)
        train_dl = build_dataloader(df_train, tokenizer, label_map, batch_size=16, shuffle=True)
        val_dl   = build_dataloader(df_val,   tokenizer, label_map, batch_size=16, shuffle=False)

        metrics = quick_train_eval(model, train_dl, val_dl, lr=lr)
        log_metrics_to_mlflow(metrics, prefix="val")
        lr_results[lr] = metrics
        print(f"  lr={lr}  macro_f1={metrics['macro_f1']:.4f}  rouge-L={metrics.get('rougeL',0):.4f}")

print("\n[EXPERIMENT C SUMMARY]")
print(f"{'LR':<10} {'Macro F1':<12} {'ROUGE-L'}")
for lr, m in lr_results.items():
    print(f"{lr:<10} {m['macro_f1']:<12.4f} {m.get('rougeL',0):.4f}")

# ══════════════════════════════════════════════════════════════
# EXPERIMENT D: Max sequence length sweep
# WHY: Longer sequences capture more context but are slower.
#      We quantify the accuracy vs. latency tradeoff on CPU.
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT D: Sequence Length Sweep  [64, 128, 256]")
print("=" * 60)

LENGTHS     = [64, 128, 256]
len_results = {}

for maxlen in LENGTHS:
    print(f"\n── max_length={maxlen} ──")
    with mlflow.start_run(run_name=f"ablation_D_seqlen_{maxlen}"):
        mlflow.log_param("experiment",  "D_seq_len_sweep")
        mlflow.log_param("max_length",  maxlen)

        model    = MedPromptModel(MODEL_ID, num_classes, class_weights).to(DEVICE)
        train_dl = build_dataloader(df_train, tokenizer, label_map,
                                    batch_size=16, shuffle=True,  max_length=maxlen)
        val_dl   = build_dataloader(df_val,   tokenizer, label_map,
                                    batch_size=16, shuffle=False, max_length=maxlen)

        t0      = time.time()
        metrics = quick_train_eval(model, train_dl, val_dl)
        elapsed = time.time() - t0

        metrics["train_time_s"] = round(elapsed, 1)
        log_metrics_to_mlflow(metrics, prefix="val")
        len_results[maxlen] = metrics
        print(f"  len={maxlen}  macro_f1={metrics['macro_f1']:.4f}  time={elapsed:.0f}s")

print("\n[EXPERIMENT D SUMMARY — Accuracy vs Latency]")
print(f"{'Max Len':<10} {'Macro F1':<12} {'Train Time (s)'}")
for ln, m in len_results.items():
    print(f"{ln:<10} {m['macro_f1']:<12.4f} {m.get('train_time_s',0):.1f}")

print("\n" + "=" * 60)
print("🎉 All ablation experiments complete!")
print("=" * 60)
print("\n📸 ACTION REQUIRED (these screenshots = 20% of your grade):")
print("  1. Open terminal: mlflow ui --backend-store-uri ./outputs/mlruns")
print("  2. Go to http://127.0.0.1:5000 in your browser")
print("  3. Click 'medprompt_ablation' experiment")
print("  4. Select all runs → click 'Compare'")
print("  5. Screenshot each experiment's chart")
print("  6. Save to artifacts/ folder in your GitHub repo")
