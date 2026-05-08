# =============================================================
# Notebook 03: Fine-Tuning — DistilBERT (CPU-friendly)
# MedPrompt MLOps Project
#
# PURPOSE:
#   Fine-tune DistilBERT on MTSamples for:
#     - Medical specialty classification (15 classes)
#     - Extractive clinical summarization
#   Logs every metric to MLflow and saves the best checkpoint.
#
# HOW TO RUN:
#   python notebooks/03_fine_tuning.py
#
# ESTIMATED TIME (CPU):
#   ~20–30 min for 3 epochs on a standard laptop
#
# OUTPUT:
#   outputs/results/best_checkpoint/model.pt
#   MLflow experiment: medprompt_training
# =============================================================

import sys, os, json, time, yaml
sys.path.insert(0, ".")

import torch
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd

from src.model   import MedPromptModel, load_tokenizer
from src.utils   import build_dataloader, load_label_map, load_class_weights
from src.metrics import compute_classification_metrics, compute_rouge, log_metrics_to_mlflow

# ── Load config ───────────────────────────────────────────────
with open("./config/training_config.yaml") as f:
    cfg = yaml.safe_load(f)

MODEL_ID    = cfg["model"]["base_model_id"]
OUTPUT_DIR  = cfg["training"]["output_dir"]
EPOCHS      = cfg["training"]["num_train_epochs"]
LR          = cfg["training"]["learning_rate"]
BATCH_SIZE  = cfg["training"]["per_device_train_batch_size"]
MAX_LEN     = cfg["data"]["max_input_length"]
HUB_ID      = cfg["huggingface"]["hub_model_id"]
MODEL_NAME  = cfg["mlflow"]["model_name"]
LABEL_MAP   = "./outputs/label_map.json"
WEIGHTS_PATH= "./outputs/class_weights.json"
DEVICE      = "cpu"   # DistilBERT runs fine on CPU

CKPT_DIR = f"{OUTPUT_DIR}/best_checkpoint"
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"Device     : {DEVICE}")
print(f"Model      : {MODEL_ID}")
print(f"Epochs     : {EPOCHS}")
print(f"Batch size : {BATCH_SIZE}")
print(f"Max tokens : {MAX_LEN}")

# ── Load data ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Loading preprocessed splits")
print("=" * 60)

label_map     = load_label_map(LABEL_MAP)
class_weights = load_class_weights(WEIGHTS_PATH, device=DEVICE)
id_to_label   = {int(k): v for k, v in label_map["id_to_specialty"].items()}
num_classes   = len(label_map["specialty_to_id"])

df_train = pd.read_csv("./outputs/df_train.csv")
df_val   = pd.read_csv("./outputs/df_val.csv")
df_test  = pd.read_csv("./outputs/df_test.csv")

print(f"  Train: {len(df_train):,}  |  Val: {len(df_val):,}  |  Test: {len(df_test):,}")
print(f"  Classes: {num_classes}")

# ── Load model & tokenizer ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Loading DistilBERT")
print("=" * 60)

tokenizer = load_tokenizer(MODEL_ID)
model     = MedPromptModel(
    model_id      = MODEL_ID,
    num_classes   = num_classes,
    class_weights = class_weights,
)
model.to(DEVICE)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params     : {total_params:,}")
print(f"  Trainable params : {trainable_params:,}")

# ── DataLoaders ───────────────────────────────────────────────
print("\nSTEP 3: Building DataLoaders")
train_loader = build_dataloader(df_train, tokenizer, label_map, batch_size=BATCH_SIZE, shuffle=True,  max_length=MAX_LEN)
val_loader   = build_dataloader(df_val,   tokenizer, label_map, batch_size=BATCH_SIZE, shuffle=False, max_length=MAX_LEN)
test_loader  = build_dataloader(df_test,  tokenizer, label_map, batch_size=BATCH_SIZE, shuffle=False, max_length=MAX_LEN)
print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

# ── Optimizer & scheduler ─────────────────────────────────────
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=cfg["training"]["weight_decay"],
)
total_steps = EPOCHS * len(train_loader)
warmup_steps= int(cfg["training"]["warmup_ratio"] * total_steps)
scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

# ── MLflow setup ──────────────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///outputs/mlflow.db")
mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

class_names  = [id_to_label[i] for i in sorted(id_to_label.keys())]

print("\n" + "=" * 60)
print("STEP 4: Training")
print("=" * 60)

with mlflow.start_run(run_name=f"distilbert_lr{LR}_ep{EPOCHS}") as run:

    mlflow.log_params({
        "model_id":    MODEL_ID,
        "epochs":      EPOCHS,
        "lr":          LR,
        "batch_size":  BATCH_SIZE,
        "max_len":     MAX_LEN,
        "num_classes": num_classes,
        "device":      DEVICE,
        "total_params":total_params,
    })

    best_val_f1  = 0.0
    global_step  = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        t0         = time.time()

        for step, batch in enumerate(train_loader):
            input_ids    = batch["input_ids"].to(DEVICE)
            attn_mask    = batch["attention_mask"].to(DEVICE)
            cls_labels   = batch["cls_labels"].to(DEVICE)
            token_labels = batch["token_labels"].to(DEVICE)

            optimizer.zero_grad()
            out  = model(input_ids, attn_mask,
                         cls_labels=cls_labels, token_labels=token_labels)
            loss = out["total_loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["max_grad_norm"])
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % cfg["training"]["logging_steps"] == 0:
                avg = epoch_loss / (step + 1)
                mlflow.log_metric("train/loss",     avg,                       step=global_step)
                mlflow.log_metric("train/cls_loss", out["cls_loss"].item(),    step=global_step)
                mlflow.log_metric("train/sum_loss", out.get("summary_loss", torch.tensor(0)).item(), step=global_step)
                mlflow.log_metric("lr",             scheduler.get_last_lr()[0],step=global_step)
                print(f"  Epoch {epoch+1}/{EPOCHS}  step {global_step:4d}  "
                      f"loss={avg:.4f}  cls={out['cls_loss'].item():.4f}")

        print(f"\n  ── Epoch {epoch+1} done in {time.time()-t0:.0f}s ──")

        # ── Validation ────────────────────────────────────────
        model.eval()
        val_preds, val_true = [], []
        val_summaries, val_refs = [], []

        with torch.no_grad():
            for batch in val_loader:
                ids   = batch["input_ids"].to(DEVICE)
                mask  = batch["attention_mask"].to(DEVICE)
                out   = model(ids, mask)
                preds = out["logits"].argmax(-1).cpu().tolist()
                val_preds.extend(preds)
                val_true.extend(batch["cls_labels"].tolist())

                if len(val_summaries) < 32:
                    sums = model.extract_summary(ids, mask, tokenizer, top_k=20)
                    val_summaries.extend(sums)
                    val_refs.extend(batch["reference_summary"])

        val_cls   = compute_classification_metrics(val_preds, val_true, class_names)
        val_rouge = compute_rouge(val_summaries, val_refs)
        val_all   = {**val_cls, **val_rouge}
        log_metrics_to_mlflow(val_all, step=global_step, prefix="val")

        print(f"  [VAL] acc={val_cls['accuracy']:.4f}  "
              f"macro_f1={val_cls['macro_f1']:.4f}  "
              f"ROUGE-L={val_rouge.get('rougeL',0):.4f}")

        if val_cls["macro_f1"] > best_val_f1:
            best_val_f1 = val_cls["macro_f1"]
            torch.save(model.state_dict(), f"{CKPT_DIR}/model.pt")
            tokenizer.save_pretrained(CKPT_DIR)
            mlflow.log_metric("best_val_macro_f1", best_val_f1, step=global_step)
            print(f"  ✅ Best model saved (macro_f1={best_val_f1:.4f})")

    # ── Test evaluation ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Test evaluation")
    print("=" * 60)

    model.load_state_dict(torch.load(f"{CKPT_DIR}/model.pt", map_location=DEVICE))
    model.eval()
    test_preds, test_true = [], []

    with torch.no_grad():
        for batch in test_loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            out  = model(ids, mask)
            test_preds.extend(out["logits"].argmax(-1).cpu().tolist())
            test_true.extend(batch["cls_labels"].tolist())

    test_metrics = compute_classification_metrics(test_preds, test_true, class_names)
    log_metrics_to_mlflow(test_metrics, step=global_step, prefix="test")

    print(f"\n  [TEST] accuracy={test_metrics['accuracy']:.4f}  "
          f"macro_f1={test_metrics['macro_f1']:.4f}  "
          f"weighted_f1={test_metrics['weighted_f1']:.4f}  "
          f"kappa={test_metrics['kappa']:.4f}")

    # ── Register model ────────────────────────────────────────
    print("\nSTEP 6: Registering model in MLflow Model Registry")
    mlflow.pytorch.log_model(model, artifact_path="medprompt_model")
    model_uri = f"runs:/{run.info.run_id}/medprompt_model"

    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    try:
        client.create_registered_model(MODEL_NAME)
    except Exception:
        pass
    mv = client.create_model_version(name=MODEL_NAME, source=model_uri, run_id=run.info.run_id)
    client.transition_model_version_stage(
        name=MODEL_NAME, version=mv.version,
        stage="Production", archive_existing_versions=True,
    )
    print(f"  ✅ '{MODEL_NAME}' v{mv.version} → Production")

    # ── Push to Hugging Face Hub ──────────────────────────────
    print(f"\nSTEP 7: Pushing to Hugging Face Hub: {HUB_ID}")
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("  ⚠️  HF_TOKEN not set. Set it with:")
        print("       export HF_TOKEN=hf_xxxxxxxxxxxx")
        print(f"     Then run: huggingface-cli upload {HUB_ID} {CKPT_DIR}")
    else:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_folder(
                folder_path=CKPT_DIR,
                repo_id=HUB_ID,
                repo_type="model",
                token=hf_token,
                commit_message=f"MedPrompt DistilBERT (val macro_f1={best_val_f1:.4f})",
            )
            api.upload_file(
                path_or_fileobj="./outputs/label_map.json",
                path_in_repo="label_map.json",
                repo_id=HUB_ID,
                repo_type="model",
                token=hf_token,
            )
            print(f"  ✅ Model live at https://huggingface.co/{HUB_ID}")
        except Exception as e:
            print(f"  ❌ Push failed: {e}")

    # ── Auto-deploy: only push to HF Hub if test F1 exceeds threshold ──
    DEPLOY_THRESHOLD = 0.2   # minimum macro F1 required to deploy
    print("\nSTEP 8: Auto-deploy check")
    print(f"  Test macro_f1 : {test_metrics['macro_f1']:.4f}")
    print(f"  Threshold     : {DEPLOY_THRESHOLD}")

    if test_metrics["macro_f1"] >= DEPLOY_THRESHOLD:
        print(f"  ✅ Threshold passed — triggering auto-deploy to HF Hub")
        mlflow.log_metric("auto_deploy_triggered", 1)
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                api.upload_folder(
                    folder_path=CKPT_DIR,
                    repo_id=HUB_ID,
                    repo_type="model",
                    token=hf_token,
                    commit_message=f"Auto-deploy: macro_f1={test_metrics['macro_f1']:.4f} passed threshold={DEPLOY_THRESHOLD}",
                )
                api.upload_file(
                    path_or_fileobj="./outputs/label_map.json",
                    path_in_repo="label_map.json",
                    repo_id=HUB_ID,
                    repo_type="model",
                    token=hf_token,
                )
                print(f"  ✅ Model auto-deployed to https://huggingface.co/{HUB_ID}")
            except Exception as e:
                print(f"  ❌ Auto-deploy failed: {e}")
        else:
            print("  ⚠️  HF_TOKEN not set — skipping actual push")
            print("     Set export HF_TOKEN=hf_xxx and re-run to deploy")
    else:
        print(f"  ❌ Threshold NOT passed — model not deployed")
        print(f"     Improve the model before deploying to production")
        mlflow.log_metric("auto_deploy_triggered", 0)

print(f"\n🎉 Notebook 03 complete!")
print(f"   Best val macro-F1 : {best_val_f1:.4f}")
print(f"   Test accuracy     : {test_metrics['accuracy']:.4f}")
print(f"   MLflow run ID     : {run.info.run_id}")
print("\nNext: Run notebook 04_ablation.py")
