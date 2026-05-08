# =============================================================
# Notebook 05: Register Best Model & Push to Hugging Face Hub
# MedPrompt MLOps Project — CPU / Local version
#
# PURPOSE:
#   1. Find best MLflow run by val/macro_f1
#   2. Promote it to "Production" in MLflow Model Registry
#   3. Push model checkpoint + label_map to Hugging Face Hub
#
# HOW TO RUN:
#   export HF_TOKEN=hf_xxxxxxxxxxxx
#   python notebooks/05_register_push.py
# =============================================================

import sys, os, json
sys.path.insert(0, ".")

import mlflow
from mlflow.tracking import MlflowClient

# ── Configuration ─────────────────────────────────────────────
EXPERIMENT_NAME = "medprompt_training"
MODEL_NAME      = "MedPrompt_DistilBERT"
HF_HUB_ID       = "abhishekdarji23/medprompt-distilbert"  # ← CHANGE THIS
LABEL_MAP_PATH  = "./outputs/label_map.json"
CKPT_DIR        = "./outputs/results/best_checkpoint"

mlflow.set_tracking_uri("./outputs/mlruns")

print("=" * 60)
print("STEP 1: Finding best MLflow run by val/macro_f1")
print("=" * 60)

client     = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found. Run notebook 03 first.")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.`val/macro_f1` DESC"],
    max_results=10,
)

if not runs:
    raise RuntimeError("No runs found. Complete notebook 03 first.")

best_run = runs[0]
best_f1  = best_run.data.metrics.get("val/macro_f1", 0)

print(f"  Best run ID      : {best_run.info.run_id}")
print(f"  Best val macro_f1: {best_f1:.4f}")
print(f"  LR               : {best_run.data.params.get('lr', 'N/A')}")
print(f"  Epochs           : {best_run.data.params.get('epochs', 'N/A')}")

# ── Register in MLflow ────────────────────────────────────────
print("\nSTEP 2: Registering in MLflow Model Registry → Production")

model_uri = f"runs:/{best_run.info.run_id}/medprompt_model"
try:
    client.create_registered_model(MODEL_NAME)
    print(f"  Created registered model: '{MODEL_NAME}'")
except Exception:
    print(f"  Model '{MODEL_NAME}' already exists — adding new version")

mv = client.create_model_version(
    name=MODEL_NAME,
    source=model_uri,
    run_id=best_run.info.run_id,
    description=f"DistilBERT MedPrompt, val macro_f1={best_f1:.4f}",
)
client.transition_model_version_stage(
    name=MODEL_NAME, version=mv.version,
    stage="Production", archive_existing_versions=True,
)
print(f"  ✅ '{MODEL_NAME}' v{mv.version} → Production")

# ── Push to Hugging Face Hub ──────────────────────────────────
print(f"\nSTEP 3: Pushing to Hugging Face Hub: {HF_HUB_ID}")

hf_token = os.environ.get("HF_TOKEN", "")
if not hf_token:
    print("  ⚠️  HF_TOKEN not set.")
    print("  Set it with:  export HF_TOKEN=hf_xxxxxxxxxxxx")
    print(f"  Then re-run this script.")
else:
    try:
        from huggingface_hub import HfApi
        api = HfApi()

        if not os.path.exists(CKPT_DIR):
            raise FileNotFoundError(f"Checkpoint not found at {CKPT_DIR}. Run notebook 03 first.")

        api.upload_folder(
            folder_path=CKPT_DIR,
            repo_id=HF_HUB_ID,
            repo_type="model",
            token=hf_token,
            commit_message=f"MedPrompt DistilBERT checkpoint (macro_f1={best_f1:.4f})",
        )
        api.upload_file(
            path_or_fileobj=LABEL_MAP_PATH,
            path_in_repo="label_map.json",
            repo_id=HF_HUB_ID,
            repo_type="model",
            token=hf_token,
            commit_message="Add label_map.json for inference",
        )
        print(f"  ✅ Model live at https://huggingface.co/{HF_HUB_ID}")
    except Exception as e:
        print(f"  ❌ Push failed: {e}")

print("\n" + "=" * 60)
print("STEP 4: Summary")
print("=" * 60)
print(f"  MLflow Registry : {MODEL_NAME} v{mv.version} (Production)")
print(f"  HF Hub          : https://huggingface.co/{HF_HUB_ID}")
print(f"  Best val F1     : {best_f1:.4f}")
print("\n  ✅ Gradio app can now load weights from HF Hub.")
print("     Your demo stays live permanently (HF Spaces is free).")
print("\nNext step: Deploy app/app.py to Hugging Face Spaces")
