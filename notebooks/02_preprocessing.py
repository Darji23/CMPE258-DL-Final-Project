# =============================================================
# Notebook 02: Preprocessing — Silver & Gold Layers
# MedPrompt MLOps Project — CPU / Local version
#
# PURPOSE:
#   1. Clean transcription text
#   2. Filter top-15 medical specialties
#   3. Compute inverse-frequency class weights
#   4. Stratified 70/15/15 train/val/test split
#   5. Save label_map.json and class_weights.json
#
# HOW TO RUN:
#   python notebooks/02_preprocessing.py
#
# OUTPUT:
#   outputs/df_train.csv, df_val.csv, df_test.csv
#   outputs/label_map.json
#   outputs/class_weights.json
# =============================================================

import sys, os
sys.path.insert(0, ".")

import json
import mlflow
import pandas as pd
from src.utils import load_and_preprocess

# ── Configuration ─────────────────────────────────────────────
CSV_PATH   = "./data/mtsamples.csv"
OUTPUT_DIR = "./outputs"
TOP_K      = 15
TRAIN_R    = 0.70
VAL_R      = 0.15
SEED       = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("MedPrompt — Preprocessing Pipeline")
print("=" * 60)

# ── Run full pipeline ─────────────────────────────────────────
df_train, df_val, df_test, label_map, weights_dict = load_and_preprocess(
    csv_path    = CSV_PATH,
    top_k       = TOP_K,
    train_ratio = TRAIN_R,
    val_ratio   = VAL_R,
    random_seed = SEED,
    output_dir  = OUTPUT_DIR,
)

# ── Save splits as CSVs ───────────────────────────────────────
print("\nSaving splits to CSV ...")
df_train.to_csv(f"{OUTPUT_DIR}/df_train.csv", index=False)
df_val.to_csv(f"{OUTPUT_DIR}/df_val.csv",   index=False)
df_test.to_csv(f"{OUTPUT_DIR}/df_test.csv",  index=False)
print(f"  ✅ df_train.csv : {len(df_train):,} rows")
print(f"  ✅ df_val.csv   : {len(df_val):,} rows")
print(f"  ✅ df_test.csv  : {len(df_test):,} rows")

# ── Show class weight table ───────────────────────────────────
print("\nClass weight summary (inverse frequency):")
id_to_spec = label_map["id_to_specialty"]
counts = df_train["medical_specialty"].value_counts()
print(f"  {'Label':<4} {'Specialty':<40} {'Train count':>12} {'Weight':>8}")
for lbl_str, spec in sorted(id_to_spec.items(), key=lambda x: int(x[0])):
    lbl  = int(lbl_str)
    cnt  = int(counts.get(spec, 0))
    w    = weights_dict.get(lbl, 0)
    print(f"  [{lbl:2d}] {spec:<40} {cnt:>12,} {w:>8.3f}")

# ── Log to MLflow ─────────────────────────────────────────────
print("\nLogging preprocessing metrics to MLflow ...")
mlflow.set_tracking_uri("sqlite:///outputs/mlflow.db")
mlflow.set_experiment("medprompt_preprocessing")

with mlflow.start_run(run_name="preprocessing"):
    mlflow.log_param("top_k_specialties", TOP_K)
    mlflow.log_param("train_ratio",       TRAIN_R)
    mlflow.log_param("val_ratio",         VAL_R)
    mlflow.log_param("random_seed",       SEED)
    mlflow.log_metric("train_size",       len(df_train))
    mlflow.log_metric("val_size",         len(df_val))
    mlflow.log_metric("test_size",        len(df_test))
    mlflow.log_artifact(f"{OUTPUT_DIR}/label_map.json")
    mlflow.log_artifact(f"{OUTPUT_DIR}/class_weights.json")

print("  ✅ Metrics logged to MLflow")
print("\n🎉 Notebook 02 complete. Run notebook 03 next.")
