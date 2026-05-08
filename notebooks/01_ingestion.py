# =============================================================
# Notebook 01: Data Ingestion & Quality Audit
# MedPrompt MLOps Project — CPU / Local version
#
# PURPOSE:
#   Load mtsamples.csv, run a data quality audit, and save
#   a clean copy to the data/ folder. No Spark required.
#
# HOW TO RUN:
#   1. Place mtsamples.csv inside the data/ folder
#   2. From the project root, run:
#        python notebooks/01_ingestion.py
#
# OUTPUT:
#   data/mtsamples.csv   (confirmed present)
#   outputs/audit.json   (row counts, null counts, specialty list)
# =============================================================

import os, json
import pandas as pd

# ── Configuration ─────────────────────────────────────────────
RAW_DATA_PATH = "./data/mtsamples.csv"
OUTPUT_DIR    = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("STEP 1: Reading raw MTSamples CSV")
print("=" * 60)

if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(
        f"\n❌ File not found: {RAW_DATA_PATH}"
        f"\nFix: copy mtsamples.csv into the data/ folder."
    )

df = pd.read_csv(RAW_DATA_PATH)
print(f"  ✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
print(f"  Columns: {df.columns.tolist()}")

# ── Quality audit ─────────────────────────────────────────────
print("\nSTEP 2: Data quality audit")
print("-" * 40)

total_rows         = len(df)
null_transcription = df["transcription"].isna().sum()
null_keywords      = df["keywords"].isna().sum()
null_specialty     = df["medical_specialty"].isna().sum()
unique_specialties = df["medical_specialty"].nunique()

print(f"  Total rows          : {total_rows:,}")
print(f"  Null transcriptions : {null_transcription}  ← dropped in notebook 02")
print(f"  Null keywords       : {null_keywords}  ← not used as label, acceptable")
print(f"  Null specialties    : {null_specialty}")
print(f"  Unique specialties  : {unique_specialties}")

# ── Specialty distribution ────────────────────────────────────
print("\nSTEP 3: Specialty distribution (all)")
print("-" * 40)
dist = df["medical_specialty"].str.strip().value_counts()
for spec, cnt in dist.items():
    print(f"  {spec:<45} {cnt:>5}")

# ── Avg transcription length ──────────────────────────────────
print("\nSTEP 4: Transcription length stats")
print("-" * 40)
lengths = df["transcription"].dropna().apply(len)
print(f"  Min chars  : {lengths.min():,}")
print(f"  Max chars  : {lengths.max():,}")
print(f"  Mean chars : {lengths.mean():,.0f}")
print(f"  Median     : {lengths.median():,.0f}")

# ── Save audit report ─────────────────────────────────────────
audit = {
    "total_rows":         int(total_rows),
    "null_transcriptions":int(null_transcription),
    "null_keywords":      int(null_keywords),
    "null_specialties":   int(null_specialty),
    "unique_specialties": int(unique_specialties),
    "specialty_counts":   dist.to_dict(),
    "transcription_len":  {
        "min":    int(lengths.min()),
        "max":    int(lengths.max()),
        "mean":   round(float(lengths.mean()), 1),
        "median": round(float(lengths.median()), 1),
    },
}
audit_path = f"{OUTPUT_DIR}/audit.json"
with open(audit_path, "w") as f:
    json.dump(audit, f, indent=2)

print(f"\n✅ Audit report saved → {audit_path}")
print("\n🎉 Notebook 01 complete. Run notebook 02 next.")
