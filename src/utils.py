# =============================================================
# src/utils.py — Data loading & preprocessing utilities
# (CPU / local version — no Spark, pure pandas)
# =============================================================

import json
import re
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast
from typing import Dict, List, Optional, Tuple
from collections import Counter
from sklearn.model_selection import train_test_split


# ── Text cleaner ──────────────────────────────────────────────
def clean_medical_text(text: str) -> str:
    """
    Clean a raw medical transcription string.
    Removes de-id brackets, ALL-CAPS headers, excess whitespace.
    Lowercases the result.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'\[\*\*.*?\*\*\]', '', text)           # de-id tags
    text = re.sub(r'\n[A-Z][A-Z\s/\-]{3,}:\s*\n', '\n', text)  # headers
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [ln.strip() for ln in text.split('\n') if len(ln.strip()) > 5]
    return ' '.join(lines).strip().lower()


# ── Full preprocessing pipeline (replaces notebooks 01 + 02) ──
def load_and_preprocess(
    csv_path:     str,
    top_k:        int   = 15,
    train_ratio:  float = 0.70,
    val_ratio:    float = 0.15,
    random_seed:  int   = 42,
    output_dir:   str   = "./outputs",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """
    One-shot data pipeline:
      1. Load CSV
      2. Drop nulls
      3. Clean text
      4. Filter top-K specialties
      5. Compute class weights
      6. Stratified 70/15/15 split
      7. Save label_map.json and class_weights.json

    Returns: df_train, df_val, df_test, label_map, class_weights_dict
    """
    import os, json
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # 2. Drop nulls
    before = len(df)
    df = df.dropna(subset=["transcription", "medical_specialty"])
    print(f"  Dropped {before - len(df)} null rows → {len(df):,} remain")

    # 3. Clean
    df["medical_specialty"] = df["medical_specialty"].str.strip()
    df["cleaned_text"]      = df["transcription"].apply(clean_medical_text)
    df = df[df["cleaned_text"].str.len() > 20].reset_index(drop=True)

    # 4. Top-K filter
    counts      = df["medical_specialty"].value_counts()
    top_specs   = counts.head(top_k).index.tolist()
    df          = df[df["medical_specialty"].isin(top_specs)].reset_index(drop=True)
    print(f"  After top-{top_k} filter: {len(df):,} rows")

    # 5. Label encoding
    specialty_to_id = {s: i for i, s in enumerate(sorted(top_specs))}
    id_to_specialty = {str(v): k for k, v in specialty_to_id.items()}
    df["label"]     = df["medical_specialty"].map(specialty_to_id)

    label_map = {"specialty_to_id": specialty_to_id, "id_to_specialty": id_to_specialty}
    with open(f"{output_dir}/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"  Label map saved → {output_dir}/label_map.json")

    # 6. Class weights (inverse frequency)
    total = len(df)
    n_cls = len(specialty_to_id)
    weights_dict = {}
    for spec, cnt in df["medical_specialty"].value_counts().items():
        lbl = specialty_to_id[spec]
        weights_dict[lbl] = round(total / (n_cls * cnt), 4)

    with open(f"{output_dir}/class_weights.json", "w") as f:
        json.dump(weights_dict, f, indent=2)
    print(f"  Class weights saved → {output_dir}/class_weights.json")

    # 7. Stratified split
    test_ratio = 1.0 - train_ratio - val_ratio
    df_train, df_temp = train_test_split(
        df, test_size=(1 - train_ratio), stratify=df["label"], random_state=random_seed
    )
    val_frac = val_ratio / (val_ratio + test_ratio)
    df_val, df_test = train_test_split(
        df_temp, test_size=(1 - val_frac), stratify=df_temp["label"], random_state=random_seed
    )

    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

    print(f"  Split → train:{len(df_train):,}  val:{len(df_val):,}  test:{len(df_test):,}")
    return df_train, df_val, df_test, label_map, weights_dict


# ── PyTorch Dataset ───────────────────────────────────────────
class MedSamplesDataset(Dataset):
    """
    PyTorch Dataset for MTSamples.

    Each item:
        input_ids      : tokenised transcription (max 256 tokens)
        attention_mask : padding mask
        cls_labels     : integer specialty label
        token_labels   : binary per-token labels for extractive summary
                         (1 = token appears in the description, 0 = not)
        reference_summary: the description field (str) for ROUGE eval
    """

    def __init__(
        self,
        df:         pd.DataFrame,
        tokenizer:  DistilBertTokenizerFast,
        label_map:  Dict,
        max_length: int = 256,
    ):
        self.df         = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.s2id       = label_map["specialty_to_id"]
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row       = self.df.iloc[idx]
        text      = str(row["cleaned_text"])
        specialty = str(row["medical_specialty"]).strip()
        desc      = str(row.get("description", ""))

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Build token-level labels for extractive summary
        # Any token that appears in `desc` gets label 1
        desc_tokens = set(
            self.tokenizer.encode(desc[:200], add_special_tokens=False)
        )
        token_labels = torch.zeros(self.max_length, dtype=torch.float32)
        for i, tid in enumerate(input_ids.tolist()):
            if tid in desc_tokens:
                token_labels[i] = 1.0

        cls_label = self.s2id.get(specialty, 0)

        return {
            "input_ids":         input_ids,
            "attention_mask":    attention_mask,
            "cls_labels":        torch.tensor(cls_label, dtype=torch.long),
            "token_labels":      token_labels,
            "reference_summary": desc,
        }


# ── DataLoader builder ────────────────────────────────────────
def build_dataloader(
    df:         pd.DataFrame,
    tokenizer:  DistilBertTokenizerFast,
    label_map:  Dict,
    batch_size: int  = 16,
    shuffle:    bool = True,
    max_length: int  = 256,
) -> DataLoader:
    ds = MedSamplesDataset(df, tokenizer, label_map, max_length)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


# ── Helper loaders ────────────────────────────────────────────
def load_label_map(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def load_class_weights(path: str, device: str = "cpu") -> torch.Tensor:
    with open(path) as f:
        d = json.load(f)
    n = len(d)
    w = torch.zeros(n, dtype=torch.float32)
    for k, v in d.items():
        w[int(k)] = float(v)
    return w.to(device)
