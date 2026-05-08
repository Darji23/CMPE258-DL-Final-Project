# =============================================================
# src/metrics.py — Evaluation Metrics for MedPrompt
# =============================================================
#
# Metrics used:
#
#   Classification:
#     - Accuracy       : overall proportion correct
#     - Macro F1       : mean F1 across all classes
#                        (punishes ignoring minority classes)
#     - Weighted F1    : F1 weighted by class frequency
#     - Per-class F1   : used in ablation confusion matrix
#     - Cohen's Kappa  : agreement corrected for chance
#
#   Summarization (ROUGE):
#     - ROUGE-1 F1     : unigram overlap
#     - ROUGE-2 F1     : bigram overlap (captures phrases)
#     - ROUGE-L F1     : longest common subsequence
#
#   WHY ROUGE not BLEU:
#     BLEU heavily penalises short outputs. Clinical summaries
#     are intentionally concise, making BLEU unreliable.
#     ROUGE-L is the standard in clinical NLP papers.
#
#   Latency:
#     - p50 / p95 ms   : used in ablation Experiment D
# =============================================================

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
)

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️  rouge_score not installed. Run: pip install rouge-score")


# ── Classification metrics ────────────────────────────────────
def compute_classification_metrics(
    preds:       List[int],
    labels:      List[int],
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    acc         = accuracy_score(labels, preds)
    macro_f1    = f1_score(labels, preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    kappa       = cohen_kappa_score(labels, preds)
    per_class   = f1_score(labels, preds, average=None,       zero_division=0)

    metrics = {
        "accuracy":    round(float(acc),         4),
        "macro_f1":    round(float(macro_f1),    4),
        "weighted_f1": round(float(weighted_f1), 4),
        "kappa":       round(float(kappa),       4),
    }

    for i, score in enumerate(per_class):
        label = class_names[i] if class_names else str(i)
        # sanitise label for MLflow key (no spaces/slashes)
        key   = label.replace(" ", "_").replace("/", "_")
        metrics[f"f1_{key}"] = round(float(score), 4)

    return metrics


def get_confusion_matrix(
    preds:       List[int],
    labels:      List[int],
    class_names: List[str],
) -> np.ndarray:
    cm = confusion_matrix(labels, preds)
    return cm.astype(float) / cm.sum(axis=1, keepdims=True)


def print_classification_report(
    preds:       List[int],
    labels:      List[int],
    class_names: List[str],
) -> None:
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))


# ── ROUGE metrics ─────────────────────────────────────────────
def compute_rouge(
    predictions: List[str],
    references:  List[str],
) -> Dict[str, float]:
    if not ROUGE_AVAILABLE or not predictions or not references:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            continue
        r = scorer.score(str(ref), str(pred))
        for k in scores:
            scores[k].append(r[k].fmeasure)

    return {k: round(float(np.mean(v)) if v else 0.0, 4) for k, v in scores.items()}


# ── Latency tracker ───────────────────────────────────────────
class LatencyTracker:
    """Records per-request latency for the monitoring dashboard."""

    def __init__(self, window: int = 100):
        self.window     = window
        self._latencies: List[float] = []

    def record(self, ms: float) -> None:
        self._latencies.append(ms)
        if len(self._latencies) > self.window:
            self._latencies.pop(0)

    @property
    def p50(self) -> float:
        return float(np.percentile(self._latencies, 50)) if self._latencies else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(self._latencies, 95)) if self._latencies else 0.0

    def summary(self) -> Dict[str, float]:
        return {
            "latency_p50_ms":  round(self.p50, 1),
            "latency_p95_ms":  round(self.p95, 1),
            "latency_mean_ms": round(float(np.mean(self._latencies)) if self._latencies else 0, 1),
            "n_requests":      len(self._latencies),
        }


# ── MLflow logging helper ─────────────────────────────────────
def log_metrics_to_mlflow(
    metrics: Dict[str, float],
    step:    int = 0,
    prefix:  str = "",
) -> None:
    try:
        import mlflow
        for k, v in metrics.items():
            key = f"{prefix}/{k}" if prefix else k
            mlflow.log_metric(key, v, step=step)
    except Exception as e:
        print(f"MLflow logging skipped: {e}")
