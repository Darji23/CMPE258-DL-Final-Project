# =============================================================
# app/app.py — MedPrompt Gradio Application
# Hugging Face Spaces Deployment
#
# Features:
#   Tab 1 — Inference: classify + summarize clinical notes
#   Tab 2 — Monitoring: live metrics + drift detection
#   Tab 3 — About: architecture + design decisions
# =============================================================

import os, json, time, math
import torch
import gradio as gr
from transformers import DistilBertTokenizerFast
from huggingface_hub import hf_hub_download
from collections import Counter
import numpy as np

HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "abhishekdarji23/medprompt-distilbert")
DEVICE      = "cpu"

# ── Drift detection config ────────────────────────────────────
# We track the distribution of predicted specialties over time.
# If the current window diverges significantly from the training
# distribution, we flag it as potential data drift.
# We use Jensen-Shannon divergence (0 = identical, 1 = max drift)
DRIFT_THRESHOLD = 0.3   # flag drift if JS divergence exceeds this

# Training set class distribution (approximate, from MTSamples top-15)
# Used as the "reference distribution" for drift detection
TRAIN_DISTRIBUTION = {
    "Surgery": 0.234,
    "Consult - History and Phy.": 0.149,
    "Cardiovascular / Pulmonary": 0.090,
    "Orthopedic": 0.085,
    "Radiology": 0.083,
    "General Medicine": 0.065,
    "Gastroenterology": 0.052,
    "Neurology": 0.050,
    "SOAP / Chart / Progress Notes": 0.047,
    "Obstetrics / Gynecology": 0.044,
    "Urology": 0.036,
    "Discharge Summary": 0.032,
    "ENT - Otolaryngology": 0.030,
    "Neurosurgery": 0.028,
    "Hematology - Oncology": 0.025,
}

# ── Load label map ────────────────────────────────────────────
try:
    lm_path    = hf_hub_download(repo_id=HF_MODEL_ID, filename="label_map.json")
    label_map  = json.load(open(lm_path))
    id_to_spec = {int(k): v for k, v in label_map["id_to_specialty"].items()}
    NUM_CLASSES= len(id_to_spec)
except Exception as e:
    print(f"Warning: {e}")
    id_to_spec = {i: f"Class {i}" for i in range(15)}
    NUM_CLASSES= 15

# ── In-memory monitoring store ────────────────────────────────
request_log   = []   # list of dicts, one per inference request
feedback_log  = []   # list of thumbs up/down feedback entries

# ── Lazy model load ───────────────────────────────────────────
_model, _tokenizer = None, None

def get_model():
    global _model, _tokenizer
    if _model is None:
        import sys; sys.path.insert(0, ".")
        from src.model import MedPromptModel
        _tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        _model     = MedPromptModel("distilbert-base-uncased", NUM_CLASSES)
        try:
            weights_path = hf_hub_download(repo_id=HF_MODEL_ID, filename="model.pt")
            _model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            print("✅ Weights loaded from HF Hub")
        except Exception as e:
            print(f"⚠️  Could not load weights: {e}. Using random init for demo.")
        _model.eval()
    return _model, _tokenizer


# ── Drift detection helpers ───────────────────────────────────
def compute_js_divergence(observed: dict, reference: dict) -> float:
    """
    Compute Jensen-Shannon divergence between observed and reference
    specialty distributions.
    JS divergence = 0 means identical distributions (no drift).
    JS divergence = 1 means maximum divergence (severe drift).
    """
    all_keys = set(list(observed.keys()) + list(reference.keys()))
    p = np.array([reference.get(k, 1e-10) for k in all_keys])
    q = np.array([observed.get(k, 1e-10)  for k in all_keys])

    # Normalise
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return round(min(js, 1.0), 4)


def check_drift() -> dict:
    """
    Compare the last 50 predictions against the training distribution.
    Returns drift score and whether drift is detected.
    """
    if len(request_log) < 10:
        return {"score": 0.0, "detected": False, "message": "Need at least 10 requests to detect drift."}

    recent    = request_log[-50:]
    counts    = Counter(r["specialty"] for r in recent)
    total     = sum(counts.values())
    observed  = {k: v / total for k, v in counts.items()}

    js_score  = compute_js_divergence(observed, TRAIN_DISTRIBUTION)
    detected  = js_score > DRIFT_THRESHOLD

    return {
        "score":    js_score,
        "detected": detected,
        "message":  f"⚠️ DRIFT DETECTED (JS={js_score:.3f})" if detected
                    else f"✅ No drift detected (JS={js_score:.3f})",
    }


# ── Inference ─────────────────────────────────────────────────
def run_inference(clinical_note: str):
    if not clinical_note.strip():
        return "Please enter a clinical note.", "", "", ""

    model, tokenizer = get_model()
    enc = tokenizer(
        clinical_note[:1500],
        return_tensors="pt", max_length=256,
        truncation=True, padding="max_length",
    )

    t0 = time.time()
    with torch.no_grad():
        out    = model(enc["input_ids"], enc["attention_mask"])
        probs  = torch.softmax(out["logits"], dim=-1)
        cls_idx= int(probs.argmax(-1).item())
        conf   = float(probs.max().item()) * 100
        sums   = model.extract_summary(
            enc["input_ids"], enc["attention_mask"], tokenizer, top_k=25
        )
    latency = round((time.time() - t0) * 1000, 1)

    specialty = id_to_spec.get(cls_idx, f"Class {cls_idx}")

    # Log request for monitoring + drift detection
    request_log.append({
        "timestamp":   time.strftime("%H:%M:%S"),
        "specialty":   specialty,
        "confidence":  round(conf, 1),
        "latency_ms":  latency,
        "note_length": len(clinical_note),
    })
    if len(request_log) > 500:
        request_log.pop(0)

    return specialty, f"{conf:.1f}%", sums[0] if sums else "", f"{latency} ms"


def record_feedback(specialty: str, feedback: str):
    """Record thumbs up/down feedback for monitoring."""
    if specialty and feedback:
        feedback_log.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "specialty": specialty,
            "correct":   feedback == "👍 Correct",
        })


# ── Monitoring dashboard ──────────────────────────────────────
def get_monitoring_stats() -> str:
    if not request_log:
        return "### No requests yet.\nMake some predictions in Tab 1 first."

    lats  = [r["latency_ms"]  for r in request_log]
    confs = [r["confidence"]  for r in request_log]
    top   = Counter(r["specialty"] for r in request_log).most_common(5)

    # Feedback accuracy
    if feedback_log:
        correct = sum(1 for f in feedback_log if f["correct"])
        acc     = round(correct / len(feedback_log) * 100, 1)
        feedback_str = f"**User-reported accuracy:** {acc}% ({correct}/{len(feedback_log)} correct)"
    else:
        feedback_str = "**User feedback:** No feedback submitted yet."

    # Drift check
    drift = check_drift()

    lines = [
        "## 📊 Live Monitoring Dashboard",
        "",
        "### Request Statistics",
        f"- **Total requests:** {len(request_log)}",
        f"- **Avg latency:** {sum(lats)/len(lats):.0f} ms",
        f"- **P95 latency:** {sorted(lats)[int(len(lats)*0.95)]:.0f} ms",
        f"- **Avg confidence:** {sum(confs)/len(confs):.1f}%",
        "",
        "### Top Predicted Specialties",
    ]
    for spec, cnt in top:
        pct = round(cnt / len(request_log) * 100, 1)
        lines.append(f"- {spec}: {cnt} requests ({pct}%)")

    lines += [
        "",
        "### Feedback",
        feedback_str,
        "",
        "### 🔍 Drift Detection",
        f"**Method:** Jensen-Shannon divergence vs training distribution",
        f"**Threshold:** {DRIFT_THRESHOLD}",
        f"**Status:** {drift['message']}",
        f"**Requests analysed:** {min(len(request_log), 50)} (last 50)",
    ]

    if drift["detected"]:
        lines += [
            "",
            "> ⚠️ **Action required:** The incoming note distribution differs",
            "> significantly from training data. Consider retraining the model",
            "> with new data that reflects this distribution shift.",
        ]

    return "\n".join(lines)


# ── Sample notes ──────────────────────────────────────────────
SAMPLES = {
    "Cardiology": (
        "Patient is a 58-year-old male presenting with substernal chest pain "
        "radiating to the left arm. EKG shows ST elevation in leads II, III, aVF. "
        "Troponin I elevated at 3.2. Started on aspirin and heparin drip."
    ),
    "Orthopedic": (
        "28-year-old female presents after twisting her right knee during soccer. "
        "Positive Lachman test and anterior drawer sign. MRI confirms complete ACL "
        "tear with lateral femoral condyle bone bruising."
    ),
    "Neurology": (
        "75-year-old male with sudden onset left-sided weakness and facial droop. "
        "NIHSS score 14. CT head negative for hemorrhage. MRI shows acute infarct "
        "in right MCA territory. Outside tPA window, admitted to stroke unit."
    ),
    "Gastroenterology": (
        "45-year-old female with 3-week history of epigastric pain and nausea. "
        "H. pylori breath test positive. Endoscopy shows 1.2cm gastric ulcer. "
        "Started on triple therapy. NSAIDs discontinued."
    ),
}


# ── UI ────────────────────────────────────────────────────────
with gr.Blocks(title="MedPrompt", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        "# 🏥 MedPrompt\n"
        "### Clinical Note Classification & Summarization\n"
        "`distilbert-base-uncased` · MTSamples · 15 Medical Specialties"
    )

    # ── Tab 1: Inference ──────────────────────────────────────
    with gr.Tab("🔬 Inference"):
        with gr.Row():
            with gr.Column(scale=2):
                note_in   = gr.Textbox(
                    label="Clinical Note",
                    lines=10,
                    placeholder="Paste a medical transcription here..."
                )
                sample_dd = gr.Dropdown(
                    choices=list(SAMPLES.keys()),
                    label="Or load a sample note"
                )
                with gr.Row():
                    load_btn = gr.Button("Load sample", variant="secondary")
                    run_btn  = gr.Button("▶  Analyse note", variant="primary")

            with gr.Column(scale=1):
                spec_out = gr.Textbox(label="Predicted Specialty", interactive=False)
                conf_out = gr.Textbox(label="Confidence",           interactive=False)
                lat_out  = gr.Textbox(label="Inference Latency",    interactive=False)
                sum_out  = gr.Textbox(label="Extractive Summary",   interactive=False, lines=5)

                feedback_radio = gr.Radio(
                    choices=["👍 Correct", "👎 Incorrect"],
                    label="Was this prediction correct?",
                    value=None,
                )
                gr.Markdown("*Feedback is used for drift monitoring.*")

        load_btn.click(
            fn=lambda c: SAMPLES.get(c, ""),
            inputs=sample_dd,
            outputs=note_in,
        )
        run_btn.click(
            fn=run_inference,
            inputs=note_in,
            outputs=[spec_out, conf_out, sum_out, lat_out],
        )
        feedback_radio.change(
            fn=record_feedback,
            inputs=[spec_out, feedback_radio],
            outputs=[],
        )

    # ── Tab 2: Monitoring + Drift ─────────────────────────────
    with gr.Tab("📊 Monitoring & Drift Detection"):
        gr.Markdown(
            "This dashboard tracks live inference metrics and detects "
            "**data drift** by comparing the incoming note distribution "
            "against the training set distribution using "
            "**Jensen-Shannon divergence**."
        )

        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh dashboard", variant="primary")

        monitor_out = gr.Markdown("Make some predictions in Tab 1 first.")
        refresh_btn.click(fn=get_monitoring_stats, outputs=monitor_out)

        gr.Markdown("""
---
### How drift detection works
1. Every prediction is logged with its predicted specialty
2. The last 50 predictions form the **observed distribution**
3. This is compared to the **training distribution** (MTSamples top-15 frequencies)
4. **Jensen-Shannon divergence** measures how different they are
   - JS = 0.0 → identical distributions, no drift
   - JS > 0.3 → significant drift, retraining recommended
5. If drift is detected, a warning is shown above
""")

    # ── Tab 3: Metrics ────────────────────────────────────────
    with gr.Tab("📈 Model Metrics"):
        gr.Markdown("""
## Training Results & Ablation Studies

*(Replace the values below with your actual results after running notebooks 03 and 04)*

### Test Set Performance
| Metric | Value |
|--------|-------|
| Accuracy | *run notebook 03* |
| Macro F1 | *run notebook 03* |
| Weighted F1 | *run notebook 03* |
| ROUGE-L | *run notebook 03* |
| Cohen's Kappa | *run notebook 03* |

### Ablation Study Summary
| Experiment | Variable | Best Value | Best Metric |
|------------|----------|-----------|-------------|
| A — Dropout | 0.1 / 0.2 / 0.3 / 0.4 | *from MLflow* | *macro F1* |
| B — Data size | 25% / 50% / 75% / 100% | *from MLflow* | *macro F1* |
| C — Learning rate | 1e-5 / 3e-5 / 5e-5 / 1e-4 | *from MLflow* | *macro F1* |
| D — Seq length | 64 / 128 / 256 | *from MLflow* | *F1 + latency* |

*(Add your MLflow screenshot images here)*
""")

    # ── Tab 4: About ──────────────────────────────────────────
    with gr.Tab("ℹ️ About"):
        gr.Markdown(f"""
## About MedPrompt

**Base model:** distilbert-base-uncased (66M parameters, runs on CPU)
**Task 1:** Medical specialty classification — 15 classes
**Task 2:** Extractive clinical summarization
**Dataset:** MTSamples — 4,999 medical transcriptions

### Model Architecture
```
Clinical Note
     ↓
DistilBERT encoder (768-d hidden states)
     ↓                    ↓
[CLS] token          All token states
     ↓                    ↓
MedPromptHead     ExtractiveSummaryHead
(15 specialties)  (token importance scores)
```

### Design Choices
| Component | Choice | Why |
|-----------|--------|-----|
| Model | DistilBERT | 97% of BERT quality, CPU-friendly |
| Head | 2-layer MLP | Prevents abrupt 768→15 collapse |
| Activation | GELU | Matches DistilBERT FFN internals |
| Normalisation | LayerNorm | Stabilises activations |
| Loss | CrossEntropy + label smoothing | Handles class imbalance |
| Summarization | Extractive token scoring | No decoder, fast on CPU |
| Drift detection | Jensen-Shannon divergence | Symmetric, bounded 0-1 |

### MLOps Pipeline
```
Code push → GitHub Actions CI/CD
         → Auto-retrain (if metric passes threshold)
         → Auto-deploy to HF Hub
         → Gradio app loads new weights
         → Drift monitoring in production
```

[GitHub Repository](https://github.com/your-username/medprompt-mlops)
""")

# ── Launch ────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
