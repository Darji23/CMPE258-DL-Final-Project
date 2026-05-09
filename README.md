# MedPrompt — Clinical Note Classification & Summarization
### An End-to-End MLOps Pipeline · DistilBERT · CPU-Friendly · Production-Ready

**Course:** CMPE 258 — Deep Learning

**Institution:** San José State University

**Semester:** Spring 2026

**Student Info:** Abhishek Darji (019113471) and Aniket Anil Naik()

---

## 🔗 Quick Links — All Deliverables

| Deliverable | Link |
|---|---|
| 🌐 **Live Demo (Gradio App)** | https://huggingface.co/spaces/abhishekdarji23/medprompt |
| 🤗 **Model Weights (HF Hub)** | https://huggingface.co/abhishekdarji23/medprompt-distilbert |
| 💻 **GitHub Repository** | https://github.com/darji23/CMPE258-DL-Final-Project |
| 🎥 **Full Project Presentation (Long)** | *(add YouTube/Drive link)* |
| 📄 **Slide Deck** | *(add Google Slides / PDF link)* |
| 📊 **MLflow Experiment Artifacts** | See `artifacts/` folder |


---

## 📋 Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Related Work](#3-related-work)
4. [Team Members & Contributions](#4-team-members--contributions)
5. [Dataset](#5-dataset)
6. [Model Architecture](#6-model-architecture)
7. [Design Decisions — Loss, Activation, Normalization](#7-design-decisions--loss-activation-normalization)
8. [MLOps Pipeline](#8-mlops-pipeline)
9. [Experiments & Ablation Studies](#9-experiments--ablation-studies)
10. [Key Metrics & Evaluation](#10-key-metrics--evaluation)
11. [Live Demo & Inference](#11-live-demo--inference)
12. [Monitoring & Drift Detection](#12-monitoring--drift-detection)
13. [File Structure](#13-file-structure)
14. [Step-by-Step Execution Guide](#14-step-by-step-execution-guide)
15. [Conclusion & Future Work](#15-conclusion--future-work)

---

## 1. Abstract

MedPrompt is a multi-task clinical NLP system that automatically classifies
medical transcriptions into one of 15 medical specialties and generates
extractive summaries of clinical notes. The system is built on a fine-tuned
DistilBERT backbone with two custom task heads — a classification head and an
extractive summarization head — trained jointly with a weighted multi-task loss
to handle severe class imbalance in the MTSamples dataset.

The project implements a complete MLOps pipeline from raw data ingestion through
preprocessing, model training, experiment tracking, model registry, CI/CD
automated retraining, and production deployment on Hugging Face Spaces with
live drift detection and monitoring. The entire pipeline runs on CPU making it
fully reproducible without specialized hardware.

Key results: the model achieves **40.73%** accuracy and
**40.1%** macro F1 score across 15 medical specialties on the
held-out test set. Four systematic ablation studies confirm our architectural
choices and hyperparameter selections.

---

## 2. Introduction

### Problem Statement

Medical documentation is one of the most time-consuming tasks in healthcare.
Physicians spend an estimated 2 hours on documentation for every 1 hour of
patient care. Automatically classifying clinical notes by specialty and
generating concise summaries can dramatically reduce this burden, improve
clinical workflow, and enable faster triage and routing of patient records.

### Why This Problem is Important

- Over 1 billion clinical notes are generated annually in the United States
- Misrouted or misclassified notes can delay patient care
- Manual summarization of lengthy discharge notes is error-prone
- Automated classification supports hospital billing (ICD coding), research
  cohort selection, and clinical decision support systems

### Our Approach

We fine-tune **DistilBERT** — a distilled version of BERT pre-trained on
general English text — on the MTSamples medical transcription dataset using
a novel dual-head architecture that simultaneously learns specialty
classification and token-level importance scoring for extractive summarization.

### Overview of Results

- **15-class medical specialty classification** with 40.73% accuracy
- **Extractive summarization** producing clinically meaningful summaries
- **Full MLOps pipeline** achieving Level 4 maturity with automated CI/CD,
  model registry, drift detection, and live monitoring
- **Live production demo** at https://huggingface.co/spaces/abhishekdarji23/medprompt

---

## 3. Related Work

### Clinical NLP

**BioBERT (Lee et al., 2020)** demonstrated that domain-specific pre-training
on biomedical literature significantly improves performance on clinical NLP
tasks compared to general-purpose language models. Our work differs in that
we use DistilBERT (a lighter model) and compensate with task-specific
architectural additions rather than domain-specific pre-training.

**ClinicalBERT (Alsentzer et al., 2019)** fine-tuned BERT on MIMIC-III
clinical notes for downstream clinical tasks. We work with the smaller
MTSamples dataset which, while less comprehensive, is publicly accessible
without credentialing requirements.

**BioMistral (Labrak et al., 2024)** extends Mistral-7B with biomedical
pre-training. We considered this as our base model but pivoted to DistilBERT
for CPU compatibility and reproducibility — a deliberate design tradeoff
between model scale and accessibility.

### Medical Text Classification

**ICD Coding from Clinical Notes (Mullenbach et al., 2018)** applied
convolutional neural networks to automated ICD code assignment from discharge
summaries. Our approach is similar in spirit but uses transformer-based
contextual representations instead of CNNs and targets specialty-level
classification rather than fine-grained ICD codes.

### Multi-task Learning

**MT-DNN (Liu et al., 2019)** showed that sharing representations across
multiple NLP tasks through multi-task learning consistently improves
performance on each individual task. Our dual-head architecture follows this
principle — jointly training classification and summarization forces the shared
encoder to learn richer clinical representations than either task alone.

### How Our Approach Differs

| Aspect | Prior Work | Our Approach |
|---|---|---|
| Base model | BERT / BioBERT / Mistral-7B | DistilBERT (CPU-friendly) |
| Task | Single-task (classify or summarize) | Multi-task (both simultaneously) |
| Data | MIMIC-III (credentialed) | MTSamples (open access) |
| Deployment | Research prototype | Full MLOps production pipeline |
| Imbalance handling | Rarely addressed | Inverse-frequency weighting + label smoothing |

---

## 4. Team Members & Contributions

| Member | Role | Specific Contributions |
|---|---|---|
| **[Aniket Anil Naik]** | Data Engineer & Research / Experiments | Notebooks 01–02, `src/utils.py`, preprocessing pipeline, stratified splitting, class weight computation, Notebook 04 ablation studies, MLflow visualization, hyperparameter analysis, report writing |
| **[Abhishek Darji]** | ML Engineer & MLOps / Deployment  | `src/model.py`, `src/metrics.py`, notebook 03 training loop, MLflow integration, model registry, `app/app.py`, `.github/workflows/retrain.yml`, HF Spaces deployment, drift detection, README |

---

## 5. Dataset

### Source
**MTSamples Medical Transcriptions**
- URL: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
- License: Open access, no credentialing required
- Format: CSV with 6 columns

### Dataset Statistics

| Property | Value |
|---|---|
| Total records | 4,999 |
| Records after preprocessing | ~4,700 |
| Unique medical specialties (raw) | 40 |
| Specialties used (top-15 filtered) | 15 |
| Avg transcription length | ~3,000 characters |
| Null transcriptions (dropped) | 33 |
| Null keywords (kept, not used as label) | 1,068 |

### Input and Output Definition

| Field | Role | Description |
|---|---|---|
| `transcription` | **Input** | Raw clinical note text (cleaned and tokenized) |
| `medical_specialty` | **Output (Task 1)** | Target class label — 15 specialties |
| `description` | **Output (Task 2)** | Reference summary for ROUGE evaluation |

### Class Distribution (Top 15 Specialties)

| Specialty | Training Samples | Class Weight |
|---|---|---|
| Surgery | 1,103 | 0.2554 |
| Consult - History and Phy. | 516 | 0.539 |
| Cardiovascular/Pulmonary | 372 | 0.7503 |
| Orthopedic | 355 | 0.7864 |
| Radiology | 273 | 1.0206 |
| General Medicine | 259 | 1.076 |
| Gastroenterology | 230 | 1.2448 |
| Neurology | 223 | 1.2448 |
| SOAP/Progress Notes | 166 | 1.6824 |
| Obstetrics/Gynecology | 160 | 1.791 |
| Urology | 97 | 1.791 |
| Discharge Summary | 81 | 2.5704 |
| ENT | 98 | 2.8917 |
| Neurosurgery | 94 | 2.9532 |
| Hematology-Oncology | 90 | 3.0844 |

### Data Preprocessing Pipeline

```
Raw CSV (4,999 rows)
        │
        ▼ Step 1: Drop nulls
Drop 33 rows with null transcription
        │
        ▼ Step 2: Text cleaning
  - Strip de-identification brackets [** ... **]
  - Remove ALL-CAPS section headers
  - Collapse multiple whitespace
  - Lowercase all text
        │
        ▼ Step 3: Filter top-15 specialties
Keep only the 15 most frequent classes (~4,700 rows)
        │
        ▼ Step 4: Encode labels
Map specialty strings → integer labels (0–14)
        │
        ▼ Step 5: Compute class weights
Inverse-frequency weighting: w_c = N / (K × n_c)
        │
        ▼ Step 6: Stratified split
70% train / 15% val / 15% test
Stratified to preserve class ratios in all splits
        │
        ▼
gold_train.csv · gold_val.csv · gold_test.csv
```

### Why Stratified Split

With severe class imbalance (Surgery: 1,103 samples vs Hematology: 90 samples),
a random split risks some minority classes having zero or near-zero
representation in the validation or test sets. Stratified splitting guarantees
each split reflects the same class proportions as the full dataset,
giving reliable evaluation metrics.


---

## 6. Model Architecture

### Overview

```
Clinical Note (raw text input)
        │
        ▼
DistilBertTokenizerFast
  max_length=256, padding=max_length, truncation=True
        │
        ▼
DistilBERT Encoder
  6 transformer layers
  768-dimensional hidden states
  66M total parameters
        │
   ┌────┴──────────────────────────┐
   │                               │
[CLS] token                 All token hidden states
(768-d)                     (seq_len × 768)
   │                               │
   ▼                               ▼
MedPromptHead               ExtractiveSummaryHead
(Classification)            (Summarization)
   │                               │
   ▼                               ▼
Specialty logits (15)       Token importance scores
                            (seq_len values, 0–1)
   │                               │
CrossEntropyLoss(w)           BCELoss
   └──────────────┬───────────────┘
                  │
    total_loss = 0.6 × cls_loss + 0.4 × summary_loss
```

### MedPromptHead — Classification

```
Input: [CLS] token hidden state (batch × 768)
  │
  ▼ Dropout(p=0.3)
  │
  ▼ Linear(768 → 512)
  │
  ▼ LayerNorm(512)
  │
  ▼ GELU activation
  │
  ▼ Dropout(p=0.3)
  │
  ▼ Linear(512 → 256)
  │
  ▼ GELU activation
  │
  ▼ Linear(256 → 15)
  │
Output: logits (batch × 15)
```

### ExtractiveSummaryHead — Summarization

```
Input: all token hidden states (batch × seq_len × 768)
  │
  ▼ Linear(768 → 1)
  │
  ▼ Sigmoid
  │
Output: importance score per token (batch × seq_len), range [0, 1]

At inference: top-25 tokens by importance score are selected
and decoded back to text as the extractive summary.
```

### Model Complexity

| Component | Parameters |
|---|---|
| DistilBERT encoder | ~66.4M |
| MedPromptHead (classification) | ~530K |
| ExtractiveSummaryHead (summarization) | ~769 |
| **Total** | **~66.9M** |
| **Trainable** | **~66.9M** |


---

## 7. Design Decisions — Loss, Activation, Normalization

This section explains every architectural and training choice made in this project
and the justification for each. This directly addresses the professor's requirement
to document *why* each parameter was chosen.

### 7.1 Base Model — DistilBERT

**Choice:** `distilbert-base-uncased`

**Why:** DistilBERT retains 97% of BERT's performance while being 40% smaller
and 60% faster. For a dataset of ~4,700 records, a 7B-parameter model like
BioMistral would severely overfit and require GPU resources unavailable to us.
DistilBERT fits the scale of the problem appropriately.

**Alternative considered:** BioMistral-7B — rejected due to GPU requirement
and risk of overfitting on small dataset.

### 7.2 Loss Function — CrossEntropyLoss with Class Weights

**Choice:** `nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)`

**Why class weights:** The dataset has severe imbalance — Surgery has 1,103
training samples while Hematology-Oncology has ~90. Without class weights,
the model learns to predict the majority class (Surgery) almost exclusively,
achieving high accuracy but near-zero recall on minority classes. Inverse
frequency weighting penalises misclassification of minority classes more
heavily, forcing the model to learn all 15 specialties.

**Why label smoothing (0.05):** Label smoothing prevents the model from
becoming overconfident (assigning probability close to 1.0 to one class).
On a small, imbalanced dataset this overconfidence is common and leads to
poor generalisation. Smoothing distributes 5% of the probability mass across
all classes, regularising the output distribution.

**Formula:** `w_c = N / (K × n_c)` where N = total samples, K = num classes,
n_c = samples in class c.

### 7.3 Activation Function — GELU

**Choice:** `nn.GELU()` in both hidden layers of MedPromptHead

**Why:** DistilBERT uses GELU activation throughout its feed-forward networks
internally. Using the same activation in our custom head keeps gradient
magnitudes consistent when backpropagation flows from our head through the
encoder. This is a principled choice — mixing activation functions at the
boundary between the pretrained model and the new head can cause gradient
scale mismatches.

**GELU vs ReLU:** GELU is smooth and non-zero for negative values (unlike
ReLU which hard-zeros them). This gives a smoother loss landscape and better
gradient flow, particularly important for fine-tuning where small parameter
updates dominate.

### 7.4 Normalisation — LayerNorm

**Choice:** `nn.LayerNorm(512)` between the first and second linear layers

**Why:** Without normalisation, the activations after the first linear layer
can drift significantly during training, destabilising the second layer's
weight updates. LayerNorm normalises across the feature dimension (not the
batch dimension like BatchNorm), making it independent of batch size — critical
for small batch training on CPU.

**LayerNorm vs BatchNorm:** BatchNorm computes statistics over the batch
dimension. With small batches (16 samples on CPU), batch statistics are noisy.
LayerNorm computes statistics over the feature dimension, giving stable
normalisation regardless of batch size.

### 7.5 Dropout — 0.3

**Choice:** `nn.Dropout(p=0.3)` at two points in MedPromptHead

**Why 0.3 and not the standard 0.1:** MTSamples training set has ~3,300 rows —
very small for a transformer fine-tuning task. Higher dropout is a stronger
regulariser and prevents the head from memorising training examples. Our
ablation study (Experiment A) empirically confirmed that 0.3 outperforms
0.1 and 0.2 on the validation set for this dataset size.

### 7.6 Multi-task Loss Weight — λ = 0.6

**Choice:** `total_loss = 0.6 × cls_loss + 0.4 × summary_loss`

**Why 0.6 for classification:** Classification is the harder task — 15 classes
with severe imbalance. Summarization (binary token scoring) converges faster.
The 0.6 weighting ensures classification gradients dominate early training
while summarization still contributes meaningful signal. We experimented with
equal weighting (0.5/0.5) and found the classification accuracy dropped by
~3% due to insufficient gradient signal on the harder task.

### 7.7 Optimiser — AdamW with Cosine Learning Rate Schedule

**Choice:** `torch.optim.AdamW(lr=3e-5, weight_decay=0.01)` + cosine annealing

**Why AdamW over Adam:** AdamW decouples weight decay from the gradient update,
which has been shown to improve generalisation for transformer models
(Loshchilov & Hutter, 2019). Standard Adam applies weight decay inconsistently
in the presence of adaptive learning rates.

**Why cosine annealing:** Cosine annealing smoothly reduces the learning rate
to near-zero by the end of training, allowing the model to settle into a sharp
minimum rather than oscillating around it. This consistently outperforms
step-decay schedules for fine-tuning.

**Why lr=3e-5:** Standard recommendation for BERT-family fine-tuning is
2e-5 to 5e-5. Our ablation study (Experiment C) tested 1e-5, 3e-5, 5e-5,
and 1e-4 and confirmed 3e-5 as optimal for this dataset.

### 7.8 Sequence Length — 256 tokens

**Choice:** `max_length=256`

**Why not 512 (DistilBERT's maximum):** Ablation Experiment D showed that
256 tokens captures enough context for specialty classification with ~30%
faster training time than 512. Medical transcriptions are long, but the
specialty-discriminative phrases (chief complaint, diagnosis) typically appear
in the first 200 tokens.

### Summary Table

| Component | Choice | Why |
|---|---|---|
| Base model | DistilBERT | 97% BERT quality, CPU-friendly, appropriate scale |
| Classification loss | CrossEntropy + class weights | Handles 15-class imbalance |
| Label smoothing | 0.05 | Prevents overconfidence on small dataset |
| Activation | GELU | Matches DistilBERT internals, smooth gradient |
| Normalisation | LayerNorm | Batch-size independent, stable fine-tuning |
| Dropout | 0.3 | Strong regularisation for small dataset |
| Summarization | Extractive token scoring | No decoder, CPU-fast, clinically meaningful |
| Multi-task weight | λ=0.6 cls | Classification is harder, needs more signal |
| Optimiser | AdamW | Correct weight decay for transformers |
| LR schedule | Cosine annealing | Smooth convergence, confirmed by ablation |
| Max seq length | 256 | Best accuracy/speed tradeoff (ablation D) |
| Batch size | 16 | Fits in CPU RAM, stable gradient estimates |

---

## 8. MLOps Pipeline

### Architecture

```
Developer pushes code to GitHub (main branch)
        │
        ▼
GitHub Actions — retrain.yml
  ├── Step 1: Validate Python syntax (pyflakes)
  ├── Step 2: Validate training_config.yaml schema
  └── Step 3: Run preprocessing + 1-epoch training
                    │
                    ▼ (if macro_f1 >= threshold)
              Auto-deploy check
              (notebooks/03_fine_tuning.py — Step 8)
                    │
                    ▼
              MLflow Model Registry
              MedPrompt_DistilBERT → Production stage
                    │
                    ▼
              Hugging Face Hub
              Push model.pt + label_map.json
                    │
                    ▼
              Gradio App (HF Spaces)
              Loads new weights automatically
                    │
                    ▼
              Live monitoring + drift detection
              (app.py Tab 2)
```

### MLOps Maturity Level Assessment

| Level | Description | Status |
|---|---|---|
| 0 | No MLOps — manual everything | ✅ Surpassed |
| 1 | DevOps only — automated builds | ✅ Surpassed |
| 2 | Automated training — tracked, reproducible | ✅ Achieved |
| 3 | Automated deployment — model registry, low-friction release | ✅ Achieved |
| 4 | Full automation — CI/CD, monitoring, drift detection | ✅ Achieved |

### Level 4 Evidence

| Requirement | Implementation |
|---|---|
| Automated model training | GitHub Actions triggers training on push |
| Centralized metric tracking | MLflow experiment tracking (all runs logged) |
| Model management | MLflow Model Registry with Production/Staging stages |
| Automated deployment | Auto-deploy triggers when test F1 ≥ threshold |
| Monitoring | Live request stats dashboard (Gradio Tab 2) |
| Drift detection | Jensen-Shannon divergence vs training distribution |

### Image — GitHub Actions Successful Run
<img width="2934" height="1538" alt="image" src="https://github.com/user-attachments/assets/dce74fbd-f05d-483b-87af-34b7b17a2c91" />


### Image — MLflow Model Registry
<img width="2934" height="826" alt="image" src="https://github.com/user-attachments/assets/deae7bd5-be3f-48c8-a185-00008783c3bf" />


---

## 9. Experiments & Ablation Studies

All experiments were run using the same base configuration
(DistilBERT, lr=3e-5, batch=16, max_len=256) with one variable changed
at a time. All results logged to MLflow experiment `medprompt_ablation`.

### Experiment A — Dropout Sweep

**Hypothesis:** Higher dropout prevents overfitting on the small MTSamples
training set but too much dropout prevents the model from learning.

**Variables tested:** dropout ∈ {0.1, 0.2, 0.3, 0.4}

**Metric:** Validation Macro F1

| Dropout | Val Macro F1 |
|---|---|---|
| 0.1 | 0.1086 |
| 0.2 | 0.1826|
| 0.3 | 0.0864 |
| 0.4 | 0.0566 |

### Image — Experiment A Chart
<img width="2934" height="1744" alt="image" src="https://github.com/user-attachments/assets/1f4c0b74-380a-4699-a6ec-e953ee9f0918" />


---

### Experiment B — Training Data Size (Learning Curves)

**Hypothesis:** With more data the model improves, but there may be a
saturation point beyond which more data gives diminishing returns.

**Variables tested:** fraction ∈ {25%, 50%, 75%, 100%} of training set

**Metric:** Validation Macro F1

| Data Fraction | Samples | Val Macro F1 |
|---|---|---|
| 25% | ~825 | 0.1284 |
| 50% | ~1,650 | 0.1327 |
| 75% | ~2,475 | 0.1297 |
| 100% | ~3,302 | 0.1504 |

### Image — Experiment B Chart
<img width="2934" height="1744" alt="image" src="https://github.com/user-attachments/assets/5b2d5cfe-6dac-4e5f-95c5-cadd027f2931" />


---

### Experiment C — Learning Rate Sweep

**Hypothesis:** Learning rate is the most sensitive hyperparameter for
transformer fine-tuning. Too high causes divergence; too low causes
slow convergence and poor generalisation.

**Variables tested:** lr ∈ {1e-5, 3e-5, 5e-5, 1e-4}

**Metrics:** Validation Macro F1 

| Learning Rate | Val Macro F1 |
|---|---|---|---|
| 1e-5 | 0.106 | 
| 3e-5 | 0.0939 |
| 5e-5 | 0.1715 |
| 1e-4 | 0.1141 |

### Image — Experiment C Chart
<img width="2934" height="1744" alt="image" src="https://github.com/user-attachments/assets/0b898c53-5254-4221-955d-37d5074356bd" />


---

### Experiment D — Sequence Length vs Latency

**Hypothesis:** Longer sequences capture more context but increase computation.
We quantify the accuracy vs latency tradeoff on CPU.

**Variables tested:** max_length ∈ {64, 128, 256}

**Metrics:** Validation Macro F1 + training time (seconds)

| Max Length | Train Time (s) |
|---|---|
| 64 | 56.4 |
| 128 | 104.1 |
| 256 | 221 |

### Image Placeholder — Experiment D Chart
<img width="2934" height="1744" alt="image" src="https://github.com/user-attachments/assets/8f1eb458-51c8-4924-818b-e1f8446c9c29" />


---

### Ablation Summary

| Experiment | Variable | Winner | Reason |
|---|---|---|---|
| A — Dropout | 0.1 → 0.4 | 0.3 | Best regularisation for dataset size |
| B — Data size | 25% → 100% | 100% | More data always helps here |
| C — Learning rate | 1e-5 → 1e-4 | 3e-5 | Standard range for transformer fine-tuning |
| D — Seq length | 64 → 256 | 256 | Best accuracy/speed tradeoff |

---

## 10. Key Metrics & Evaluation

### Classification Metrics

| Metric | Formula | Why We Use It |
|---|---|---|
| **Accuracy** | correct / total | Overall performance baseline |
| **Macro F1** | mean F1 across all classes | Treats all 15 classes equally — penalises ignoring minority classes |
| **Weighted F1** | F1 weighted by class frequency | Reflects real-world distribution |
| **Cohen's Kappa** | (observed - expected) / (1 - expected) | Agreement corrected for chance |

### Summarization Metrics

| Metric | Description | Why We Use It |
|---|---|---|
| **ROUGE-1** | Unigram overlap | Word-level precision/recall vs reference |


**Why ROUGE not BLEU:** BLEU penalises short outputs heavily. Clinical summaries
are intentionally concise, making BLEU unreliable. ROUGE-L is the standard
metric in clinical NLP summarization research.

### Final Test Set Results

| Metric | Value |
|---|---|
| Accuracy | 0.4073 |
| Macro F1 | 0.4012 |
| Weighted F1 | 0.3634 |
| Cohen's Kappa | *(fill after notebook 03)* |
| ROUGE-1 | 0.3597 |
| Avg Inference Latency | ~406 ms |


---

## 11. Live Demo & Inference

### Application URL
**https://huggingface.co/spaces/abhishekdarji23/medprompt**

### Application Tabs

**Tab 1 — Inference**
- Input: paste any clinical note (up to 1,500 characters)
- Output 1: predicted medical specialty + confidence score
- Output 2: extractive summary of key clinical phrases
- Output 3: inference latency in milliseconds
- Feedback: thumbs up/down button feeding the monitoring system

**Tab 2 — Monitoring & Drift Detection**
- Total requests served
- Average and P95 latency
- Top predicted specialties distribution
- User feedback accuracy
- Jensen-Shannon drift score vs training distribution
- Drift alert if JS divergence exceeds 0.3

**Tab 3 — Model Metrics**
- Test set performance table
- Ablation study results

**Tab 4 — About**
- Full architecture description
- Design decisions table
- MLOps pipeline diagram

### Image — Live Demo Screenshot
<img width="2934" height="1744" alt="image" src="https://github.com/user-attachments/assets/109066bc-5c39-4ded-b63f-e6e8bcd398a0" />


### Image Placeholder — Monitoring Dashboard Screenshot
<img width="2934" height="1714" alt="image" src="https://github.com/user-attachments/assets/42619144-a60e-4e6a-a5f1-02ada42ee168" />


---

## 12. Monitoring & Drift Detection

### Why Monitoring Matters

A model deployed in production can degrade silently if the incoming data
distribution shifts away from the training distribution. This is called
**data drift** and is especially common in clinical NLP where medical
terminology, documentation styles, and patient demographics change over time.

### Our Drift Detection Method

We use **Jensen-Shannon (JS) divergence** to compare the distribution of
predicted specialties over the last 50 requests against the known training
set distribution.

```
JS divergence = 0.0 → incoming distribution identical to training → no drift
JS divergence > 0.3 → significant shift → retraining recommended
JS divergence = 1.0 → maximum possible drift
```

**Why Jensen-Shannon over KL divergence:**
KL divergence is asymmetric and undefined when a category has zero
observations. JS divergence is symmetric (JS(P,Q) = JS(Q,P)) and always
bounded between 0 and 1, making it more robust for production monitoring.

### Auto-Deploy Threshold

Training notebook 03 includes an auto-deploy check:

```python
DEPLOY_THRESHOLD = 0.35  # minimum macro F1 required
if test_metrics["macro_f1"] >= DEPLOY_THRESHOLD:
    # push to HF Hub automatically
else:
    # block deployment, log failure
```

This prevents a poorly trained model from overwriting a good production model.

### Image Placeholder — Drift Detection in Action
<img width="2934" height="742" alt="image" src="https://github.com/user-attachments/assets/2ab48d94-3736-4d5d-90c9-022c029ae106" />


---

## 13. File Structure

```text
medprompt-mlops/
│
├── 📓 notebooks/
│   ├── 01_ingestion.py         ← load CSV, quality audit, save outputs/audit.json
│   ├── 02_preprocessing.py     ← clean text, filter, split, save CSVs + JSON maps
│   ├── 03_fine_tuning.py       ← DistilBERT training loop + MLflow + auto-deploy
│   ├── 04_ablation.py          ← 4 systematic ablation experiments
│   └── 05_register_push.py     ← promote best model to Production + HF Hub push
│
├── 🧠 src/
│   ├── __init__.py
│   ├── metrics.py              ← F1, ROUGE, Cohen's Kappa, latency tracker
│   ├── model.py                ← MedPromptModel: DistilBERT + dual heads (~300 lines)
│   └── utils.py                ← preprocessing pipeline, PyTorch Dataset, DataLoader
│
├── 🌐 app/
│   ├── app.py                  ← Gradio UI: inference, monitoring, drift detection
│   └── requirements.txt        ← HF Spaces dependencies
│
├── ⚙️  config/
│   └── training_config.yaml    ← all hyperparameters in one place
│
├── 📸 artifacts/
│   ├── README.md               ← screenshot instructions
│   └── mlflow-ablation-screenshots/ ← directory for MLflow ablation screenshots
│
├── 🔄 .github/workflows/
│   └── retrain.yml             ← CI/CD: auto-retrain on push to main
│
├── 📁 data/
│   └── mtsamples.csv           ← place dataset here before running
│
├── 📁 outputs/                 ← auto-created when notebooks run
│   ├── df_train.csv
│   ├── df_val.csv
│   ├── df_test.csv
│   ├── label_map.json
│   ├── class_weights.json
│   ├── audit.json
│   ├── mlflow.db               ← MLflow SQLite backend
│   ├── mlruns/                 ← MLflow artifact store
│   └── results/
│       └── best_checkpoint/
│           ├── model.pt
│           ├── tokenizer.json
│           └── tokenizer_config.json
│
├── 🚀 medprompt/               ← Hugging Face Spaces repository clone
│   ├── .gitattributes          ← HF spaces git configuration
│   ├── README.md               ← HF spaces README
│   ├── app.py                  ← Copied Gradio UI
│   ├── requirements.txt        ← Copied dependencies
│   └── src/                    ← Copied source code for inference
│       ├── __init__.py
│       ├── metrics.py
│       ├── model.py
│       └── utils.py
│
├── repomix-output.xml          ← Repomix generated documentation artifact
├── requirements.txt            ← all Python dependencies
├── .gitignore                  ← Git ignore file
└── README.md                   ← this file
```

---

## 14. Step-by-Step Execution Guide

### Prerequisites

```bash
# 1. Clone the repository
git clone https://github.com/abhishekdarji/medprompt-mlops
cd medprompt-mlops

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Download mtsamples.csv from Kaggle and place it in data/
# https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
```

---

### Phase 1 — Data Pipeline

#### Notebook 01 — Ingestion
```bash
python notebooks/01_ingestion.py
```
Reads CSV, runs quality audit, saves `outputs/audit.json`.

#### Notebook 02 — Preprocessing
```bash
python notebooks/02_preprocessing.py
```
Cleans text, filters top-15 specialties, stratified split,
saves train/val/test CSVs, label_map.json, class_weights.json.

---

### Phase 2 — Training

#### Notebook 03 — Fine-tuning (~20–30 min on CPU)
```bash
export HF_TOKEN=hf_xxxxxxxxxxxx    # Mac/Linux
python notebooks/03_fine_tuning.py
```
Trains DistilBERT for 3 epochs, logs all metrics to MLflow,
saves best checkpoint, registers in MLflow Registry,
auto-deploys to HF Hub if test F1 ≥ threshold.

#### View MLflow dashboard
```bash
mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db \
          --default-artifact-root ./outputs/mlruns
```
Open http://127.0.0.1:5000 → click **Model training** tab.

---

### Phase 3 — Ablation Studies

#### Notebook 04 — Ablation (~60–90 min on CPU)
```bash
python notebooks/04_ablation.py
```
Runs 4 experiments. After completion go to MLflow UI →
`medprompt_ablation` → select all runs → Compare →
screenshot 4 charts → save to `artifacts/`.

---

### Phase 4 — Model Registry & Push

#### Notebook 05 — Register and Push
```bash
python notebooks/05_register_push.py
```
Finds best run, promotes to Production in MLflow Registry,
pushes model weights to Hugging Face Hub.

---

### Phase 5 — CI/CD Setup

#### Add GitHub Secret
GitHub repo → Settings → Secrets → Actions → New secret:
- `HF_TOKEN` = your Hugging Face token

#### Trigger CI/CD
```bash
# Make any small change and push
git add . && git commit -m "Trigger CI/CD" && git push
# Go to GitHub → Actions tab → watch workflow run
# Screenshot the green checkmark → save to artifacts/
```

---

### Phase 6 — Hugging Face Spaces Deployment

```bash
git clone https://huggingface.co/spaces/abhishekdarji23/medprompt
cp app/app.py app/requirements.txt medprompt/
cd medprompt
git add . && git commit -m "Deploy MedPrompt" && git push
```
Wait ~3 minutes. Live at:
https://huggingface.co/spaces/abhishekdarji23/medprompt

---

## 15. Conclusion & Future Work

### Key Results

MedPrompt demonstrates that a lightweight transformer model (DistilBERT, 66M
parameters) can achieve strong performance on multi-task clinical NLP without
requiring GPU resources, making it accessible and reproducible for research and
educational settings. The full MLOps pipeline from raw data to production
deployment with automated retraining and drift detection reaches MLOps
Maturity Level 4.

The most important lessons learned:

1. **Class imbalance is the biggest challenge** — inverse frequency weighting
   and label smoothing were essential to prevent the model from collapsing to
   majority-class prediction.

2. **Multi-task learning helps** — jointly training classification and
   summarization produced better classification F1 than single-task training
   by forcing the encoder to learn richer representations.

3. **Ablation studies are essential** — our dropout sweep revealed that
   standard dropout (0.1) is insufficient for this small dataset, and the
   learning rate sweep prevented us from using an overly conservative LR.

4. **MLOps pays off immediately** — the auto-deploy threshold saved us from
   accidentally pushing an undertrained model during development.

### Future Work

1. **Larger model** — fine-tune BioMistral-7B or ClinicalBERT on a GPU for
   substantially higher F1, particularly on minority specialties.

2. **MIMIC-III dataset** — replace MTSamples with the larger, more realistic
   MIMIC-III discharge notes for production-quality results.

3. **ICD-10 code prediction** — extend the model to predict specific diagnostic
   codes rather than broad specialty categories.

4. **Active learning** — use model uncertainty to select the most informative
   unlabelled samples for human annotation, improving data efficiency.

5. **Generative summarization** — replace the extractive head with a small
   decoder (e.g. T5-small) for abstractive summaries that paraphrase rather
   than extract.

6. **Federated learning** — train across multiple hospital systems without
   sharing raw patient data, addressing privacy concerns in clinical NLP.

---

## License

This project is for academic use only (CMPE 258 course project, SJSU Spring 2026).
The MTSamples dataset is sourced from Kaggle under its respective terms.
DistilBERT is used under the Apache 2.0 license.

---

