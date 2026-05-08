# =============================================================
# src/model.py — MedPrompt Model Architecture (DistilBERT)
# =============================================================
#
# WHY THESE CHOICES:
#   Base model   : distilbert-base-uncased
#                  40% smaller than BERT, 60% faster, retains 97%
#                  of BERT's performance. Runs on CPU — no GPU needed.
#   Hidden size  : 768-d (DistilBERT standard).
#   [CLS] token  : DistilBERT is a bidirectional encoder. The [CLS]
#                  token aggregates context from the full sequence,
#                  making it ideal for sequence-level classification.
#   Activation   : GELU — matches DistilBERT's internal FFN layers.
#                  Smoother gradient than ReLU near zero.
#   Normalisation: LayerNorm between classification layers — prevents
#                  logit drift and stabilises activations.
#   Loss         : CrossEntropyLoss with class weights + label
#                  smoothing 0.05. Handles severe class imbalance
#                  (Surgery=1103 vs Allergy/Immunology=7).
#   Summarization: Extractive — scores each token for importance and
#                  returns the top sentence. No decoder needed → fast
#                  on CPU and produces meaningful clinical summaries.
#   Multi-task   : total_loss = 0.6*cls_loss + 0.4*summary_loss
#                  Higher weight on classification because the 15-class
#                  imbalanced task is harder than extractive scoring.
# =============================================================

import torch
import torch.nn as nn
import json
from transformers import DistilBertModel, DistilBertTokenizerFast
from typing import Optional, Dict, Tuple


# ── Tokenizer ─────────────────────────────────────────────────
def load_tokenizer(model_id: str = "distilbert-base-uncased") -> DistilBertTokenizerFast:
    return DistilBertTokenizerFast.from_pretrained(model_id)


# ── Classification Head ───────────────────────────────────────
class MedPromptHead(nn.Module):
    """
    Two-layer MLP on top of DistilBERT [CLS] token.

    Architecture:
        [CLS] (768) → Dropout(0.3) → Linear(768→512)
                    → LayerNorm(512) → GELU
                    → Dropout(0.3)  → Linear(512→256)
                    → GELU          → Linear(256→num_classes)

    WHY two hidden layers:
        A single linear from 768→15 collapses the representation
        too abruptly. Two layers learn an intermediate
        medical-domain feature space before the final projection.

    WHY Dropout(0.3):
        MTSamples is small (~4700 rows). Higher dropout (0.3 vs
        the typical 0.1) is needed to prevent overfitting.
    """

    def __init__(self, hidden_size: int = 768, num_classes: int = 15):
        super().__init__()
        self.drop1 = nn.Dropout(0.3)
        self.fc1   = nn.Linear(hidden_size, 512)
        self.norm1 = nn.LayerNorm(512)
        self.act1  = nn.GELU()
        self.drop2 = nn.Dropout(0.3)
        self.fc2   = nn.Linear(512, 256)
        self.act2  = nn.GELU()
        self.out   = nn.Linear(256, num_classes)

    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        x = self.drop1(cls_hidden)
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = self.act2(x)
        return self.out(x)   # (batch, num_classes)


# ── Extractive Summary Head ───────────────────────────────────
class ExtractiveSummaryHead(nn.Module):
    """
    Scores each token for inclusion in the extractive summary.

    Architecture:
        token hidden states (batch, seq_len, 768)
          → Linear(768→1) → Sigmoid

    At inference time we pick the top-scoring tokens and
    reconstruct the most salient sentence from the input text.

    WHY extractive (not generative):
        Generative summarization requires a full autoregressive
        decoder and GPU. Extractive scoring is a single linear
        layer — fast on CPU while still producing clinically
        meaningful output by selecting important phrases.
    """

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.scorer = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, seq_len, 768)
        return torch.sigmoid(self.scorer(hidden_states).squeeze(-1))


# ── Full MedPrompt Model ──────────────────────────────────────
class MedPromptModel(nn.Module):
    """
    Full MedPrompt model — DistilBERT backbone + dual task heads.

    Forward pass returns:
        logits      : (batch, num_classes) classification scores
        token_scores: (batch, seq_len)     extractive summary scores
        cls_loss    : CrossEntropy with class weights (if cls_labels given)
        summary_loss: BCE on token scores  (if token_labels given)
        total_loss  : 0.6*cls + 0.4*summary
    """

    LAMBDA = 0.6   # weight of classification in combined loss

    def __init__(
        self,
        model_id:      str = "distilbert-base-uncased",
        num_classes:   int = 15,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.encoder      = DistilBertModel.from_pretrained(model_id)
        self.hidden_size  = self.encoder.config.dim   # 768
        self.cls_head     = MedPromptHead(self.hidden_size, num_classes)
        self.summary_head = ExtractiveSummaryHead(self.hidden_size)
        self.num_classes  = num_classes

        self.cls_criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.05,
        )
        self.summary_criterion = nn.BCELoss()

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        cls_labels:     Optional[torch.Tensor] = None,
        token_labels:   Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        outputs      = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden       = outputs.last_hidden_state   # (batch, seq_len, 768)
        cls_hidden   = hidden[:, 0, :]             # [CLS] token

        cls_logits   = self.cls_head(cls_hidden)
        token_scores = self.summary_head(hidden)

        result = {"logits": cls_logits, "token_scores": token_scores}

        if cls_labels is not None:
            cls_loss = self.cls_criterion(cls_logits, cls_labels)
            result["cls_loss"] = cls_loss

            if token_labels is not None:
                # Mask padding positions
                mask_f  = attention_mask.float()
                s_loss  = self.summary_criterion(
                    token_scores * mask_f,
                    token_labels.float() * mask_f,
                )
                result["summary_loss"] = s_loss
                result["total_loss"]   = self.LAMBDA * cls_loss + (1 - self.LAMBDA) * s_loss
            else:
                result["total_loss"] = cls_loss

        return result

    def extract_summary(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        tokenizer:      DistilBertTokenizerFast,
        top_k:          int = 20,
    ) -> list:
        """
        Return extractive summaries for a batch.
        Selects top_k highest-scored tokens and decodes them.
        """
        self.eval()
        with torch.no_grad():
            out    = self.forward(input_ids, attention_mask)
            scores = out["token_scores"]   # (batch, seq_len)

        summaries = []
        for i in range(input_ids.size(0)):
            valid     = attention_mask[i].bool()
            ids_i     = input_ids[i][valid]
            scores_i  = scores[i][valid]
            top_idx   = scores_i.topk(min(top_k, len(scores_i))).indices.sort().values
            top_ids   = ids_i[top_idx]
            text      = tokenizer.decode(top_ids, skip_special_tokens=True)
            summaries.append(text)

        return summaries


# ── Inference loader ──────────────────────────────────────────
def load_inference_model(
    checkpoint_dir: str,
    label_map_path: str,
) -> Tuple["MedPromptModel", DistilBertTokenizerFast, Dict]:
    """
    Load a saved MedPrompt model for inference.
    Used by the Gradio app (app/app.py).
    """
    with open(label_map_path) as f:
        label_map = json.load(f)

    num_classes = len(label_map["specialty_to_id"])
    model       = MedPromptModel(
        model_id="distilbert-base-uncased",
        num_classes=num_classes,
    )
    state = torch.load(f"{checkpoint_dir}/model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    print(f"Inference model loaded from '{checkpoint_dir}' ✅")
    return model, tokenizer, label_map
