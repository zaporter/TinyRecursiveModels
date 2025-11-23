from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        if labels.ndim == 2:
            labels = labels.unsqueeze(1)

        batch_size, num_targets, seq_len = labels.shape
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1)
        outputs["preds"] = preds

        # Expand logits to match targets for computing per-target losses
        logits_expanded = logits.unsqueeze(1).expand(-1, num_targets, -1, -1)
        logits_flat = logits_expanded.reshape(batch_size * num_targets, seq_len, logits.shape[-1])
        labels_flat = labels.reshape(batch_size * num_targets, seq_len)
        mask = (labels != IGNORE_LABEL_ID)
        mask_flat = mask.reshape(batch_size * num_targets, seq_len)

        per_token_loss = self.loss_fn(
            logits_flat,
            labels_flat,
            ignore_index=IGNORE_LABEL_ID,
            valid_mask=mask_flat,
        ).reshape(batch_size, num_targets, seq_len)

        loss_counts = mask.sum(-1)
        loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1).to(per_token_loss.dtype)
        per_target_mean_loss = (per_token_loss / loss_divisor).sum(-1)
        best_loss_values, best_target_idx = per_target_mean_loss.min(dim=1)
        lm_loss = best_loss_values.sum()

        target_offsets = torch.arange(batch_size, device=labels.device, dtype=best_target_idx.dtype) * max(num_targets, 1)
        best_flat_indices = target_offsets + best_target_idx
        best_labels = labels_flat.index_select(0, best_flat_indices)
        best_masks = mask_flat.index_select(0, best_flat_indices)
        best_loss_counts = loss_counts.reshape(batch_size * num_targets).index_select(0, best_flat_indices)
        best_loss_divisor = best_loss_counts.clamp_min(1).unsqueeze(-1).to(torch.float32)

        with torch.no_grad():
            is_correct = best_masks & (preds == best_labels)
            seq_is_correct = is_correct.sum(-1) == best_loss_counts

            valid_metrics = new_carry.halted & (best_loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics,
                    (is_correct.to(torch.float32) / best_loss_divisor).sum(-1),
                    0,
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()

