"""Loss utilities shared across pretrain/SFT/RL."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def unified_loss(
    logits: Tensor,
    labels: Tensor,
    token_weights: Tensor,
) -> Tensor:
    """Compute a weighted next-token loss over packed sequences.

    Args:
        logits: (..., seq_len, vocab)
        labels: (..., seq_len) - next-token labels aligned to logits
        token_weights: (..., seq_len) - per-token weights (mask/advantage)
    """
    if logits.ndim < 2:
        raise ValueError("logits must have at least 2 dims (seq_len, vocab)")
    if labels.shape != token_weights.shape:
        raise ValueError("labels and token_weights must have the same shape")

    vocab = logits.shape[-1]
    # Align to next-token prediction by dropping the last position.
    if logits.ndim == 2:
        logits_t = logits[:-1].contiguous().view(-1, vocab)
        labels_t = labels[:-1].contiguous().view(-1)
        weights_t = token_weights[:-1].contiguous().view(-1)
    else:
        logits_t = logits[..., :-1, :].contiguous().view(-1, vocab)
        labels_t = labels[..., :-1].contiguous().view(-1)
        weights_t = token_weights[..., :-1].contiguous().view(-1)

    log_probs = -F.cross_entropy(logits_t, labels_t, reduction="none")
    denom = weights_t.abs().sum().clamp(min=1)
    return -(weights_t * log_probs).sum() / denom
