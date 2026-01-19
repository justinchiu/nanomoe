from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class Sample:
    """A single sample for GRPO training."""

    tokens: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)  # 1 for response tokens, 0 for prompt
    log_probs: list[float] = field(default_factory=list)  # log prob of each response token
    token_weights: list[float] = field(default_factory=list)  # weights aligned to labels (tokens[1:])
    reward: float = 0.0
    prompt_len: int = 0
    group_id: int | None = None  # for grouping samples from same prompt


@dataclass
class PackedBatch:
    """A packed batch for efficient training with variable-length sequences."""

    tokens: Tensor  # (total_tokens,)
    position_ids: Tensor  # (total_tokens,)
    cu_seqlens: Tensor  # (num_seqs + 1,) cumulative sequence lengths
    token_weights: Tensor  # (total_tokens,) weights aligned to labels (tokens[1:])
    labels: Tensor | None = None  # (total_tokens,)
    log_probs: Tensor | None = None  # (total_response_tokens,)
    rewards: Tensor | None = None  # (num_seqs,)

    def to(self, device: torch.device) -> "PackedBatch":
        return PackedBatch(
            tokens=self.tokens.to(device),
            position_ids=self.position_ids.to(device),
            cu_seqlens=self.cu_seqlens.to(device),
            token_weights=self.token_weights.to(device),
            labels=self.labels.to(device) if self.labels is not None else None,
            log_probs=self.log_probs.to(device) if self.log_probs is not None else None,
            rewards=self.rewards.to(device) if self.rewards is not None else None,
        )


@dataclass
class SampleOutput:
    """Output from the sampling/generation step."""

    tokens: Tensor  # (batch, seq_len) - full sequences including prompt
    log_probs: Tensor  # (batch, num_generated) - log prob of each generated token
    prompt_lens: Tensor  # (batch,) - length of each prompt
