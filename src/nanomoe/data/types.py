from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class Sample:
    """A single sample for GRPO training."""

    tokens: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)  # 1 for response tokens, 0 for prompt
    log_probs: list[float] = field(default_factory=list)  # log prob of each response token
    reward: float = 0.0
    prompt_len: int = 0
    group_id: int | None = None  # for grouping samples from same prompt


@dataclass
class PackedBatch:
    """A packed batch for efficient training with variable-length sequences."""

    tokens: Tensor  # (total_tokens,)
    loss_mask: Tensor  # (total_tokens,)
    position_ids: Tensor  # (total_tokens,)
    cu_seqlens: Tensor  # (num_seqs + 1,) cumulative sequence lengths
    log_probs: Tensor  # (total_response_tokens,)
    advantages: Tensor  # (total_response_tokens,)
    rewards: Tensor  # (num_seqs,)


@dataclass
class SampleOutput:
    """Output from the sampling/generation step."""

    tokens: Tensor  # (batch, seq_len) - full sequences including prompt
    log_probs: Tensor  # (batch, num_generated) - log prob of each generated token
    prompt_lens: Tensor  # (batch,) - length of each prompt
