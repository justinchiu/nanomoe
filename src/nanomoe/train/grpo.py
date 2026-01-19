"""GRPO (Group Relative Policy Optimization) training with FSDP2.

Based on DeepSeek-R1's GRPO algorithm:
- Group-relative advantages (no value function needed)
- No KL penalty / reference model
- Simple policy gradient with reward normalization
"""

import logging
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Module
from torch.optim import Optimizer

from nanomoe.data.packing import pack_sequences
from nanomoe.data.types import PackedBatch, Sample

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    # GRPO parameters
    n_samples_per_prompt: int = 8
    reward_clip: float | None = None  # Clip rewards to [-clip, clip]
    advantage_clip: float | None = None  # Clip advantages

    # Training parameters
    max_tokens_per_batch: int = 4096
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Logging
    log_interval: int = 10


def compute_grpo_advantages(
    samples: list[Sample],
    n_samples_per_prompt: int,
    reward_clip: float | None = None,
    advantage_clip: float | None = None,
) -> list[Sample]:
    """Compute group-relative advantages for GRPO.

    For each group of n_samples_per_prompt samples from the same prompt:
    advantage = (reward - mean_reward) / (std_reward + eps)

    The advantage is then broadcast to all response tokens.

    Args:
        samples: List of samples with rewards
        n_samples_per_prompt: Number of samples per prompt group
        reward_clip: Optional reward clipping
        advantage_clip: Optional advantage clipping

    Returns:
        Samples with log_probs replaced by per-token advantages
    """
    # Group samples by prompt
    num_groups = len(samples) // n_samples_per_prompt

    for group_idx in range(num_groups):
        start = group_idx * n_samples_per_prompt
        end = start + n_samples_per_prompt
        group_samples = samples[start:end]

        # Get rewards for this group
        rewards = torch.tensor([s.reward for s in group_samples])

        # Clip rewards if specified
        if reward_clip is not None:
            rewards = rewards.clamp(-reward_clip, reward_clip)

        # Compute group-relative advantages
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        advantages = (rewards - mean_reward) / (std_reward + 1e-8)

        # Clip advantages if specified
        if advantage_clip is not None:
            advantages = advantages.clamp(-advantage_clip, advantage_clip)

        # Assign per-token advantages (same advantage for all tokens in response)
        for i, sample in enumerate(group_samples):
            adv = advantages[i].item()
            # Replace log_probs with advantages (same shape: one per response token)
            num_response_tokens = sum(sample.loss_mask)
            sample.log_probs = [adv] * num_response_tokens

    return samples


def grpo_loss(
    model: Module,
    packed_batch: PackedBatch,
    rollout_log_probs: Tensor | None = None,
) -> tuple[Tensor, dict]:
    """Compute GRPO policy gradient loss.

    Loss = -E[advantage * log_prob]

    For importance sampling (if rollout_log_probs provided):
    Loss = -E[advantage * min(ratio, 1+eps) * log_prob]
    where ratio = exp(log_prob - rollout_log_prob)

    Args:
        model: Policy model
        packed_batch: Packed batch with tokens, masks, advantages
        rollout_log_probs: Log probs from rollout (for importance sampling)

    Returns:
        loss: Scalar loss tensor
        metrics: Dictionary of training metrics
    """
    device = next(model.parameters()).device

    # Move batch to device
    tokens = packed_batch.tokens.to(device)
    loss_mask = packed_batch.loss_mask.to(device)
    position_ids = packed_batch.position_ids.to(device)
    advantages = packed_batch.advantages.to(device)
    # cu_seqlens available in packed_batch for Flash Attention (future use)

    # Forward pass
    outputs = model(
        input_ids=tokens.unsqueeze(0),  # Add batch dim
        position_ids=position_ids.unsqueeze(0),
        # TODO: Add cu_seqlens support for Flash Attention
    )

    logits = outputs.logits.squeeze(0)  # (seq_len, vocab)

    # Compute log probs for each token
    log_probs = torch.log_softmax(logits[:-1], dim=-1)  # (seq_len-1, vocab)
    token_log_probs = log_probs.gather(-1, tokens[1:].unsqueeze(-1)).squeeze(-1)  # (seq_len-1,)

    # Mask to response tokens only
    response_mask = loss_mask[1:].float()  # (seq_len-1,)
    masked_log_probs = token_log_probs * response_mask

    # Get advantages for response tokens
    # advantages tensor has one value per response token (matching loss_mask==1)
    # We need to expand it to match the full sequence
    response_indices = response_mask.nonzero(as_tuple=True)[0]
    expanded_advantages = torch.zeros_like(masked_log_probs)
    if len(response_indices) > 0 and len(advantages) > 0:
        # Truncate if needed (in case of mismatch)
        n = min(len(response_indices), len(advantages))
        expanded_advantages[response_indices[:n]] = advantages[:n]

    # Policy gradient loss: -E[advantage * log_prob]
    loss = -(expanded_advantages * masked_log_probs).sum() / (response_mask.sum() + 1e-8)

    # Compute metrics
    with torch.no_grad():
        metrics = {
            "loss": loss.item(),
            "mean_log_prob": (masked_log_probs.sum() / (response_mask.sum() + 1e-8)).item(),
            "mean_advantage": advantages.mean().item() if len(advantages) > 0 else 0.0,
            "num_tokens": response_mask.sum().item(),
            "mean_reward": packed_batch.rewards.mean().item(),
        }

    return loss, metrics


class GRPOTrainer:
    """GRPO trainer with FSDP2 support."""

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        config: GRPOConfig,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config

        self._step = 0
        self._accumulated_loss = 0.0
        self._accumulated_metrics: dict[str, float] = {}

    def train_step(self, samples: list[Sample]) -> dict:
        """Run a single GRPO training step.

        Args:
            samples: List of samples with rewards (from generation + reward computation)

        Returns:
            Dictionary of training metrics
        """
        # Compute group-relative advantages
        samples = compute_grpo_advantages(
            samples,
            n_samples_per_prompt=self.config.n_samples_per_prompt,
            reward_clip=self.config.reward_clip,
            advantage_clip=self.config.advantage_clip,
        )

        # Pack sequences into batches
        packed_batches = pack_sequences(
            samples,
            max_tokens_per_batch=self.config.max_tokens_per_batch,
        )

        total_loss = 0.0
        all_metrics: dict[str, list[float]] = {}

        self.model.train()

        for packed_batch in packed_batches:
            # Forward + loss
            loss, metrics = grpo_loss(self.model, packed_batch)

            # Scale loss for gradient accumulation
            scaled_loss = loss / len(packed_batches)
            scaled_loss.backward()

            total_loss += loss.item()

            # Accumulate metrics
            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            if isinstance(self.model, FSDP):
                self.model.clip_grad_norm_(self.config.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        self._step += 1

        # Average metrics
        avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}
        avg_metrics["total_loss"] = total_loss
        avg_metrics["num_samples"] = len(samples)
        avg_metrics["num_batches"] = len(packed_batches)

        return avg_metrics


def setup_fsdp2(
    model: Module,
    mixed_precision: bool = True,
) -> Module:
    """Wrap model with FSDP2.

    Args:
        model: Model to wrap
        mixed_precision: Use bf16 mixed precision

    Returns:
        FSDP-wrapped model
    """
    from functools import partial

    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    # Mixed precision policy
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )
    else:
        mp_policy = None

    # Auto wrap policy for transformer layers
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
    )

    return model


def init_distributed():
    """Initialize distributed training."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
