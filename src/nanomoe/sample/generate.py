"""Generation utilities for GRPO sampling.

Simple generation loop with log probability tracking.
Designed for colocated training + sampling (no separate inference engine).
"""

import time
from dataclasses import dataclass, field
from functools import partial

import torch
from torch import Tensor
from torch.nn import Module

from nanomoe.data.types import Sample, SampleOutput


@dataclass
class SamplingMetrics:
    """Metrics from a sampling run."""

    num_samples: int = 0
    num_tokens_generated: int = 0
    num_prompt_tokens: int = 0
    elapsed_time: float = 0.0
    avg_generation_len: float = 0.0
    avg_prompt_len: float = 0.0

    # Derived metrics
    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_time <= 0:
            return 0.0
        return self.num_tokens_generated / self.elapsed_time

    @property
    def samples_per_second(self) -> float:
        if self.elapsed_time <= 0:
            return 0.0
        return self.num_samples / self.elapsed_time

    def to_dict(self, prefix: str = "rollout/") -> dict:
        """Convert to dict for logging."""
        return {
            f"{prefix}num_samples": self.num_samples,
            f"{prefix}tokens_generated": self.num_tokens_generated,
            f"{prefix}prompt_tokens": self.num_prompt_tokens,
            f"{prefix}elapsed_time": self.elapsed_time,
            f"{prefix}avg_gen_len": self.avg_generation_len,
            f"{prefix}avg_prompt_len": self.avg_prompt_len,
            f"{prefix}tokens_per_sec": self.tokens_per_second,
            f"{prefix}samples_per_sec": self.samples_per_second,
        }


@dataclass
class SamplingState:
    """Accumulates sampling metrics across multiple batches."""

    metrics_history: list[SamplingMetrics] = field(default_factory=list)
    total_samples: int = 0
    total_tokens: int = 0
    total_time: float = 0.0

    def add(self, metrics: SamplingMetrics) -> None:
        self.metrics_history.append(metrics)
        self.total_samples += metrics.num_samples
        self.total_tokens += metrics.num_tokens_generated
        self.total_time += metrics.elapsed_time

    def summary(self, prefix: str = "rollout/") -> dict:
        """Get aggregate metrics."""
        return {
            f"{prefix}total_samples": self.total_samples,
            f"{prefix}total_tokens": self.total_tokens,
            f"{prefix}total_time": self.total_time,
            f"{prefix}avg_tokens_per_sec": self.total_tokens / max(self.total_time, 1e-6),
            f"{prefix}avg_samples_per_sec": self.total_samples / max(self.total_time, 1e-6),
        }

    def reset(self) -> None:
        self.metrics_history.clear()
        self.total_samples = 0
        self.total_tokens = 0
        self.total_time = 0.0


def top_p_sample(
    logits: Tensor,
    p: float = 0.95,
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Sample from logits with nucleus (top-p) sampling.

    Args:
        logits: (batch, vocab_size) unnormalized log probabilities
        p: Cumulative probability threshold
        temperature: Sampling temperature

    Returns:
        sampled_tokens: (batch,) sampled token ids
        log_probs: (batch,) log probability of sampled tokens
    """
    if temperature != 1.0:
        logits = logits / temperature

    # Sort by probability
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask tokens beyond top-p threshold
    mask = cumsum_probs - sorted_probs > p
    sorted_logits[mask] = float("-inf")

    # Sample from filtered distribution
    probs = torch.softmax(sorted_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1)
    sampled_token = sorted_idx.gather(-1, sampled_idx).squeeze(-1)

    # Get log probability of sampled token
    log_probs = torch.log_softmax(logits, dim=-1)
    sampled_log_prob = log_probs.gather(-1, sampled_token.unsqueeze(-1)).squeeze(-1)

    return sampled_token, sampled_log_prob


def top_k_sample(
    logits: Tensor,
    k: int = 50,
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Sample from logits with top-k sampling.

    Args:
        logits: (batch, vocab_size) unnormalized log probabilities
        k: Number of top tokens to consider
        temperature: Sampling temperature

    Returns:
        sampled_tokens: (batch,) sampled token ids
        log_probs: (batch,) log probability of sampled tokens
    """
    if temperature != 1.0:
        logits = logits / temperature

    # Get top-k
    top_logits, top_idx = torch.topk(logits, k, dim=-1)

    # Sample from top-k
    probs = torch.softmax(top_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1)
    sampled_token = top_idx.gather(-1, sampled_idx).squeeze(-1)

    # Get log probability
    log_probs = torch.log_softmax(logits, dim=-1)
    sampled_log_prob = log_probs.gather(-1, sampled_token.unsqueeze(-1)).squeeze(-1)

    return sampled_token, sampled_log_prob


def _unwrap_outputs(outputs) -> tuple[Tensor, list[tuple[Tensor, Tensor]] | None]:
    return outputs.logits, outputs.past_key_values


@torch.no_grad()
def generate(
    model: Module,
    input_ids: Tensor,
    attention_mask: Tensor | None,
    max_new_tokens: int,
    eos_token_id: int | list[int],
    pad_token_id: int = 0,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int | None = None,
) -> SampleOutput:
    """Generate tokens from a model with logits/past_key_values attributes.

    Args:
        model: Causal LM with `.logits` and `.past_key_values`
        input_ids: (batch, prompt_len) input token ids
        attention_mask: (batch, prompt_len) attention mask, or None
        max_new_tokens: Maximum number of tokens to generate
        eos_token_id: End of sequence token id(s)
        pad_token_id: Padding token id
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling (if set, overrides top_p)

    Returns:
        SampleOutput with tokens, log_probs, and prompt_lens
    """
    batch_size, prompt_len = input_ids.shape
    device = input_ids.device

    # Handle multiple EOS tokens
    if isinstance(eos_token_id, int):
        eos_token_ids = {eos_token_id}
    else:
        eos_token_ids = set(eos_token_id)

    # Track finished sequences
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Storage
    generated_tokens: list[Tensor] = []
    generated_log_probs: list[Tensor] = []

    # KV cache
    past_key_values = None
    current_input = input_ids
    current_mask = attention_mask

    # Choose sampling function
    if top_k is not None:
        sample_fn = partial(top_k_sample, k=top_k, temperature=temperature)
    else:
        sample_fn = partial(top_p_sample, p=top_p, temperature=temperature)

    for _ in range(max_new_tokens):
        # Forward pass
        outputs = model(
            input_ids=current_input,
            attention_mask=current_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits, past_key_values = _unwrap_outputs(outputs)
        logits = logits[:, -1, :]  # (batch, vocab)

        # Sample
        next_token, log_prob = sample_fn(logits)

        # Mask finished sequences
        next_token = torch.where(done, pad_token_id, next_token)
        log_prob = torch.where(done, torch.zeros_like(log_prob), log_prob)

        generated_tokens.append(next_token)
        generated_log_probs.append(log_prob)

        # Check for EOS
        is_eos = torch.zeros_like(done)
        for eos_id in eos_token_ids:
            is_eos = is_eos | (next_token == eos_id)
        done = done | is_eos

        if done.all():
            break

        # Prepare next iteration
        current_input = next_token.unsqueeze(1)
        if current_mask is not None:
            current_mask = torch.cat(
                [current_mask, (~done).unsqueeze(1).to(current_mask.dtype)],
                dim=1,
            )

    # Stack results
    if generated_tokens:
        generated_tokens_t = torch.stack(generated_tokens, dim=1)
        generated_log_probs_t = torch.stack(generated_log_probs, dim=1)
    else:
        generated_tokens_t = torch.empty(batch_size, 0, dtype=torch.long, device=device)
        generated_log_probs_t = torch.empty(batch_size, 0, dtype=torch.float, device=device)

    # Concatenate with prompt
    full_tokens = torch.cat([input_ids, generated_tokens_t], dim=1)

    return SampleOutput(
        tokens=full_tokens,
        log_probs=generated_log_probs_t,
        prompt_lens=torch.full((batch_size,), prompt_len, dtype=torch.long, device=device),
    )


@torch.no_grad()
def generate_with_metrics(
    model: Module,
    input_ids: Tensor,
    attention_mask: Tensor | None,
    max_new_tokens: int,
    eos_token_id: int | list[int],
    **kwargs,
) -> tuple[SampleOutput, SamplingMetrics]:
    """Generate tokens with timing metrics.

    Args:
        Same as generate()

    Returns:
        Tuple of (SampleOutput, SamplingMetrics)
    """
    batch_size, prompt_len = input_ids.shape

    # Sync CUDA for accurate timing
    if input_ids.is_cuda:
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    output = generate(model, input_ids, attention_mask, max_new_tokens, eos_token_id, **kwargs)

    if input_ids.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    # Compute generation lengths (excluding prompt and padding)
    gen_lens = []
    for i in range(batch_size):
        seq = output.tokens[i, prompt_len:].tolist()
        # Count non-pad tokens
        gen_len = sum(1 for t in seq if t != kwargs.get("pad_token_id", 0))
        gen_lens.append(gen_len)

    total_gen_tokens = sum(gen_lens)

    metrics = SamplingMetrics(
        num_samples=batch_size,
        num_tokens_generated=total_gen_tokens,
        num_prompt_tokens=batch_size * prompt_len,
        elapsed_time=elapsed,
        avg_generation_len=total_gen_tokens / max(batch_size, 1),
        avg_prompt_len=float(prompt_len),
    )

    return output, metrics


@torch.no_grad()
def generate_grpo_samples(
    model: Module,
    prompts: Tensor,
    attention_mask: Tensor | None,
    n_samples_per_prompt: int,
    **generate_kwargs,
) -> SampleOutput:
    """Generate multiple samples per prompt for GRPO.

    Args:
        model: Causal LM with `.logits` and `.past_key_values`
        prompts: (num_prompts, prompt_len) prompt token ids
        attention_mask: (num_prompts, prompt_len) attention mask
        n_samples_per_prompt: Number of samples to generate per prompt
        **generate_kwargs: Arguments passed to generate()

    Returns:
        SampleOutput with shape (num_prompts * n_samples_per_prompt, ...)
    """
    # Repeat prompts n times
    batch_prompts = prompts.repeat_interleave(n_samples_per_prompt, dim=0)
    if attention_mask is not None:
        batch_mask = attention_mask.repeat_interleave(n_samples_per_prompt, dim=0)
    else:
        batch_mask = None

    return generate(model, batch_prompts, batch_mask, **generate_kwargs)


@torch.no_grad()
def generate_grpo_samples_with_metrics(
    model: Module,
    prompts: Tensor,
    attention_mask: Tensor | None,
    n_samples_per_prompt: int,
    **generate_kwargs,
) -> tuple[SampleOutput, SamplingMetrics]:
    """Generate multiple samples per prompt for GRPO with timing metrics.

    Args:
        model: Causal LM with `.logits` and `.past_key_values`
        prompts: (num_prompts, prompt_len) prompt token ids
        attention_mask: (num_prompts, prompt_len) attention mask
        n_samples_per_prompt: Number of samples to generate per prompt
        **generate_kwargs: Arguments passed to generate()

    Returns:
        Tuple of (SampleOutput, SamplingMetrics)
    """
    # Repeat prompts n times
    batch_prompts = prompts.repeat_interleave(n_samples_per_prompt, dim=0)
    if attention_mask is not None:
        batch_mask = attention_mask.repeat_interleave(n_samples_per_prompt, dim=0)
    else:
        batch_mask = None

    return generate_with_metrics(model, batch_prompts, batch_mask, **generate_kwargs)


def samples_to_grpo_batch(
    output: SampleOutput,
    rewards: Tensor,
    n_samples_per_prompt: int,
    tokenizer=None,
) -> list[Sample]:
    """Convert generation output to Sample objects for GRPO training.

    Args:
        output: SampleOutput from generate_grpo_samples
        rewards: (batch,) reward for each sequence
        n_samples_per_prompt: Number of samples per prompt (for group_id)
        tokenizer: Optional tokenizer for decoding (debugging)

    Returns:
        List of Sample objects ready for packing
    """
    batch_size = output.tokens.shape[0]
    samples = []

    for i in range(batch_size):
        prompt_len = int(output.prompt_lens[i].item())
        tokens = output.tokens[i].tolist()
        log_probs = output.log_probs[i].tolist()

        # Create loss mask: 0 for prompt, 1 for generated
        loss_mask = [0] * prompt_len + [1] * (len(tokens) - prompt_len)

        # Trim trailing pad tokens
        while tokens and tokens[-1] == 0:
            tokens.pop()
            loss_mask.pop()
            if log_probs:
                log_probs.pop()

        samples.append(
            Sample(
                tokens=tokens,
                loss_mask=loss_mask,
                log_probs=log_probs,
                reward=rewards[i].item(),
                prompt_len=prompt_len,
                group_id=i // n_samples_per_prompt,
            )
        )

    return samples
