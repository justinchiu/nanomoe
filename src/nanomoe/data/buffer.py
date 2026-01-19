"""Data buffer for streaming datasets and rollout samples.

Supports:
- HuggingFace datasets streaming for pretraining/SFT
- Rollout sample buffer for GRPO
- Hybrid mode combining both
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from datasets import IterableDataset

from nanomoe.data.types import Sample


@dataclass
class BufferState:
    """Serializable state for checkpointing."""

    epoch: int = 0
    samples_seen: int = 0
    buffer_samples: list[dict] = field(default_factory=list)


class DataBuffer:
    """Unified buffer for pretraining, SFT, and GRPO data loading.

    Three modes:
    - "streaming": Pull from HuggingFace streaming dataset (pretraining/SFT)
    - "rollout": Push rollout samples from generation (GRPO)
    - "hybrid": Combine streaming prompts with rollout samples
    """

    def __init__(
        self,
        tokenizer: Any | None = None,
        streaming_dataset: IterableDataset | None = None,
        tokenize_fn: Callable[[dict], Sample] | None = None,
        max_seq_len: int = 2048,
        buffer_size: int = 10_000,
        shuffle_buffer: int = 1_000,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        # Streaming iterator
        self._streaming_dataset = streaming_dataset
        self._streaming_iter: Iterator | None = None
        self._tokenize_fn = tokenize_fn or self._default_tokenize

        # Rollout buffer
        self._rollout_buffer: list[Sample] = []

        # State tracking
        self._epoch = 0
        self._samples_seen = 0

    def _default_tokenize(self, example: dict) -> Sample | None:
        """Default tokenization for text data."""
        if self.tokenizer is None:
            raise ValueError("tokenizer required for default tokenization")

        text = example.get("text", "")
        if not text:
            return None

        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[: self.max_seq_len]

        # For pretraining: all tokens are loss tokens
        loss_mask = [1] * len(tokens)

        return Sample(
            tokens=tokens,
            loss_mask=loss_mask,
            token_weights=[0.0] + [1.0] * (len(tokens) - 1),
            prompt_len=0,
        )

    def _get_streaming_iter(self) -> Iterator:
        """Get or create streaming iterator."""
        if self._streaming_iter is None:
            if self._streaming_dataset is None:
                raise ValueError("No streaming dataset configured")
            ds = self._streaming_dataset.shuffle(seed=self.seed + self._epoch, buffer_size=self.shuffle_buffer)
            self._streaming_iter = iter(ds)
        return self._streaming_iter

    def _fill_from_stream(self, n: int) -> list[Sample]:
        """Pull n samples from the streaming dataset."""
        samples = []
        it = self._get_streaming_iter()

        while len(samples) < n:
            try:
                raw = next(it)
                sample = self._tokenize_fn(raw)
                if sample is not None:
                    samples.append(sample)
                    self._samples_seen += 1
            except StopIteration:
                # Epoch finished, reset iterator
                self._epoch += 1
                self._streaming_iter = None
                it = self._get_streaming_iter()

        return samples

    # -------------------------------------------------------------------------
    # Rollout buffer operations (for GRPO)
    # -------------------------------------------------------------------------

    def add_rollout_samples(self, samples: list[Sample]):
        """Add samples from rollout generation to the buffer."""
        self._rollout_buffer.extend(samples)

        # Trim if over capacity
        if len(self._rollout_buffer) > self.buffer_size:
            self._rollout_buffer = self._rollout_buffer[-self.buffer_size :]

    def get_rollout_samples(self, n: int) -> list[Sample]:
        """Get and remove n samples from the rollout buffer."""
        samples = self._rollout_buffer[:n]
        self._rollout_buffer = self._rollout_buffer[n:]
        return samples

    @property
    def rollout_buffer_size(self) -> int:
        """Current size of the rollout buffer."""
        return len(self._rollout_buffer)

    # -------------------------------------------------------------------------
    # Unified interface
    # -------------------------------------------------------------------------

    def get_batch(
        self,
        batch_size: int,
        mode: str = "streaming",
    ) -> list[Sample]:
        """Get a batch of samples.

        Args:
            batch_size: Number of samples to return
            mode: "streaming" | "rollout" | "hybrid"

        Returns:
            List of Sample objects
        """
        if mode == "streaming":
            return self._fill_from_stream(batch_size)

        elif mode == "rollout":
            return self.get_rollout_samples(batch_size)

        elif mode == "hybrid":
            # First try rollout buffer, then fill from stream
            samples = self.get_rollout_samples(batch_size)
            remaining = batch_size - len(samples)
            if remaining > 0:
                samples.extend(self._fill_from_stream(remaining))
            return samples

        else:
            raise ValueError(f"Unknown mode: {mode}")

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {
            "epoch": self._epoch,
            "samples_seen": self._samples_seen,
            "rollout_buffer": [
                {
                    "tokens": s.tokens,
                    "loss_mask": s.loss_mask,
                    "log_probs": s.log_probs,
                    "token_weights": s.token_weights,
                    "reward": s.reward,
                    "prompt_len": s.prompt_len,
                    "group_id": s.group_id,
                }
                for s in self._rollout_buffer
            ],
        }

    def load_state_dict(self, state: dict):
        """Restore state from checkpoint."""
        self._epoch = state["epoch"]
        self._samples_seen = state["samples_seen"]
        self._streaming_iter = None  # Will be recreated with correct epoch

        self._rollout_buffer = [Sample(**s) for s in state.get("rollout_buffer", [])]


def create_sft_tokenize_fn(
    tokenizer: Any,
    max_seq_len: int,
    input_key: str = "messages",
) -> Callable[[dict], Sample | None]:
    """Create a tokenize function for SFT data with chat format.

    Args:
        tokenizer: HuggingFace tokenizer with chat template
        max_seq_len: Maximum sequence length
        input_key: Key for messages in the dataset

    Returns:
        Tokenize function that creates Sample objects
    """

    def tokenize(example: dict) -> Sample | None:
        messages = example.get(input_key, [])
        if not messages:
            return None

        # Apply chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]

        # Find where the assistant response starts for loss masking
        # For now, simple heuristic: loss on everything after first assistant turn
        loss_mask = [0] * len(tokens)

        # Encode just the prompt (all messages except last assistant)
        prompt_messages = []
        for msg in messages:
            if msg["role"] == "assistant":
                break
            prompt_messages.append(msg)

        if prompt_messages:
            prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_len = len(prompt_tokens)
        else:
            prompt_len = 0

        # Set loss mask: 1 for response tokens
        for i in range(prompt_len, len(tokens)):
            loss_mask[i] = 1

        return Sample(
            tokens=tokens,
            loss_mask=loss_mask,
            token_weights=[float(x) for x in loss_mask[1:]] + [0.0],
            prompt_len=prompt_len,
        )

    return tokenize
