"""Learning rate schedulers for GRPO training.

WSD (Warmup-Sustain-Decay) schedule from nmoe/opt.py:
1. Warmup: linear 0 → peak over warmup_steps
2. Sustain: hold at peak until sustain_tokens
3. Decay: cosine decay to floor over decay_tokens
4. Floor: hold at floor indefinitely
"""

import math
from dataclasses import dataclass


@dataclass
class WSDConfig:
    """Configuration for WSD learning rate schedule."""

    peak_lr: float = 1e-4
    floor_lr: float = 1e-6
    warmup_steps: int = 100
    sustain_tokens: int = 0  # Tokens before decay starts (0 = decay immediately after warmup)
    decay_tokens: int = 1_000_000_000  # Tokens over which to decay


class WSDScheduler:
    """Warmup-Sustain-Decay learning rate scheduler.

    WSD schedule:
    1. Warmup: Linear from 0 to peak_lr over warmup_steps
    2. Sustain: Hold at peak_lr until sustain_tokens seen
    3. Decay: Cosine decay from peak_lr to floor_lr over decay_tokens
    4. Floor: Hold at floor_lr indefinitely

    Usage:
        scheduler = WSDScheduler(optimizer, config)

        for step in range(num_steps):
            # ... training step ...
            scheduler.step(step, tokens_seen)
    """

    def __init__(self, optimizer, config: WSDConfig):
        self.optimizer = optimizer
        self.config = config

        self._step = 0
        self._tokens_seen = 0
        self._last_lr = config.peak_lr

    def get_lr(self, step: int, tokens_seen: int) -> float:
        """Compute learning rate for given step and tokens seen."""
        cfg = self.config

        # Phase 1: Warmup (by steps)
        if step < cfg.warmup_steps:
            # Linear warmup from 0 to peak
            lr_scale = (step + 1) / max(1, cfg.warmup_steps)
            return cfg.peak_lr * lr_scale

        # Phase 2: Sustain (by tokens)
        if tokens_seen < cfg.sustain_tokens:
            return cfg.peak_lr

        # Phase 3: Decay (by tokens after sustain)
        decay_progress = tokens_seen - cfg.sustain_tokens
        if decay_progress < cfg.decay_tokens:
            # Cosine decay from peak to floor
            cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress / cfg.decay_tokens))
            return cfg.floor_lr + (cfg.peak_lr - cfg.floor_lr) * cosine

        # Phase 4: Floor
        return cfg.floor_lr

    def step(self, step: int | None = None, tokens_seen: int | None = None) -> float:
        """Update learning rate and return current LR.

        Args:
            step: Current training step (if None, uses internal counter)
            tokens_seen: Total tokens seen so far (if None, uses internal counter)

        Returns:
            Current learning rate
        """
        if step is not None:
            self._step = step
        if tokens_seen is not None:
            self._tokens_seen = tokens_seen

        lr = self.get_lr(self._step, self._tokens_seen)
        self._last_lr = lr

        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self._step += 1
        return lr

    def state_dict(self) -> dict:
        """Get scheduler state for checkpointing."""
        return {
            "step": self._step,
            "tokens_seen": self._tokens_seen,
            "last_lr": self._last_lr,
            "config": {
                "peak_lr": self.config.peak_lr,
                "floor_lr": self.config.floor_lr,
                "warmup_steps": self.config.warmup_steps,
                "sustain_tokens": self.config.sustain_tokens,
                "decay_tokens": self.config.decay_tokens,
            },
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore scheduler state from checkpoint."""
        self._step = state["step"]
        self._tokens_seen = state["tokens_seen"]
        self._last_lr = state["last_lr"]
        # Note: config is not restored - assumed to be set at init

    @property
    def last_lr(self) -> float:
        """Get the last computed learning rate."""
        return self._last_lr


class ConstantScheduler:
    """Constant learning rate (for testing/comparison)."""

    def __init__(self, optimizer, lr: float):
        self.optimizer = optimizer
        self._lr = lr
        self._step = 0

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def step(self, step: int | None = None, tokens_seen: int | None = None) -> float:
        self._step = step if step is not None else self._step + 1
        return self._lr

    def state_dict(self) -> dict:
        return {"step": self._step, "lr": self._lr}

    def load_state_dict(self, state: dict) -> None:
        self._step = state["step"]

    @property
    def last_lr(self) -> float:
        return self._lr


class CosineScheduler:
    """Cosine decay with warmup (simpler alternative to WSD)."""

    def __init__(
        self,
        optimizer,
        peak_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
    ):
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        self._step = 0
        self._last_lr = peak_lr

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak_lr * (step + 1) / max(1, self.warmup_steps)

        decay_steps = self.total_steps - self.warmup_steps
        decay_progress = step - self.warmup_steps

        if decay_progress >= decay_steps:
            return self.min_lr

        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress / decay_steps))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine

    def step(self, step: int | None = None, tokens_seen: int | None = None) -> float:
        if step is not None:
            self._step = step

        lr = self.get_lr(self._step)
        self._last_lr = lr

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self._step += 1
        return lr

    def state_dict(self) -> dict:
        return {"step": self._step}

    def load_state_dict(self, state: dict) -> None:
        self._step = state["step"]

    @property
    def last_lr(self) -> float:
        return self._last_lr
