"""Metric logging utilities for GRPO training.

Inspired by tinker-cookbook's ml_log.py and slime's wandb_utils.py.

Supports:
- Weights & Biases (wandb)
- JSON file logging
- Console printing
- Multiple backends simultaneously
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _serialize_config(obj: Any) -> Any:
    """Recursively serialize config objects for logging."""
    if obj is None:
        return None
    if isinstance(obj, str | int | float | bool):
        return obj
    if isinstance(obj, list | tuple):
        return [_serialize_config(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _serialize_config(v) for k, v in obj.items()}
    if is_dataclass(obj) and not isinstance(obj, type):
        return _serialize_config(asdict(obj))
    if hasattr(obj, "to_dict"):
        return _serialize_config(obj.to_dict())
    if hasattr(obj, "__dict__"):
        return _serialize_config(obj.__dict__)
    if callable(obj):
        return f"{obj.__module__}.{obj.__name__}" if hasattr(obj, "__name__") else str(obj)
    return str(obj)


class Logger(ABC):
    """Abstract base class for metric loggers."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        """Log metrics at a given step."""

    @abstractmethod
    def log_config(self, config: Any) -> None:
        """Log configuration/hyperparameters."""

    def close(self) -> None:  # noqa: B027
        """Clean up resources."""

    def sync(self) -> None:  # noqa: B027
        """Sync/flush any buffered data."""


class WandbLogger(Logger):
    """Weights & Biases logger."""

    def __init__(
        self,
        project: str,
        name: str | None = None,
        group: str | None = None,
        config: Any = None,
        mode: str = "online",  # "online", "offline", "disabled"
        dir: str | None = None,
        **kwargs,
    ):
        try:
            import wandb

            self._wandb = wandb
        except ImportError as err:
            raise ImportError("wandb not installed. Run: pip install wandb") from err

        # Set mode via environment if not online
        if mode != "online":
            os.environ["WANDB_MODE"] = mode

        # Ensure directory exists
        if dir:
            os.makedirs(dir, exist_ok=True)

        init_kwargs = {
            "project": project,
            "name": name,
            "group": group,
            "config": _serialize_config(config) if config else None,
            "reinit": True,
            **kwargs,
        }
        if dir:
            init_kwargs["dir"] = dir

        self._run = wandb.init(**init_kwargs)

        # Define metric groups for proper x-axis tracking
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("rollout/step")
        wandb.define_metric("rollout/*", step_metric="rollout/step")
        wandb.define_metric("perf/*", step_metric="train/step")

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        # Add step to appropriate namespace
        if "train/step" not in metrics and "rollout/step" not in metrics:
            metrics["train/step"] = step
        self._wandb.log(metrics)

    def log_config(self, config: Any) -> None:
        self._wandb.config.update(_serialize_config(config))

    def close(self) -> None:
        self._wandb.finish()

    def sync(self) -> None:
        pass  # wandb syncs automatically

    @property
    def run_id(self) -> str | None:
        return self._run.id if self._run else None

    @property
    def run_url(self) -> str | None:
        return self._run.url if self._run else None


class JsonLogger(Logger):
    """JSON lines file logger."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a")
        self._config_logged = False

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        entry = {"step": step, "timestamp": time.time(), **metrics}
        self._file.write(json.dumps(entry) + "\n")

    def log_config(self, config: Any) -> None:
        if self._config_logged:
            return
        entry = {"type": "config", "config": _serialize_config(config), "timestamp": time.time()}
        self._file.write(json.dumps(entry) + "\n")
        self._config_logged = True

    def close(self) -> None:
        self._file.close()

    def sync(self) -> None:
        self._file.flush()


class ConsoleLogger(Logger):
    """Pretty-print logger for console output."""

    def __init__(self, print_every: int = 1):
        self.print_every = print_every
        self._last_print = -1

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        if step - self._last_print < self.print_every:
            return

        self._last_print = step

        # Format metrics nicely
        parts = [f"step={step}"]
        for k, v in sorted(metrics.items()):
            if k.endswith("/step"):
                continue
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")

        print(" | ".join(parts))

    def log_config(self, config: Any) -> None:
        print(f"Config: {_serialize_config(config)}")

    def close(self) -> None:
        pass

    def sync(self) -> None:
        pass


class MultiLogger(Logger):
    """Multiplexer that forwards to multiple loggers."""

    def __init__(self, loggers: list[Logger]):
        self.loggers = loggers

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        for lg in self.loggers:
            lg.log_metrics(metrics, step)

    def log_config(self, config: Any) -> None:
        for lg in self.loggers:
            lg.log_config(config)

    def close(self) -> None:
        for lg in self.loggers:
            lg.close()

    def sync(self) -> None:
        for lg in self.loggers:
            lg.sync()


class NoOpLogger(Logger):
    """Logger that does nothing (for testing/disabled logging)."""

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        pass

    def log_config(self, config: Any) -> None:
        pass


def setup_logging(
    log_dir: str | Path | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    wandb_group: str | None = None,
    wandb_mode: str = "online",
    config: Any = None,
    console: bool = True,
    console_every: int = 10,
    rank: int = 0,
) -> Logger:
    """Set up logging with multiple backends.

    Args:
        log_dir: Directory for JSON logs (if None, no JSON logging)
        wandb_project: W&B project name (if None, no W&B logging)
        wandb_name: W&B run name
        wandb_group: W&B group name
        wandb_mode: W&B mode ("online", "offline", "disabled")
        config: Config to log
        console: Whether to log to console
        console_every: Print to console every N steps
        rank: Current rank (only rank 0 logs by default)

    Returns:
        Logger instance
    """
    # Only rank 0 logs
    if rank != 0:
        return NoOpLogger()

    loggers: list[Logger] = []

    # Console logger
    if console:
        loggers.append(ConsoleLogger(print_every=console_every))

    # JSON logger
    if log_dir:
        json_path = Path(log_dir) / "metrics.jsonl"
        loggers.append(JsonLogger(json_path))

    # W&B logger
    if wandb_project:
        loggers.append(
            WandbLogger(
                project=wandb_project,
                name=wandb_name,
                group=wandb_group,
                config=config,
                mode=wandb_mode,
                dir=str(log_dir) if log_dir else None,
            )
        )

    if not loggers:
        return NoOpLogger()

    multi = MultiLogger(loggers)

    # Log config
    if config:
        multi.log_config(config)

    return multi
