"""Tests for nanomoe.train module."""

import tempfile
from pathlib import Path

import torch

from nanomoe.train import (
    ConsoleLogger,
    ConstantScheduler,
    CosineScheduler,
    JsonLogger,
    MultiLogger,
    NoOpLogger,
    WSDConfig,
    WSDScheduler,
    setup_logging,
)


class TestWSDScheduler:
    """Tests for WSD learning rate scheduler."""

    def test_warmup_phase(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
        config = WSDConfig(
            peak_lr=1e-4,
            floor_lr=1e-6,
            warmup_steps=100,
            sustain_tokens=0,
            decay_tokens=1000,
        )
        scheduler = WSDScheduler(optimizer, config)

        # At step 0, should be 1/100 of peak
        lr = scheduler.get_lr(step=0, tokens_seen=0)
        assert abs(lr - 1e-4 * (1 / 100)) < 1e-10

        # At step 49, should be 50/100 of peak
        lr = scheduler.get_lr(step=49, tokens_seen=0)
        assert abs(lr - 1e-4 * (50 / 100)) < 1e-10

        # At step 99, should be 100/100 of peak
        lr = scheduler.get_lr(step=99, tokens_seen=0)
        assert abs(lr - 1e-4) < 1e-10

    def test_sustain_phase(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
        config = WSDConfig(
            peak_lr=1e-4,
            floor_lr=1e-6,
            warmup_steps=10,
            sustain_tokens=1000,
            decay_tokens=1000,
        )
        scheduler = WSDScheduler(optimizer, config)

        # During sustain (after warmup, before sustain_tokens)
        lr = scheduler.get_lr(step=50, tokens_seen=500)
        assert abs(lr - 1e-4) < 1e-10

    def test_decay_phase(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
        config = WSDConfig(
            peak_lr=1e-4,
            floor_lr=1e-6,
            warmup_steps=10,
            sustain_tokens=0,
            decay_tokens=1000,
        )
        scheduler = WSDScheduler(optimizer, config)

        # At halfway through decay
        lr = scheduler.get_lr(step=100, tokens_seen=500)
        # Cosine at 0.5 progress should give 0.5 * (peak - floor) + floor
        expected = 1e-6 + (1e-4 - 1e-6) * 0.5
        assert abs(lr - expected) < 1e-8

    def test_floor_phase(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
        config = WSDConfig(
            peak_lr=1e-4,
            floor_lr=1e-6,
            warmup_steps=10,
            sustain_tokens=0,
            decay_tokens=1000,
        )
        scheduler = WSDScheduler(optimizer, config)

        # After decay completes
        lr = scheduler.get_lr(step=1000, tokens_seen=2000)
        assert abs(lr - 1e-6) < 1e-10

    def test_step_updates_optimizer(self):
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=1e-4)
        config = WSDConfig(peak_lr=1e-3, warmup_steps=10)
        scheduler = WSDScheduler(optimizer, config)

        scheduler.step(step=5, tokens_seen=0)

        # Check optimizer LR was updated
        assert optimizer.param_groups[0]["lr"] == scheduler.last_lr

    def test_state_dict(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
        config = WSDConfig(peak_lr=1e-4, warmup_steps=100)
        scheduler = WSDScheduler(optimizer, config)

        scheduler.step(step=50, tokens_seen=1000)
        state = scheduler.state_dict()

        assert state["step"] == 51  # step is incremented after
        assert state["tokens_seen"] == 1000

    def test_load_state_dict(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
        config = WSDConfig(peak_lr=1e-4, warmup_steps=100)
        scheduler = WSDScheduler(optimizer, config)

        state = {"step": 50, "tokens_seen": 1000, "last_lr": 5e-5}
        scheduler.load_state_dict(state)

        assert scheduler._step == 50
        assert scheduler._tokens_seen == 1000


class TestCosineScheduler:
    """Tests for Cosine learning rate scheduler."""

    def test_warmup(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
        scheduler = CosineScheduler(
            optimizer,
            peak_lr=1e-3,
            min_lr=1e-5,
            warmup_steps=10,
            total_steps=100,
        )

        lr = scheduler.get_lr(step=4)  # 5/10 of warmup
        assert abs(lr - 1e-3 * 0.5) < 1e-10

    def test_cosine_decay(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
        scheduler = CosineScheduler(
            optimizer,
            peak_lr=1e-3,
            min_lr=1e-5,
            warmup_steps=0,
            total_steps=100,
        )

        # At end of decay
        lr = scheduler.get_lr(step=100)
        assert abs(lr - 1e-5) < 1e-10


class TestConstantScheduler:
    """Tests for Constant learning rate scheduler."""

    def test_constant(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
        scheduler = ConstantScheduler(optimizer, lr=5e-4)

        assert scheduler.last_lr == 5e-4
        scheduler.step()
        assert scheduler.last_lr == 5e-4
        scheduler.step(step=100, tokens_seen=10000)
        assert scheduler.last_lr == 5e-4


class TestLoggers:
    """Tests for logging utilities."""

    def test_noop_logger(self):
        logger = NoOpLogger()
        logger.log_metrics({"loss": 1.0}, step=0)
        logger.log_config({"lr": 1e-4})
        logger.close()
        logger.sync()
        # Should not raise

    def test_console_logger(self):
        logger = ConsoleLogger(print_every=5)
        logger.log_metrics({"loss": 1.0}, step=0)
        logger.log_metrics({"loss": 0.9}, step=3)  # Should not print
        logger.log_metrics({"loss": 0.8}, step=5)  # Should print
        logger.close()

    def test_json_logger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            logger = JsonLogger(path)

            logger.log_config({"lr": 1e-4})
            logger.log_metrics({"loss": 1.0}, step=0)
            logger.log_metrics({"loss": 0.5}, step=1)
            logger.sync()
            logger.close()

            # Check file was written
            assert path.exists()
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 3  # config + 2 metrics

    def test_multi_logger(self):
        logger1 = NoOpLogger()
        logger2 = NoOpLogger()
        multi = MultiLogger([logger1, logger2])

        multi.log_metrics({"loss": 1.0}, step=0)
        multi.log_config({"lr": 1e-4})
        multi.sync()
        multi.close()

    def test_setup_logging_noop_for_non_rank0(self):
        logger = setup_logging(rank=1)
        assert isinstance(logger, NoOpLogger)

    def test_setup_logging_console_only(self):
        logger = setup_logging(console=True, rank=0)
        assert isinstance(logger, MultiLogger)

    def test_setup_logging_with_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(log_dir=tmpdir, console=False, rank=0)
            logger.log_metrics({"loss": 1.0}, step=0)
            logger.close()

            # Check JSON file was created
            json_path = Path(tmpdir) / "metrics.jsonl"
            assert json_path.exists()
