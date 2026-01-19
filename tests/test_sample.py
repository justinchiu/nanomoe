"""Tests for nanomoe.sample module."""

import torch
import torch.nn as nn

from nanomoe.model import ModelOutput
from nanomoe.sample import SamplingMetrics, SamplingState, generate, top_k_sample, top_p_sample


class TestTopPSample:
    """Tests for top_p_sample function."""

    def test_top_p_sample_basic(self):
        # Create logits with clear preference
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])  # Strong preference for token 0
        tokens, log_probs = top_p_sample(logits, p=0.95, temperature=1.0)

        assert tokens.shape == (1,)
        assert log_probs.shape == (1,)
        assert tokens[0].item() == 0  # Should sample the high-prob token

    def test_top_p_sample_temperature(self):
        logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])  # Uniform

        # With temperature=0.01, distribution should be sharper
        # Just verify it runs without error
        tokens, log_probs = top_p_sample(logits, p=0.95, temperature=0.01)
        assert tokens.shape == (1,)

    def test_top_p_sample_batch(self):
        logits = torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        )
        tokens, log_probs = top_p_sample(logits, p=0.95)

        assert tokens.shape == (3,)
        assert log_probs.shape == (3,)
        # Each should sample its high-prob token
        assert tokens.tolist() == [0, 1, 2]


class TestTopKSample:
    """Tests for top_k_sample function."""

    def test_top_k_sample_basic(self):
        logits = torch.tensor([[10.0, 5.0, 0.0, -5.0]])
        tokens, log_probs = top_k_sample(logits, k=2, temperature=1.0)

        assert tokens.shape == (1,)
        assert log_probs.shape == (1,)
        # Should only sample from top 2 tokens (0 or 1)
        assert tokens[0].item() in [0, 1]

    def test_top_k_sample_k1(self):
        # k=1 should be greedy
        logits = torch.tensor([[1.0, 5.0, 3.0]])
        tokens, log_probs = top_k_sample(logits, k=1)

        assert tokens[0].item() == 1  # Argmax

    def test_top_k_sample_batch(self):
        logits = torch.randn(4, 100)
        tokens, log_probs = top_k_sample(logits, k=10)

        assert tokens.shape == (4,)
        assert log_probs.shape == (4,)


class TestSamplingMetrics:
    """Tests for SamplingMetrics dataclass."""

    def test_metrics_basic(self):
        metrics = SamplingMetrics(
            num_samples=10,
            num_tokens_generated=500,
            num_prompt_tokens=100,
            elapsed_time=2.0,
            avg_generation_len=50.0,
            avg_prompt_len=10.0,
        )

        assert metrics.tokens_per_second == 250.0
        assert metrics.samples_per_second == 5.0

    def test_metrics_zero_time(self):
        metrics = SamplingMetrics(elapsed_time=0.0)
        assert metrics.tokens_per_second == 0.0
        assert metrics.samples_per_second == 0.0

    def test_metrics_to_dict(self):
        metrics = SamplingMetrics(
            num_samples=10,
            num_tokens_generated=100,
            elapsed_time=1.0,
        )
        d = metrics.to_dict(prefix="test/")

        assert "test/num_samples" in d
        assert "test/tokens_generated" in d
        assert "test/tokens_per_sec" in d
        assert d["test/num_samples"] == 10
        assert d["test/tokens_per_sec"] == 100.0

    def test_metrics_to_dict_default_prefix(self):
        metrics = SamplingMetrics()
        d = metrics.to_dict()
        assert "rollout/num_samples" in d


class TestSamplingState:
    """Tests for SamplingState class."""

    def test_state_add(self):
        state = SamplingState()

        m1 = SamplingMetrics(num_samples=10, num_tokens_generated=100, elapsed_time=1.0)
        m2 = SamplingMetrics(num_samples=20, num_tokens_generated=200, elapsed_time=2.0)

        state.add(m1)
        state.add(m2)

        assert state.total_samples == 30
        assert state.total_tokens == 300
        assert state.total_time == 3.0
        assert len(state.metrics_history) == 2

    def test_state_summary(self):
        state = SamplingState()
        state.add(SamplingMetrics(num_samples=10, num_tokens_generated=100, elapsed_time=1.0))
        state.add(SamplingMetrics(num_samples=10, num_tokens_generated=100, elapsed_time=1.0))

        summary = state.summary()
        assert summary["rollout/total_samples"] == 20
        assert summary["rollout/total_tokens"] == 200
        assert summary["rollout/avg_tokens_per_sec"] == 100.0

    def test_state_reset(self):
        state = SamplingState()
        state.add(SamplingMetrics(num_samples=10, num_tokens_generated=100, elapsed_time=1.0))

        state.reset()

        assert state.total_samples == 0
        assert state.total_tokens == 0
        assert state.total_time == 0.0
        assert len(state.metrics_history) == 0


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int = 5):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=True):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size)
        logits[..., 1] = 10.0
        return ModelOutput(logits=logits, aux_loss=0.0, past_key_values=None)


class TestGenerate:
    def test_generate_uses_model_output(self):
        model = DummyModel()
        input_ids = torch.tensor([[2, 3]])

        output = generate(
            model=model,
            input_ids=input_ids,
            attention_mask=None,
            max_new_tokens=2,
            eos_token_id=0,
            top_k=1,
        )

        assert output.tokens.shape == (1, 4)
        assert output.log_probs.shape == (1, 2)
