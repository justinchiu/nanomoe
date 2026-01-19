"""Tests for ModelOutput validation."""

import pytest
import torch
from pydantic import ValidationError

from nanomoe.model import ModelOutput


def test_model_output_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ModelOutput.model_validate({"logits": torch.zeros(1, 1, 2), "aux_loss": 0.0, "extra_field": 1})
