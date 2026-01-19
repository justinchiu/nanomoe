"""Mixture of Experts layer implementation.

Features:
- Top-k routing with auxiliary load balancing loss
- SwiGLU activation (gate * up * silu)
- Optional shared expert
- Efficient batched expert computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nanomoe.model.config import MoEConfig


class SwiGLU(nn.Module):
    """SwiGLU activation: gate * silu(x) * up(x)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Expert(nn.Module):
    """Single expert FFN."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.ffn = SwiGLU(hidden_size, intermediate_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x)


class TopKRouter(nn.Module):
    """Top-k expert router with auxiliary load balancing loss."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.aux_loss_coef = config.router_aux_loss_coef
        self.jitter_noise = config.router_jitter_noise

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Route tokens to experts.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size] or [num_tokens, hidden_size]

        Returns:
            router_logits: [num_tokens, num_experts]
            expert_indices: [num_tokens, num_experts_per_tok]
            expert_weights: [num_tokens, num_experts_per_tok]
        """
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, orig_shape[-1])

        # Add jitter noise during training
        if self.training and self.jitter_noise > 0:
            hidden_states = hidden_states * (1.0 + torch.randn_like(hidden_states) * self.jitter_noise)

        # Compute router logits
        router_logits = self.gate(hidden_states)  # [num_tokens, num_experts]

        # Top-k selection
        router_weights = F.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(router_weights, self.num_experts_per_tok, dim=-1)

        # Renormalize weights
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        return router_logits, expert_indices, expert_weights

    def compute_aux_loss(self, router_logits: Tensor) -> Tensor:
        """Compute auxiliary load balancing loss.

        Encourages uniform expert utilization.
        """
        if self.aux_loss_coef == 0:
            return torch.tensor(0.0, device=router_logits.device)

        num_tokens = router_logits.shape[0]

        # Fraction of tokens routed to each expert
        router_probs = F.softmax(router_logits, dim=-1)
        expert_mask = torch.zeros_like(router_probs)
        _, indices = torch.topk(router_probs, self.num_experts_per_tok, dim=-1)
        expert_mask.scatter_(-1, indices, 1.0)

        # f_i: fraction of tokens assigned to expert i
        tokens_per_expert = expert_mask.sum(dim=0)
        f = tokens_per_expert / num_tokens

        # P_i: average router probability for expert i
        P = router_probs.mean(dim=0)

        # Auxiliary loss: sum_i(f_i * P_i) * num_experts
        aux_loss = (f * P).sum() * self.num_experts

        return aux_loss * self.aux_loss_coef


class MoELayer(nn.Module):
    """Mixture of Experts layer with top-k routing."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Router
        self.router = TopKRouter(config)

        # Experts
        self.experts = nn.ModuleList(
            [Expert(config.hidden_size, config.intermediate_size) for _ in range(config.num_experts)]
        )

        # Shared expert (always active, if enabled)
        self.shared_expert = None
        if config.shared_expert:
            shared_size = config.shared_expert_intermediate_size or config.intermediate_size
            self.shared_expert = Expert(config.hidden_size, shared_size)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass through MoE layer.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            output: [batch_size, seq_len, hidden_size]
            aux_loss: Scalar auxiliary loss for load balancing
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)

        # Route tokens to experts
        router_logits, expert_indices, expert_weights = self.router(hidden_states_flat)

        # Compute auxiliary loss
        aux_loss = self.router.compute_aux_loss(router_logits)

        # Compute expert outputs
        # For simplicity, we use a loop over experts
        # In production, use grouped GEMM or token-expert batching
        final_output = torch.zeros_like(hidden_states_flat)

        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue

            token_indices = expert_mask.nonzero(as_tuple=True)[0]
            expert_input = hidden_states_flat[token_indices]

            # Compute expert output
            expert_output = self.experts[expert_idx](expert_input)

            # Get weights for this expert
            # expert_indices: [num_tokens, num_experts_per_tok]
            # We need to find which slot (0 to num_experts_per_tok-1) has this expert
            slot_mask = expert_indices[token_indices] == expert_idx
            weights = (expert_weights[token_indices] * slot_mask.float()).sum(dim=-1, keepdim=True)

            # Accumulate weighted output
            final_output[token_indices] += weights * expert_output

        # Add shared expert output if present
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states_flat)
            # Shared expert gets equal weight to one routed expert
            final_output = final_output + shared_output / (self.num_experts_per_tok + 1)
            # Rescale routed experts
            final_output = final_output * (self.num_experts_per_tok + 1) / self.num_experts_per_tok

        return final_output.view(batch_size, seq_len, hidden_size), aux_loss


class DenseFFN(nn.Module):
    """Dense (non-MoE) FFN layer using SwiGLU."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        # Use intermediate_size * num_experts_per_tok to match active params
        intermediate = config.intermediate_size * config.num_experts_per_tok
        self.ffn = SwiGLU(config.hidden_size, intermediate)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Returns output and zero aux_loss (for API compatibility with MoE).
        """
        return self.ffn(hidden_states), torch.tensor(0.0, device=hidden_states.device)
