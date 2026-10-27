"""Sparse top-k mixture of copies of an existing feed-forward module."""

from copy import deepcopy
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class SparseMoE(nn.Module):
    """Replace an FFN while retaining its architecture and pretrained weights.

    Only valid tokens are routed, and each token is evaluated by ``top_k``
    experts. ``aux_loss`` is the unweighted load-balancing loss for the most
    recent forward; balanced assignment has loss 1. ``last_expert_counts``
    records the number of selected valid tokens per expert, without gradients.

    A token mask has the input's leading dimensions and remains active until
    replaced or cleared with ``set_token_mask(None)``. Masked outputs are zero.
    """

    def __init__(
        self,
        expert: nn.Module,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        if not isinstance(expert, nn.Module):
            raise TypeError("expert must be an nn.Module")
        if type(d_model) is not int or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if type(top_k) is not int or top_k not in (2, 3, 4):
            raise ValueError("top_k must be one of 2, 3, or 4")
        if type(num_experts) is not int or num_experts < top_k:
            raise ValueError("num_experts must be an integer at least top_k")

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(deepcopy(expert) for _ in range(num_experts))
        reference = next(expert.parameters(), None)
        device = reference.device if reference is not None else None
        self.router = nn.Linear(d_model, num_experts, bias=False, device=device)
        self.register_buffer(
            "last_expert_counts",
            torch.zeros(num_experts, dtype=torch.long, device=device),
            persistent=False,
        )
        self.aux_loss = self.router.weight.new_zeros((), dtype=torch.float32)
        self._token_mask: Optional[torch.Tensor] = None

    def set_token_mask(self, mask: Optional[torch.Tensor]):
        """Set a valid-token mask; nonzero values denote valid tokens."""
        if mask is not None and not isinstance(mask, torch.Tensor):
            raise TypeError("token mask must be a tensor or None")
        self._token_mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2 or x.shape[-1] != self.d_model:
            raise ValueError("x must have token dimensions and final dimension d_model")
        flat_x = x.reshape(-1, self.d_model)
        if self._token_mask is None:
            valid_indices = torch.arange(flat_x.shape[0], device=x.device)
        else:
            if self._token_mask.shape != x.shape[:-1]:
                raise ValueError("token mask shape must match the input's leading dimensions")
            valid_mask = self._token_mask.to(device=x.device, dtype=torch.bool).reshape(-1)
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]

        self.last_expert_counts = torch.zeros(
            self.num_experts, dtype=torch.long, device=x.device
        )
        if valid_indices.numel() == 0:
            # Empty sums connect both graphs without reading possibly NaN padding.
            zero = flat_x[:0].float().sum() + self.router.weight[:0].float().sum()
            self.aux_loss = zero
            return torch.zeros_like(x) + zero.to(x.dtype)

        valid_x = flat_x.index_select(0, valid_indices)
        # Disabling autocast also protects routing when the enclosing T5 uses AMP.
        # Explicit casts retain float32 routing after model.half()/bfloat16().
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(valid_x.float(), self.router.weight.float())
            probabilities = F.softmax(logits, dim=-1)
            selected_weights, selected_experts = torch.topk(
                probabilities, self.top_k, dim=-1
            )
            selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True)
            self.last_expert_counts = torch.bincount(
                selected_experts.reshape(-1), minlength=self.num_experts
            ).detach()
            assignment_fraction = self.last_expert_counts.float() / (
                valid_indices.numel() * self.top_k
            )
            self.aux_loss = self.num_experts * torch.sum(
                probabilities.mean(dim=0) * assignment_fraction
            )

        flat_output = torch.zeros_like(flat_x)
        for expert_index, expert in enumerate(self.experts):
            token_rows, selected_slots = (selected_experts == expert_index).nonzero(as_tuple=True)
            if token_rows.numel() == 0:
                continue
            expert_input = valid_x.index_select(0, token_rows)
            expert_output = expert(expert_input)
            weights = selected_weights[token_rows, selected_slots, None]
            weighted_output = (expert_output.float() * weights).to(flat_output.dtype)
            flat_output.index_add_(
                0, valid_indices.index_select(0, token_rows), weighted_output
            )
        return flat_output.reshape_as(x)
