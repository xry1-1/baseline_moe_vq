"""Input vector quantization with an explicitly collected final-weight memory bank."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class VQOutput:
    hidden_states: Tensor
    loss: Tensor
    codebook_loss: Tensor
    commitment_loss: Tensor
    indices: Tensor
    perplexity: Tensor


class VQVAE(nn.Module):
    """Quantize visible patch embeddings and preserve hidden/padded tokens exactly.

    ``valid_mask`` broadcasts to ``x.shape[:-1]``. Component losses are raw MSEs;
    ``loss = codebook_loss + commitment_cost * commitment_loss``. The decoder is
    trained through the downstream task objective, with straight-through encoder
    gradients. Only explicit memory collection changes ``memory_counts``.
    """

    def __init__(self, d_model: int, latent_dim: int = 64,
                 codebook_size: int = 256, commitment_cost: float = 0.25):
        super().__init__()
        for name, value in (("d_model", d_model), ("latent_dim", latent_dim),
                            ("codebook_size", codebook_size)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isfinite(commitment_cost) or commitment_cost < 0:
            raise ValueError("commitment_cost must be finite and nonnegative")
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.commitment_cost = float(commitment_cost)
        self.encoder = nn.Linear(d_model, latent_dim)
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        self.decoder = nn.Linear(latent_dim, d_model)
        nn.init.uniform_(self.codebook.weight, -1. / codebook_size, 1. / codebook_size)
        self.register_buffer("activation_counts", torch.zeros(codebook_size, dtype=torch.long))
        self.register_buffer("memory_counts", torch.zeros(codebook_size, dtype=torch.long))
        self._collect_memory = False

    @property
    def memory_indices(self) -> Tensor:
        """Sorted code indices activated during the most recent explicit pass."""
        return torch.nonzero(self.memory_counts > 0, as_tuple=False).flatten()

    @property
    def memory_bank(self) -> Tensor:
        """Final code vectors selected by the persistent memory histogram."""
        return self.codebook.weight[self.memory_indices]

    @torch.no_grad()
    def begin_memory_collection(self) -> None:
        """Start a fresh pass with final weights and training inputs only.

        The caller owns the data split and should normally use eval/no_grad for
        this pass. Historical activation counts never seed the memory bank.
        """
        self.memory_counts.zero_()
        self._collect_memory = True

    def end_memory_collection(self) -> None:
        self._collect_memory = False

    def forward(self, x: Tensor, valid_mask: Optional[Tensor] = None) -> VQOutput:
        if x.ndim < 2 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must end in d_model={self.d_model} and include a token dimension")
        if not x.is_floating_point():
            raise ValueError("x must be a floating-point tensor")
        if valid_mask is None:
            mask = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
        else:
            try:
                mask = torch.broadcast_to(valid_mask.to(device=x.device, dtype=torch.bool), x.shape[:-1])
            except RuntimeError as error:
                raise ValueError("valid_mask must broadcast to x.shape[:-1]") from error
        flat_mask = mask.reshape(-1)
        indices = torch.full(x.shape[:-1], -1, dtype=torch.long, device=x.device)
        if not flat_mask.any():
            # Empty sums preserve a backward graph without reading NaN placeholders.
            zero = x.reshape(-1)[:0].float().sum()
            for parameter in self.parameters():
                zero = zero + parameter.reshape(-1)[:0].float().sum()
            return VQOutput(x, zero, zero, zero, indices, zero.detach())

        flat_x = x.reshape(-1, self.d_model)
        encoded = self.encoder(flat_x[flat_mask])
        # Explicitly disable outer AMP so nearest-code decisions and both fitting
        # losses always use float32, including matmul on CUDA/CPU autocast paths.
        with torch.autocast(device_type=x.device.type, enabled=False):
            latent = encoded.float()
            codebook = self.codebook.weight.float()
            distances = (latent.square().sum(dim=1, keepdim=True)
                         + codebook.square().sum(dim=1).unsqueeze(0)
                         - 2. * latent @ codebook.t())
            selected = distances.argmin(dim=1)
            quantized = F.embedding(selected, codebook)
            codebook_loss = F.mse_loss(quantized, latent.detach())
            commitment_loss = F.mse_loss(latent, quantized.detach())
            loss = codebook_loss + self.commitment_cost * commitment_loss
            straight_through = latent + (quantized - latent).detach()
        decoded = self.decoder(straight_through.to(dtype=encoded.dtype)).to(dtype=x.dtype)
        # Flatten before cloning so assignment also works for noncontiguous x.
        hidden_states = flat_x.clone()
        hidden_states[flat_mask] = decoded
        indices.reshape(-1)[flat_mask] = selected

        with torch.no_grad():
            counts = torch.bincount(selected, minlength=self.codebook_size)
            if self.training:
                self.activation_counts.add_(counts)
            if self._collect_memory:
                self.memory_counts.add_(counts)
            probabilities = counts.float() / selected.numel()
            nonzero = probabilities[probabilities > 0]
            perplexity = torch.exp(-(nonzero * nonzero.log()).sum())
        return VQOutput(hidden_states.reshape_as(x), loss, codebook_loss,
                        commitment_loss, indices, perplexity)

    def _memory_config(self) -> dict:
        return dict(d_model=self.d_model, latent_dim=self.latent_dim,
                    codebook_size=self.codebook_size, commitment_cost=self.commitment_cost)

    def export_memory(self, path: Union[str, Path]) -> None:
        """Save codebook vectors, final-pass indices/counts, and configuration.

        Full model checkpoints are still needed to restore encoder/decoder weights.
        This file contains only tensors and basic metadata and supports safe loading.
        """
        payload = dict(
            format_version=1,
            config=self._memory_config(),
            codebook=self.codebook.weight.detach().cpu().clone(),
            memory_bank=self.memory_bank.detach().cpu().clone(),
            memory_indices=self.memory_indices.detach().cpu().clone(),
            memory_counts=self.memory_counts.detach().cpu().clone(),
            activation_counts=self.activation_counts.detach().cpu().clone(),
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @torch.no_grad()
    def load_memory(self, path: Union[str, Path]) -> None:
        """Restore a compatible export after validating all tensors atomically."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError("Unsupported VQ memory format")
        if payload.get("config") != self._memory_config():
            raise ValueError("VQ memory configuration does not match this module")
        codebook = payload.get("codebook")
        if (not isinstance(codebook, Tensor) or codebook.shape != (self.codebook_size, self.latent_dim)
                or not codebook.is_floating_point() or not torch.isfinite(codebook).all()):
            raise ValueError("Invalid codebook tensor in VQ memory")
        counts = payload.get("memory_counts")
        activation_counts = payload.get("activation_counts")
        for name, value in (("memory_counts", counts), ("activation_counts", activation_counts)):
            if (not isinstance(value, Tensor) or value.shape != (self.codebook_size,)
                    or value.dtype != torch.long or (value < 0).any()):
                raise ValueError(f"Invalid {name} tensor in VQ memory")
        indices = payload.get("memory_indices")
        expected_indices = torch.nonzero(counts > 0, as_tuple=False).flatten()
        if (not isinstance(indices, Tensor) or indices.ndim != 1 or indices.dtype != torch.long
                or not torch.equal(indices, expected_indices)):
            raise ValueError("VQ memory indices do not match its counts")
        memory_bank = payload.get("memory_bank")
        expected_bank = codebook[indices]
        if (not isinstance(memory_bank, Tensor) or memory_bank.shape != expected_bank.shape
                or memory_bank.dtype != codebook.dtype or not torch.equal(memory_bank, expected_bank)):
            raise ValueError("VQ memory bank does not match its codebook and indices")
        self.codebook.weight.copy_(codebook)
        self.memory_counts.copy_(counts)
        self.activation_counts.copy_(activation_counts)
        self._collect_memory = False
