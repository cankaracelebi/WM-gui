from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class WorldModel(ABC, nn.Module):
    """Abstract base for learned world models."""

    @abstractmethod
    def encode(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode frames -> (mu, logvar) of latent distribution."""
        ...

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors -> reconstructed frames."""
        ...

    @abstractmethod
    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass -> (reconstruction, mu, logvar)."""
        ...

    @abstractmethod
    def dream_step(self, z: torch.Tensor, action: int, num_actions: int) -> torch.Tensor:
        """Given current latent + action, predict next latent."""
        ...
