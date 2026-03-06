import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import WorldModel


class ConvVAE(WorldModel):
    """
    Convolutional VAE with a transition model for world modeling.

    Input: (B, 3, 64, 64) float32 in [0, 1]
    Latent: (B, latent_dim) float32
    Output: (B, 3, 64, 64) float32 in [0, 1]
    """

    def __init__(self, latent_dim: int = 32, num_actions: int = 5):
        nn.Module.__init__(self)
        self.latent_dim = latent_dim
        self.num_actions = num_actions

        # Encoder: 4 conv layers, stride 2
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Flatten size: 256 * 4 * 4 = 4096
        self.fc_mu = nn.Linear(4096, latent_dim)
        self.fc_logvar = nn.Linear(4096, latent_dim)

        # Decoder
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, 4096),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        # Transition model: predicts delta-z given (z, action_onehot)
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + num_actions, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )

    def encode(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(frames)
        h = h.reshape(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, 256, 4, 4)
        return self.decoder(h)

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(frames)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def dream_step(self, z: torch.Tensor, action: int, num_actions: int = None) -> torch.Tensor:
        na = num_actions or self.num_actions
        action_onehot = torch.zeros(z.size(0), na, device=z.device)
        action_onehot[:, action] = 1.0
        inp = torch.cat([z, action_onehot], dim=1)
        delta_z = self.transition(inp)
        return z + delta_z

    def predict_next_latent(self, z: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        """Predict next latent given current z and one-hot action (for training)."""
        inp = torch.cat([z, action_onehot], dim=1)
        delta_z = self.transition(inp)
        return z + delta_z


def vae_loss(recon: torch.Tensor, target: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute VAE loss: MSE reconstruction + beta * KL divergence."""
    recon_loss = F.mse_loss(recon, target, reduction="sum") / target.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss
