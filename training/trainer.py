import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from models.conv_vae import ConvVAE, vae_loss
from environments.base import Environment
from .replay_buffer import ReplayBuffer
from utils.config import AppConfig


class Trainer:
    """Orchestrates data collection and training of the world model."""

    def __init__(self, model: ConvVAE, config: AppConfig, device: str):
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=config.learning_rate)
        self.buffer = ReplayBuffer(config.buffer_size)
        self.train_step_count = 0

        # Metrics history
        self.loss_history: list[float] = []
        self.recon_history: list[float] = []
        self.kl_history: list[float] = []
        self.transition_history: list[float] = []

    @property
    def beta(self) -> float:
        """Current beta for KL annealing."""
        if self.train_step_count >= self.config.beta_anneal_steps:
            return self.config.beta_end
        t = self.train_step_count / self.config.beta_anneal_steps
        return self.config.beta_start + t * (self.config.beta_end - self.config.beta_start)

    def collect_episode(self, env: Environment, max_steps: int = 200, policy: str = "random") -> int:
        """Run one episode, store transitions. Returns number of frames collected."""
        frame = env.reset()
        collected = 0
        for _ in range(max_steps):
            if policy == "random":
                action = random.randint(0, env.action_space_size - 1)
            else:
                action = 0  # noop
            next_frame, _, done, _ = env.step(action)
            self.buffer.add(frame, action, next_frame)
            frame = next_frame
            collected += 1
            if done:
                break
        return collected

    def train_step(self) -> dict:
        """Single training step. Returns metrics dict."""
        if len(self.buffer) < self.config.batch_size:
            return {}

        self.model.train()
        frames, actions, next_frames = self.buffer.sample(self.config.batch_size)
        frames = frames.to(self.device)
        actions = actions.to(self.device)
        next_frames = next_frames.to(self.device)

        # VAE forward
        recon, mu, logvar = self.model(frames)
        total_loss, recon_loss, kl_loss = vae_loss(recon, frames, mu, logvar, self.beta)

        # Transition model
        z = self.model.reparameterize(mu, logvar).detach()
        action_onehot = F.one_hot(actions, self.model.num_actions).float()
        predicted_next_z = self.model.predict_next_latent(z, action_onehot)

        with torch.no_grad():
            target_mu, _ = self.model.encode(next_frames)
        transition_loss = F.mse_loss(predicted_next_z, target_mu)

        # Combined loss
        combined_loss = total_loss + self.config.transition_weight * transition_loss

        self.optimizer.zero_grad()
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        self.train_step_count += 1

        metrics = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "transition_loss": transition_loss.item(),
            "beta": self.beta,
            "buffer_size": len(self.buffer),
            "step": self.train_step_count,
        }

        self.loss_history.append(metrics["total_loss"])
        self.recon_history.append(metrics["recon_loss"])
        self.kl_history.append(metrics["kl_loss"])
        self.transition_history.append(metrics["transition_loss"])

        return metrics

    def get_reconstruction(self, frame: np.ndarray) -> np.ndarray:
        """Get VAE reconstruction of a single frame. Returns (H, W, 3) uint8."""
        self.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(
                frame.transpose(2, 0, 1).astype(np.float32) / 255.0
            ).unsqueeze(0).to(self.device)
            recon, _, _ = self.model(tensor)
            recon = recon[0].cpu().clamp(0, 1).mul(255).byte()
            return recon.permute(1, 2, 0).numpy()

    def get_latent(self, frame: np.ndarray) -> np.ndarray:
        """Encode a frame and return its latent vector as numpy."""
        self.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(
                frame.transpose(2, 0, 1).astype(np.float32) / 255.0
            ).unsqueeze(0).to(self.device)
            mu, _ = self.model.encode(tensor)
            return mu[0].cpu().numpy()
