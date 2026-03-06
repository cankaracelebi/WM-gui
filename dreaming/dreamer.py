import random
import numpy as np
import torch

from models.conv_vae import ConvVAE
from utils.image import preprocess, postprocess


class Dreamer:
    """Generates dream sequences by auto-regressing through the world model's latent space."""

    def __init__(self, model: ConvVAE, device: str, num_actions: int = 5):
        self.model = model
        self.device = device
        self.num_actions = num_actions
        self.current_z: torch.Tensor | None = None
        self.dream_frames: list[np.ndarray] = []

    def start_dream(self, initial_frame: np.ndarray):
        """Encode an initial real frame to seed the dream."""
        self.model.eval()
        with torch.no_grad():
            tensor = preprocess(initial_frame).unsqueeze(0).to(self.device)
            mu, _ = self.model.encode(tensor)
            self.current_z = mu
        self.dream_frames = [initial_frame.copy()]

    def dream_step(self, action: int, temperature: float = 1.0,
                   latent_clip: float = 3.0) -> np.ndarray:
        """Advance one step in dream-space. Returns decoded frame (H, W, 3) uint8."""
        if self.current_z is None:
            raise RuntimeError("Call start_dream() first")

        self.model.eval()
        with torch.no_grad():
            # Predict next latent
            self.current_z = self.model.dream_step(self.current_z, action, self.num_actions)

            # Apply temperature scaling
            if temperature != 1.0:
                self.current_z = self.current_z * temperature

            # Clip to prevent drift into untrained regions
            self.current_z = self.current_z.clamp(-latent_clip, latent_clip)

            # Decode
            frame_tensor = self.model.decode(self.current_z)
            frame = postprocess(frame_tensor)

        self.dream_frames.append(frame)
        return frame

    def dream_sequence(self, initial_frame: np.ndarray, num_steps: int,
                       policy: str = "random", temperature: float = 1.0,
                       latent_clip: float = 3.0) -> list[np.ndarray]:
        """Generate a full dream trajectory."""
        self.start_dream(initial_frame)
        frames = [initial_frame.copy()]

        for _ in range(num_steps):
            if policy == "random":
                action = random.randint(0, self.num_actions - 1)
            else:
                action = 0
            frame = self.dream_step(action, temperature, latent_clip)
            frames.append(frame)

        return frames

    def get_current_latent(self) -> np.ndarray | None:
        if self.current_z is None:
            return None
        return self.current_z[0].cpu().numpy()
