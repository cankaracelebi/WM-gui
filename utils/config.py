from dataclasses import dataclass, field
import torch


@dataclass
class AppConfig:
    # Environment
    frame_size: int = 64
    env_step_delay_ms: int = 50

    # Model
    latent_dim: int = 32

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    buffer_size: int = 50_000
    beta_start: float = 0.0
    beta_end: float = 1.0
    beta_anneal_steps: int = 5000
    collect_before_train: int = 500
    transition_weight: float = 0.1

    # Dream
    dream_temperature: float = 1.0
    dream_max_steps: int = 200
    dream_latent_clip: float = 3.0

    # GUI
    display_size: int = 256
    plot_update_interval_ms: int = 500

    # System
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
