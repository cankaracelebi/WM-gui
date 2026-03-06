from abc import ABC, abstractmethod
import numpy as np


class Environment(ABC):
    """Abstract base for visual game environments."""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment, return initial frame (H, W, 3) uint8."""
        ...

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Take action, return (frame, reward, done, info)."""
        ...

    @abstractmethod
    def render(self) -> np.ndarray:
        """Return current frame as (H, W, 3) uint8 numpy array."""
        ...

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        ...

    @property
    @abstractmethod
    def frame_shape(self) -> tuple[int, int, int]:
        ...

    @property
    @abstractmethod
    def action_names(self) -> list[str]:
        ...
