import numpy as np
import torch


class ReplayBuffer:
    """Circular replay buffer storing (frame, action, next_frame) transitions."""

    def __init__(self, capacity: int, frame_shape: tuple[int, int, int] = (64, 64, 3)):
        self.capacity = capacity
        self.frames = np.zeros((capacity, *frame_shape), dtype=np.uint8)
        self.next_frames = np.zeros((capacity, *frame_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.idx = 0
        self.size = 0

    def add(self, frame: np.ndarray, action: int, next_frame: np.ndarray):
        self.frames[self.idx] = frame
        self.actions[self.idx] = action
        self.next_frames[self.idx] = next_frame
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch. Returns (frames, actions, next_frames) as tensors."""
        indices = np.random.randint(0, self.size, size=batch_size)

        frames = self.frames[indices].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        next_frames = self.next_frames[indices].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        actions = self.actions[indices]

        return (
            torch.from_numpy(frames),
            torch.from_numpy(actions),
            torch.from_numpy(next_frames),
        )

    def __len__(self) -> int:
        return self.size
