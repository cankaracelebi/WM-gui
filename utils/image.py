import numpy as np
import torch


def preprocess(frame: np.ndarray) -> torch.Tensor:
    """Convert (H, W, 3) uint8 numpy frame to (3, H, W) float32 tensor in [0, 1]."""
    return torch.from_numpy(frame.transpose(2, 0, 1).copy()).float() / 255.0


def postprocess(tensor: torch.Tensor) -> np.ndarray:
    """Convert (3, H, W) or (1, 3, H, W) float tensor to (H, W, 3) uint8 numpy."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.detach().cpu().clamp(0, 1).mul(255).byte()
    return img.permute(1, 2, 0).numpy()


def batch_preprocess(frames: list[np.ndarray]) -> torch.Tensor:
    """Convert list of (H, W, 3) uint8 frames to (B, 3, H, W) float32 tensor."""
    arr = np.stack(frames, axis=0).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    return torch.from_numpy(arr)
