import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex
import torch

from training.trainer import Trainer
from dreaming.dreamer import Dreamer
from environments.base import Environment


class TrainingWorker(QThread):
    """Background worker that runs training steps."""

    step_completed = Signal(dict)  # metrics from each step
    collection_done = Signal(int)  # frames collected

    def __init__(self, trainer: Trainer, env: Environment, parent=None):
        super().__init__(parent)
        self.trainer = trainer
        self.env = env
        self.mutex = QMutex()
        self._running = False
        self._collecting = False
        self._collect_episodes = 0
        self._train_steps_per_tick = 4

    def run(self):
        self._running = True
        while self._running:
            self.mutex.lock()
            collecting = self._collecting
            collect_eps = self._collect_episodes
            self.mutex.unlock()

            if collecting:
                total = 0
                for _ in range(collect_eps):
                    n = self.trainer.collect_episode(self.env)
                    total += n
                self.mutex.lock()
                self._collecting = False
                self.mutex.unlock()
                self.collection_done.emit(total)

            # Training steps
            if len(self.trainer.buffer) >= self.trainer.config.batch_size:
                for _ in range(self._train_steps_per_tick):
                    if not self._running:
                        break
                    metrics = self.trainer.train_step()
                    if metrics:
                        self.step_completed.emit(metrics)
            else:
                self.msleep(50)

    def stop(self):
        self._running = False
        self.wait(3000)

    def request_collection(self, num_episodes: int):
        self.mutex.lock()
        self._collecting = True
        self._collect_episodes = num_episodes
        self.mutex.unlock()


class DreamWorker(QThread):
    """Background worker that generates dream sequences."""

    dream_complete = Signal(list)  # list of dream frames

    def __init__(self, dreamer: Dreamer, parent=None):
        super().__init__(parent)
        self.dreamer = dreamer
        self._initial_frame: np.ndarray | None = None
        self._num_steps = 100
        self._policy = "random"
        self._temperature = 1.0
        self._latent_clip = 3.0

    def configure(self, initial_frame: np.ndarray, num_steps: int,
                  policy: str, temperature: float, latent_clip: float = 3.0):
        self._initial_frame = initial_frame
        self._num_steps = num_steps
        self._policy = policy
        self._temperature = temperature
        self._latent_clip = latent_clip

    def run(self):
        if self._initial_frame is None:
            return
        frames = self.dreamer.dream_sequence(
            self._initial_frame, self._num_steps,
            policy=self._policy,
            temperature=self._temperature,
            latent_clip=self._latent_clip,
        )
        self.dream_complete.emit(frames)


class DecodeWorker(QThread):
    """Background worker for decoding latent vectors to frames."""

    frame_ready = Signal(np.ndarray)

    def __init__(self, model, device: str, parent=None):
        super().__init__(parent)
        self.model = model
        self.device = device
        self._z: np.ndarray | None = None

    def set_latent(self, z: np.ndarray):
        self._z = z

    def run(self):
        if self._z is None:
            return
        self.model.eval()
        with torch.no_grad():
            z_tensor = torch.from_numpy(self._z).float().unsqueeze(0).to(self.device)
            frame_tensor = self.model.decode(z_tensor)
            frame = frame_tensor[0].cpu().clamp(0, 1).mul(255).byte()
            frame = frame.permute(1, 2, 0).numpy()
        self.frame_ready.emit(frame)
