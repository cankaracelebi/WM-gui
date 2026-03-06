import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QStatusBar, QTabWidget
)
from PySide6.QtCore import Qt, QTimer
import torch

from utils.config import AppConfig
from utils.image import postprocess
from environments.cosmic_drift import CosmicDriftEnv
from models.conv_vae import ConvVAE
from training.trainer import Trainer
from dreaming.dreamer import Dreamer
from utils.workers import TrainingWorker, DreamWorker, DecodeWorker

from gui.panels.environment_panel import EnvironmentPanel
from gui.panels.training_panel import TrainingPanel
from gui.panels.reconstruction_panel import ReconstructionPanel
from gui.panels.dream_panel import DreamPanel
from gui.panels.latent_panel import LatentPanel


class MainWindow(QMainWindow):
    """Main application window with dashboard layout."""

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.setWindowTitle("World Model Visualizer")
        self.setMinimumSize(1200, 800)

        # Core components
        self.env = CosmicDriftEnv(size=config.frame_size)
        self.model = ConvVAE(
            latent_dim=config.latent_dim,
            num_actions=self.env.action_space_size
        ).to(config.device)
        self.trainer = Trainer(self.model, config, config.device)
        self.dreamer = Dreamer(self.model, config.device, self.env.action_space_size)

        self._setup_ui()
        self._setup_workers()
        self._connect_signals()
        self._setup_timers()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Tab widget for switching between dashboard and individual panels
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Dashboard tab (default)
        dashboard = QWidget()
        dash_layout = QHBoxLayout(dashboard)
        dash_layout.setContentsMargins(0, 0, 0, 0)

        # Left column: environment + dream
        left_splitter = QSplitter(Qt.Orientation.Vertical)
        self.env_panel = EnvironmentPanel(self.env, self.config.display_size)
        left_splitter.addWidget(self.env_panel)

        self.dream_panel = DreamPanel(self.config.display_size)
        left_splitter.addWidget(self.dream_panel)
        left_splitter.setSizes([400, 400])

        # Right column: reconstruction + training + latent
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.recon_panel = ReconstructionPanel(display_size=192)
        right_splitter.addWidget(self.recon_panel)

        self.training_panel = TrainingPanel()
        right_splitter.addWidget(self.training_panel)

        self.latent_panel = LatentPanel(latent_dim=self.config.latent_dim)
        right_splitter.addWidget(self.latent_panel)
        right_splitter.setSizes([300, 250, 250])

        # Main horizontal splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([400, 600])
        dash_layout.addWidget(main_splitter)

        self.tabs.addTab(dashboard, "Dashboard")

        # Individual tabs for each panel (same instances, just wrapped)
        # Note: Qt doesn't allow the same widget in two parents,
        # so the individual tabs are just for future expansion

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(f"Device: {self.config.device} | Ready")

    def _setup_workers(self):
        # Training worker
        self.training_worker = TrainingWorker(self.trainer, self.env)
        self.training_worker.step_completed.connect(self._on_train_step)
        self.training_worker.collection_done.connect(self._on_collection_done)

        # Dream worker
        self.dream_worker = DreamWorker(self.dreamer)
        self.dream_worker.dream_complete.connect(self._on_dream_complete)

        # Decode worker
        self.decode_worker = DecodeWorker(self.model, self.config.device)
        self.decode_worker.frame_ready.connect(self.latent_panel.update_preview)

    def _connect_signals(self):
        # Environment -> reconstruction
        self.env_panel.frame_updated.connect(self._on_frame_updated)
        self.env_panel.action_taken.connect(self._on_action_taken)

        # Training controls
        self.training_panel.train_requested.connect(self._start_training)
        self.training_panel.pause_requested.connect(self._stop_training)
        self.training_panel.collect_requested.connect(self._collect_data)

        # Dream controls
        self.dream_panel.dream_start_requested.connect(self._start_dream)
        self.dream_panel.dream_stop_requested.connect(self._stop_dream)

        # Latent panel
        self.latent_panel.decode_requested.connect(self._decode_latent)

    def _setup_timers(self):
        # Plot refresh timer
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self._refresh_plots)
        self.plot_timer.start(self.config.plot_update_interval_ms)

        # Latent cache timer (add latent every N frames)
        self._frame_counter = 0
        self._latent_sample_interval = 5

    def _on_frame_updated(self, frame: np.ndarray):
        """Called when environment produces a new frame."""
        # Update reconstruction if model has been trained
        if self.trainer.train_step_count > 0:
            recon = self.trainer.get_reconstruction(frame)
            self.recon_panel.update_frames(frame, recon)

            # Sample latent for latent panel
            self._frame_counter += 1
            if self._frame_counter % self._latent_sample_interval == 0:
                z = self.trainer.get_latent(frame)
                self.latent_panel.add_latent(z, frame)

    def _on_action_taken(self, action: int, prev_frame: np.ndarray, next_frame: np.ndarray):
        """Store transition in buffer if training is active."""
        if self.training_worker.isRunning():
            self.trainer.buffer.add(prev_frame, action, next_frame)

    def _on_train_step(self, metrics: dict):
        """Called from training worker on each completed step."""
        self.training_panel.update_metrics(metrics)
        step = metrics.get("step", 0)
        if step % 50 == 0:
            self.status_bar.showMessage(
                f"Device: {self.config.device} | Training step {step} | "
                f"Loss: {metrics.get('total_loss', 0):.4f}"
            )

    def _on_collection_done(self, n: int):
        self.status_bar.showMessage(f"Collected {n} frames. Buffer: {len(self.trainer.buffer)}")

    def _refresh_plots(self):
        if self.trainer.loss_history:
            self.training_panel.update_loss_plot(
                self.trainer.loss_history,
                self.trainer.recon_history,
                self.trainer.kl_history,
            )

    def _start_training(self):
        if not self.training_worker.isRunning():
            self.training_worker.start()
        self.status_bar.showMessage("Training started...")

    def _stop_training(self):
        self.training_worker.stop()
        self.training_panel.set_idle()
        self.status_bar.showMessage("Training paused.")

    def _collect_data(self, num_episodes: int):
        if not self.training_worker.isRunning():
            self.training_worker.start()
        self.training_worker.request_collection(num_episodes)
        self.status_bar.showMessage(f"Collecting {num_episodes} episodes...")

    def _start_dream(self):
        if self.trainer.train_step_count == 0:
            self.dream_panel.status_label.setText("Train the model first!")
            self.dream_panel._on_stop()
            return

        frame = self.env_panel.current_frame
        if frame is None:
            return

        self.dream_worker.configure(
            initial_frame=frame,
            num_steps=self.dream_panel.num_steps,
            policy=self.dream_panel.dream_policy,
            temperature=self.dream_panel.temperature,
            latent_clip=self.config.dream_latent_clip,
        )
        self.dream_worker.start()
        self.status_bar.showMessage("Dreaming...")

    def _stop_dream(self):
        if self.dream_worker.isRunning():
            self.dream_worker.terminate()

    def _on_dream_complete(self, frames: list):
        self.dream_panel.receive_dream_frames(frames)
        self.status_bar.showMessage(f"Dream complete: {len(frames)} frames")

    def _decode_latent(self, z):
        """Decode a latent vector and show in latent panel preview."""
        if isinstance(z, np.ndarray):
            z = z.astype(np.float32)
        self.decode_worker.set_latent(z)
        if not self.decode_worker.isRunning():
            self.decode_worker.start()

    def closeEvent(self, event):
        self.training_worker.stop()
        if self.dream_worker.isRunning():
            self.dream_worker.terminate()
        self.env_panel.pause()
        event.accept()
