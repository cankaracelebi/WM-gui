from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QProgressBar, QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Signal

from gui.widgets.plot_widget import PlotWidget


class TrainingPanel(QWidget):
    """Panel for training controls and loss visualization."""

    train_requested = Signal()
    pause_requested = Signal()
    collect_requested = Signal(int)  # num episodes

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Title
        title = QLabel("Training")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #e94560;")
        layout.addWidget(title)

        # Loss plot
        self.loss_plot = PlotWidget(title="Loss Curves", width=5, height=3)
        layout.addWidget(self.loss_plot)

        # Controls
        controls = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout(controls)

        # Train/Pause
        btn_row = QHBoxLayout()
        self.train_btn = QPushButton("Train")
        self.train_btn.clicked.connect(self._on_train)
        btn_row.addWidget(self.train_btn)

        self.collect_btn = QPushButton("Collect 500 Frames")
        self.collect_btn.clicked.connect(lambda: self.collect_requested.emit(500))
        btn_row.addWidget(self.collect_btn)
        ctrl_layout.addLayout(btn_row)

        # Hyperparameters
        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Batch:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(8, 256)
        self.batch_spin.setValue(32)
        param_row.addWidget(self.batch_spin)

        param_row.addWidget(QLabel("LR:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.01)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.001)
        param_row.addWidget(self.lr_spin)
        ctrl_layout.addLayout(param_row)

        # Buffer progress
        buf_row = QHBoxLayout()
        buf_row.addWidget(QLabel("Buffer:"))
        self.buffer_bar = QProgressBar()
        self.buffer_bar.setRange(0, 50000)
        self.buffer_bar.setValue(0)
        buf_row.addWidget(self.buffer_bar)
        ctrl_layout.addLayout(buf_row)

        # Status
        self.status_label = QLabel("Idle | Step: 0 | Beta: 0.000")
        self.status_label.setStyleSheet("color: #a0a0c0; font-size: 11px;")
        ctrl_layout.addWidget(self.status_label)

        layout.addWidget(controls)
        layout.addStretch()

        self._training = False

    def _on_train(self):
        if self._training:
            self._training = False
            self.train_btn.setText("Train")
            self.pause_requested.emit()
        else:
            self._training = True
            self.train_btn.setText("Pause")
            self.train_requested.emit()

    def update_metrics(self, metrics: dict):
        """Called from main thread with new training metrics."""
        step = metrics.get("step", 0)
        beta = metrics.get("beta", 0)
        buf = metrics.get("buffer_size", 0)

        self.buffer_bar.setValue(buf)
        status = f"Training | Step: {step} | Beta: {beta:.3f} | Buffer: {buf}"
        self.status_label.setText(status)

    def update_loss_plot(self, loss_hist: list, recon_hist: list, kl_hist: list):
        self.loss_plot.update_plot({
            "Total Loss": loss_hist,
            "Recon Loss": recon_hist,
            "KL Loss": kl_hist,
        })

    def set_idle(self):
        self._training = False
        self.train_btn.setText("Train")
