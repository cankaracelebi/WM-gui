import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox
)
from PySide6.QtCore import Qt

from gui.widgets.frame_display import FrameDisplay


class ReconstructionPanel(QWidget):
    """Side-by-side comparison of original vs VAE reconstruction."""

    def __init__(self, display_size: int = 192, parent=None):
        super().__init__(parent)
        self.display_size = display_size
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Title
        title = QLabel("Reconstruction")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #e94560;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Frame displays side by side
        frames_row = QHBoxLayout()

        # Original
        orig_col = QVBoxLayout()
        orig_label = QLabel("Original")
        orig_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        orig_label.setStyleSheet("color: #00d4ff; font-weight: bold;")
        orig_col.addWidget(orig_label)
        self.original_display = FrameDisplay(display_size)
        orig_col.addWidget(self.original_display)
        frames_row.addLayout(orig_col)

        # Reconstructed
        recon_col = QVBoxLayout()
        recon_label = QLabel("Reconstructed")
        recon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        recon_label.setStyleSheet("color: #50fa7b; font-weight: bold;")
        recon_col.addWidget(recon_label)
        self.recon_display = FrameDisplay(display_size)
        recon_col.addWidget(self.recon_display)
        frames_row.addLayout(recon_col)

        layout.addLayout(frames_row)

        # Difference view
        diff_row = QHBoxLayout()
        self.diff_check = QCheckBox("Show Difference Map")
        self.diff_check.setStyleSheet("color: #a0a0c0;")
        diff_row.addWidget(self.diff_check)
        diff_row.addStretch()
        layout.addLayout(diff_row)

        self.diff_display = FrameDisplay(display_size)
        self.diff_display.setVisible(False)
        layout.addWidget(self.diff_display, alignment=Qt.AlignmentFlag.AlignCenter)

        self.diff_check.toggled.connect(self.diff_display.setVisible)

        # Metrics
        self.metrics_label = QLabel("MSE: -- | PSNR: --")
        self.metrics_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.metrics_label.setStyleSheet("color: #ffb86c; font-size: 13px; font-weight: bold;")
        layout.addWidget(self.metrics_label)

        layout.addStretch()

        self._last_original = None
        self._last_recon = None

    def update_frames(self, original: np.ndarray, reconstructed: np.ndarray):
        """Update both displays and compute metrics."""
        self._last_original = original
        self._last_recon = reconstructed

        self.original_display.update_frame(original)
        self.recon_display.update_frame(reconstructed)

        # Compute metrics
        mse = float(np.mean((original.astype(float) - reconstructed.astype(float)) ** 2))
        psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-10)) if mse > 0 else float("inf")
        self.metrics_label.setText(f"MSE: {mse:.1f} | PSNR: {psnr:.1f} dB")

        # Difference map (amplified 5x)
        if self.diff_check.isChecked():
            diff = np.abs(original.astype(float) - reconstructed.astype(float))
            diff = np.clip(diff * 5, 0, 255).astype(np.uint8)
            self.diff_display.update_frame(diff)
