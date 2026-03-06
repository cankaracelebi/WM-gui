import numpy as np
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt


class FrameDisplay(QLabel):
    """Widget that displays a numpy (H, W, 3) uint8 frame, upscaled with nearest-neighbor."""

    def __init__(self, display_size: int = 256, parent=None):
        super().__init__(parent)
        self.display_size = display_size
        self.setFixedSize(display_size, display_size)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            f"border: 2px solid #e94560; border-radius: 4px; background: #0a0a1e;"
        )
        self._show_placeholder()

    def _show_placeholder(self):
        self.setText("No Frame")
        self.setStyleSheet(
            self.styleSheet() + "color: #606080; font-size: 14px;"
        )

    def update_frame(self, frame: np.ndarray):
        """Update display with a new frame. frame: (H, W, 3) uint8 numpy array."""
        if frame is None:
            self._show_placeholder()
            return

        frame = np.ascontiguousarray(frame)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(
            self.display_size, self.display_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation  # nearest-neighbor for pixel-art look
        )
        self.setPixmap(pixmap)
