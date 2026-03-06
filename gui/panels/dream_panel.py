import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSlider, QComboBox, QGroupBox, QScrollArea
)
from PySide6.QtCore import Qt, QTimer, Signal

from gui.widgets.frame_display import FrameDisplay


class FilmstripWidget(QScrollArea):
    """Horizontal strip of dream frame thumbnails."""

    frame_selected = Signal(int)

    def __init__(self, thumb_size: int = 48, parent=None):
        super().__init__(parent)
        self.thumb_size = thumb_size
        self.setFixedHeight(thumb_size + 16)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("background: #0a0a1e; border: 1px solid #0f3460; border-radius: 4px;")

        self._container = QWidget()
        self._layout = QHBoxLayout(self._container)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)
        self.setWidget(self._container)

        self.thumbnails: list[FrameDisplay] = []

    def clear(self):
        for thumb in self.thumbnails:
            self._layout.removeWidget(thumb)
            thumb.deleteLater()
        self.thumbnails.clear()

    def add_frame(self, frame: np.ndarray, index: int):
        thumb = FrameDisplay(self.thumb_size)
        thumb.setStyleSheet("border: 1px solid #0f3460; border-radius: 2px;")
        thumb.update_frame(frame)
        thumb.mousePressEvent = lambda e, idx=index: self.frame_selected.emit(idx)
        self._layout.addWidget(thumb)
        self.thumbnails.append(thumb)
        # Auto-scroll to end
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().maximum())


class DreamPanel(QWidget):
    """Panel for viewing and controlling dream sequences."""

    dream_start_requested = Signal()
    dream_stop_requested = Signal()

    def __init__(self, display_size: int = 256, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Title
        title = QLabel("Dream Sequence")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #e94560;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Dream display
        self.frame_display = FrameDisplay(display_size)
        layout.addWidget(self.frame_display, alignment=Qt.AlignmentFlag.AlignCenter)

        # Filmstrip
        self.filmstrip = FilmstripWidget(thumb_size=48)
        layout.addWidget(self.filmstrip)

        # Controls
        controls = QGroupBox("Dream Controls")
        ctrl_layout = QVBoxLayout(controls)

        # Buttons
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Dream")
        self.start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self.stop_btn)
        ctrl_layout.addLayout(btn_row)

        # Dream policy
        policy_row = QHBoxLayout()
        policy_row.addWidget(QLabel("Policy:"))
        self.policy_combo = QComboBox()
        self.policy_combo.addItems(["Random", "No-op"])
        policy_row.addWidget(self.policy_combo)
        ctrl_layout.addLayout(policy_row)

        # Steps slider
        steps_row = QHBoxLayout()
        steps_row.addWidget(QLabel("Steps:"))
        self.steps_slider = QSlider(Qt.Orientation.Horizontal)
        self.steps_slider.setRange(10, 500)
        self.steps_slider.setValue(100)
        steps_row.addWidget(self.steps_slider)
        self.steps_label = QLabel("100")
        self.steps_slider.valueChanged.connect(lambda v: self.steps_label.setText(str(v)))
        steps_row.addWidget(self.steps_label)
        ctrl_layout.addLayout(steps_row)

        # Temperature slider
        temp_row = QHBoxLayout()
        temp_row.addWidget(QLabel("Temperature:"))
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(10, 200)  # 0.1 to 2.0
        self.temp_slider.setValue(100)
        temp_row.addWidget(self.temp_slider)
        self.temp_label = QLabel("1.00")
        self.temp_slider.valueChanged.connect(
            lambda v: self.temp_label.setText(f"{v / 100:.2f}")
        )
        temp_row.addWidget(self.temp_label)
        ctrl_layout.addLayout(temp_row)

        # Speed slider
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Playback:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(20, 500)
        self.speed_slider.setValue(100)
        speed_row.addWidget(self.speed_slider)
        self.playback_label = QLabel("100ms")
        self.speed_slider.valueChanged.connect(
            lambda v: self.playback_label.setText(f"{v}ms")
        )
        speed_row.addWidget(self.playback_label)
        ctrl_layout.addLayout(speed_row)

        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #a0a0c0; font-size: 11px;")
        ctrl_layout.addWidget(self.status_label)

        layout.addWidget(controls)
        layout.addStretch()

        # Playback state
        self.dream_frames: list[np.ndarray] = []
        self.playback_idx = 0
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._playback_tick)
        self._dreaming = False

    @property
    def temperature(self) -> float:
        return self.temp_slider.value() / 100.0

    @property
    def num_steps(self) -> int:
        return self.steps_slider.value()

    @property
    def dream_policy(self) -> str:
        return self.policy_combo.currentText().lower().replace("-", "")

    def _on_start(self):
        self._dreaming = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.filmstrip.clear()
        self.dream_frames.clear()
        self.playback_idx = 0
        self.status_label.setText("Dreaming...")
        self.dream_start_requested.emit()

    def _on_stop(self):
        self._dreaming = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.playback_timer.stop()
        self.status_label.setText(f"Stopped at frame {self.playback_idx}")
        self.dream_stop_requested.emit()

    def receive_dream_frames(self, frames: list[np.ndarray]):
        """Called when dream worker completes. Starts playback."""
        self.dream_frames = frames
        self.playback_idx = 0
        self.status_label.setText(f"Playing {len(frames)} dream frames...")

        # Add thumbnails for every 10th frame
        self.filmstrip.clear()
        for i in range(0, len(frames), max(1, len(frames) // 20)):
            self.filmstrip.add_frame(frames[i], i)

        # Start playback
        self.playback_timer.start(self.speed_slider.value())

    def _playback_tick(self):
        if self.playback_idx >= len(self.dream_frames):
            self.playback_timer.stop()
            self._on_stop()
            self.status_label.setText(f"Dream complete ({len(self.dream_frames)} frames)")
            return

        self.frame_display.update_frame(self.dream_frames[self.playback_idx])
        self.status_label.setText(
            f"Frame {self.playback_idx + 1}/{len(self.dream_frames)}"
        )
        self.playback_idx += 1

    def show_single_frame(self, frame: np.ndarray):
        self.frame_display.update_frame(frame)
