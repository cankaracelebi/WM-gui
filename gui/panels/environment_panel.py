from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QSlider, QLabel, QGroupBox
)
from PySide6.QtCore import Qt, QTimer, Signal
import numpy as np

from environments.base import Environment
from gui.widgets.frame_display import FrameDisplay


class EnvironmentPanel(QWidget):
    """Panel for live game environment visualization and control."""

    frame_updated = Signal(np.ndarray)  # emitted on each new frame
    action_taken = Signal(int, np.ndarray, np.ndarray)  # action, frame_before, frame_after

    def __init__(self, env: Environment, display_size: int = 256, parent=None):
        super().__init__(parent)
        self.env = env
        self.current_frame: np.ndarray | None = None
        self.running = False
        self._pending_action: int | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Title
        title = QLabel("Live Environment")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #e94560;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Frame display
        self.frame_display = FrameDisplay(display_size)
        layout.addWidget(self.frame_display, alignment=Qt.AlignmentFlag.AlignCenter)

        # Controls
        controls = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout(controls)

        # Play/Pause/Reset row
        btn_row = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        btn_row.addWidget(self.play_btn)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.do_reset)
        btn_row.addWidget(self.reset_btn)
        ctrl_layout.addLayout(btn_row)

        # Speed slider
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(10, 200)
        self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(self._update_timer_interval)
        speed_row.addWidget(self.speed_slider)
        self.speed_label = QLabel("50ms")
        speed_row.addWidget(self.speed_label)
        ctrl_layout.addLayout(speed_row)

        # Policy dropdown
        policy_row = QHBoxLayout()
        policy_row.addWidget(QLabel("Policy:"))
        self.policy_combo = QComboBox()
        self.policy_combo.addItems(["Manual (WASD)", "Random", "No-op"])
        policy_row.addWidget(self.policy_combo)
        ctrl_layout.addLayout(policy_row)

        # Info
        self.info_label = QLabel("Step: 0 | Reward: 0.0")
        self.info_label.setStyleSheet("color: #a0a0c0; font-size: 11px;")
        ctrl_layout.addWidget(self.info_label)

        layout.addWidget(controls)
        layout.addStretch()

        # Game timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)

        self.total_reward = 0.0
        self.step_count = 0
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Initialize
        self.do_reset()

    def toggle_play(self):
        if self.running:
            self.pause()
        else:
            self.play()

    def play(self):
        self.running = True
        self.play_btn.setText("Pause")
        self.timer.start(self.speed_slider.value())

    def pause(self):
        self.running = False
        self.play_btn.setText("Play")
        self.timer.stop()

    def do_reset(self):
        self.current_frame = self.env.reset()
        self.frame_display.update_frame(self.current_frame)
        self.frame_updated.emit(self.current_frame)
        self.total_reward = 0.0
        self.step_count = 0
        self._update_info()

    def _tick(self):
        policy = self.policy_combo.currentText()
        if policy == "Random":
            action = np.random.randint(0, self.env.action_space_size)
        elif policy == "No-op":
            action = 0
        else:
            action = self._pending_action if self._pending_action is not None else 0
            self._pending_action = None

        prev_frame = self.current_frame
        self.current_frame, reward, done, info = self.env.step(action)
        self.total_reward += reward
        self.step_count += 1

        self.frame_display.update_frame(self.current_frame)
        self.frame_updated.emit(self.current_frame)
        self.action_taken.emit(action, prev_frame, self.current_frame)
        self._update_info(reward)

        if done:
            self.do_reset()

    def _update_info(self, last_reward: float = 0.0):
        r_str = f"+{last_reward:.0f}" if last_reward > 0 else f"{last_reward:.0f}" if last_reward < 0 else ""
        self.info_label.setText(
            f"Step: {self.step_count} | Total Reward: {self.total_reward:.0f} {r_str}"
        )

    def _update_timer_interval(self, value: int):
        self.speed_label.setText(f"{value}ms")
        if self.running:
            self.timer.setInterval(value)

    def keyPressEvent(self, event):
        key = event.key()
        action_map = {
            Qt.Key.Key_W: 1, Qt.Key.Key_Up: 1,
            Qt.Key.Key_S: 2, Qt.Key.Key_Down: 2,
            Qt.Key.Key_A: 3, Qt.Key.Key_Left: 3,
            Qt.Key.Key_D: 4, Qt.Key.Key_Right: 4,
        }
        if key in action_map:
            self._pending_action = action_map[key]
        else:
            super().keyPressEvent(event)
