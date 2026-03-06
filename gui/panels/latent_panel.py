import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QComboBox, QSlider, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from sklearn.decomposition import PCA

from gui.widgets.frame_display import FrameDisplay
from gui.widgets.latent_scatter import LatentScatter


class LatentPanel(QWidget):
    """Interactive latent space explorer with scatter plot and dimension sliders."""

    decode_requested = Signal(object)  # numpy latent vector to decode

    def __init__(self, latent_dim: int = 32, parent=None):
        super().__init__(parent)
        self.latent_dim = latent_dim
        self.latent_cache: list[np.ndarray] = []
        self.frame_cache: list[np.ndarray] = []
        self.pca: PCA | None = None
        self.projected: np.ndarray | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Title
        title = QLabel("Latent Space Explorer")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #e94560;")
        layout.addWidget(title)

        # Scatter plot
        self.scatter = LatentScatter(width=5, height=4)
        self.scatter.point_clicked.connect(self._on_point_clicked)
        self.scatter.empty_clicked.connect(self._on_empty_clicked)
        layout.addWidget(self.scatter)

        # Preview frame
        preview_row = QHBoxLayout()
        self.preview = FrameDisplay(128)
        preview_row.addWidget(self.preview)

        # Controls column
        ctrl_col = QVBoxLayout()

        # Projection method
        proj_row = QHBoxLayout()
        proj_row.addWidget(QLabel("Projection:"))
        self.proj_combo = QComboBox()
        self.proj_combo.addItems(["PCA", "t-SNE"])
        self.proj_combo.currentTextChanged.connect(self._reproject)
        proj_row.addWidget(self.proj_combo)
        ctrl_col.addLayout(proj_row)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._reproject)
        ctrl_col.addWidget(self.refresh_btn)

        # Latent walk
        self.walk_btn = QPushButton("Latent Walk")
        self.walk_btn.setToolTip("Shift-click two points to interpolate between them")
        self.walk_btn.setEnabled(False)
        self.walk_btn.clicked.connect(self._do_latent_walk)
        ctrl_col.addWidget(self.walk_btn)

        self.info_label = QLabel(f"Points: 0 | Dim: {latent_dim}")
        self.info_label.setStyleSheet("color: #a0a0c0; font-size: 11px;")
        ctrl_col.addWidget(self.info_label)
        ctrl_col.addStretch()

        preview_row.addLayout(ctrl_col)
        layout.addLayout(preview_row)

        # Dimension sliders
        slider_group = QGroupBox("Latent Dimensions")
        slider_scroll = QScrollArea()
        slider_scroll.setWidgetResizable(True)
        slider_scroll.setMaximumHeight(150)
        slider_scroll.setStyleSheet("background: #16213e;")

        slider_container = QWidget()
        slider_layout = QVBoxLayout(slider_container)
        slider_layout.setSpacing(2)

        self.dim_sliders: list[QSlider] = []
        self.dim_labels: list[QLabel] = []
        for i in range(latent_dim):
            row = QHBoxLayout()
            label = QLabel(f"z{i}:")
            label.setFixedWidth(30)
            label.setStyleSheet("color: #a0a0c0; font-size: 9px;")
            row.addWidget(label)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(-300, 300)  # -3.0 to 3.0
            slider.setValue(0)
            slider.valueChanged.connect(self._on_slider_changed)
            row.addWidget(slider)

            val_label = QLabel("0.00")
            val_label.setFixedWidth(40)
            val_label.setStyleSheet("color: #a0a0c0; font-size: 9px;")
            row.addWidget(val_label)

            slider_layout.addLayout(row)
            self.dim_sliders.append(slider)
            self.dim_labels.append(val_label)

        slider_scroll.setWidget(slider_container)
        sg_layout = QVBoxLayout(slider_group)
        sg_layout.addWidget(slider_scroll)
        layout.addWidget(slider_group)

        layout.addStretch()

        # Walk state
        self._walk_points: list[int] = []

    def add_latent(self, z: np.ndarray, frame: np.ndarray | None = None):
        """Add a latent vector (and optionally its frame) to the cache."""
        self.latent_cache.append(z.copy())
        if frame is not None:
            self.frame_cache.append(frame.copy())
        else:
            self.frame_cache.append(None)

    def _reproject(self):
        if len(self.latent_cache) < 3:
            self.info_label.setText(f"Points: {len(self.latent_cache)} (need 3+)")
            return

        method = self.proj_combo.currentText()
        latents = np.array(self.latent_cache)

        if method == "PCA":
            self.pca = PCA(n_components=2)
            self.projected = self.pca.fit_transform(latents)
        else:
            # t-SNE (can be slow for large N)
            from sklearn.manifold import TSNE
            perp = min(30, len(latents) - 1)
            tsne = TSNE(n_components=2, perplexity=max(5, perp), random_state=42)
            self.projected = tsne.fit_transform(latents)

        colors = np.arange(len(latents), dtype=float)
        self.scatter.update_scatter(self.projected, colors)
        self.info_label.setText(f"Points: {len(latents)} | Method: {method}")

    def _on_point_clicked(self, idx: int):
        if idx < len(self.frame_cache) and self.frame_cache[idx] is not None:
            self.preview.update_frame(self.frame_cache[idx])
        elif idx < len(self.latent_cache):
            self.decode_requested.emit(self.latent_cache[idx])

        # Set sliders to this latent
        z = self.latent_cache[idx]
        self._set_sliders(z)

        # Walk selection
        self._walk_points.append(idx)
        if len(self._walk_points) >= 2:
            self.walk_btn.setEnabled(True)
        if len(self._walk_points) > 2:
            self._walk_points = self._walk_points[-2:]

    def _on_empty_clicked(self, x: float, y: float):
        if self.pca is not None and self.proj_combo.currentText() == "PCA":
            z = self.pca.inverse_transform(np.array([[x, y]]))[0]
            self._set_sliders(z)
            self.decode_requested.emit(z)

    def _set_sliders(self, z: np.ndarray):
        for i, val in enumerate(z):
            if i < len(self.dim_sliders):
                self.dim_sliders[i].blockSignals(True)
                self.dim_sliders[i].setValue(int(val * 100))
                self.dim_sliders[i].blockSignals(False)
                self.dim_labels[i].setText(f"{val:.2f}")

    def _on_slider_changed(self):
        z = np.array([s.value() / 100.0 for s in self.dim_sliders], dtype=np.float32)
        for i, label in enumerate(self.dim_labels):
            label.setText(f"{z[i]:.2f}")
        self.decode_requested.emit(z)

    def _do_latent_walk(self):
        if len(self._walk_points) < 2:
            return
        idx1, idx2 = self._walk_points[-2], self._walk_points[-1]
        z1 = self.latent_cache[idx1]
        z2 = self.latent_cache[idx2]

        # Generate interpolation path
        steps = 30
        path = []
        for t in np.linspace(0, 1, steps):
            z_interp = z1 * (1 - t) + z2 * t
            path.append(z_interp)
            self.decode_requested.emit(z_interp)

        # Draw path on scatter
        if self.projected is not None:
            if self.pca is not None and self.proj_combo.currentText() == "PCA":
                path_2d = self.pca.transform(np.array(path))
                self.scatter.draw_path(path_2d)

    def update_preview(self, frame: np.ndarray):
        """Called when a decoded frame is ready."""
        self.preview.update_frame(frame)
