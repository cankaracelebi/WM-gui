import numpy as np
import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Signal


class LatentScatter(FigureCanvasQTAgg):
    """Interactive 2D scatter plot for latent space visualization."""

    point_clicked = Signal(int)  # index of clicked point
    empty_clicked = Signal(float, float)  # (x, y) in data coords

    def __init__(self, width: int = 5, height: int = 4, parent=None):
        self.fig = Figure(figsize=(width, height), dpi=80)
        self.fig.patch.set_facecolor("#16213e")
        super().__init__(self.fig)

        self.ax = self.fig.add_subplot(111)
        self._style_axis()

        self.projected: np.ndarray | None = None
        self.colors: np.ndarray | None = None
        self._scatter = None

        self.mpl_connect("button_press_event", self._on_click)

    def _style_axis(self):
        self.ax.set_facecolor("#1a1a2e")
        self.ax.set_title("Latent Space", color="#e94560", fontsize=11, fontweight="bold")
        self.ax.tick_params(colors="#a0a0c0", labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color("#0f3460")

    def update_scatter(self, points_2d: np.ndarray, colors: np.ndarray | None = None,
                       labels: list[str] | None = None):
        """Update scatter with new projected points. points_2d: (N, 2)."""
        self.projected = points_2d
        self.ax.clear()
        self._style_axis()

        if points_2d is None or len(points_2d) == 0:
            self.ax.text(0.5, 0.5, "No data yet", transform=self.ax.transAxes,
                        ha="center", va="center", color="#606080", fontsize=12)
            self.draw_idle()
            return

        if colors is None:
            colors = np.arange(len(points_2d), dtype=float)
        self.colors = colors

        self._scatter = self.ax.scatter(
            points_2d[:, 0], points_2d[:, 1],
            c=colors, cmap="plasma", s=12, alpha=0.7, edgecolors="none"
        )
        self.ax.set_xlabel("Component 1", fontsize=9, color="#a0a0c0")
        self.ax.set_ylabel("Component 2", fontsize=9, color="#a0a0c0")
        self.fig.tight_layout(pad=1.5)
        self.draw_idle()

    def draw_path(self, points_2d: np.ndarray):
        """Draw an interpolation path on top of the scatter."""
        if points_2d is not None and len(points_2d) > 1:
            self.ax.plot(points_2d[:, 0], points_2d[:, 1],
                        color="#e94560", linewidth=2, alpha=0.8, linestyle="--")
            self.draw_idle()

    def _on_click(self, event):
        if event.inaxes != self.ax or self.projected is None:
            return

        distances = np.sqrt(
            (self.projected[:, 0] - event.xdata) ** 2 +
            (self.projected[:, 1] - event.ydata) ** 2
        )
        nearest_idx = int(np.argmin(distances))

        # Threshold: if click is near a point, select it; otherwise emit empty click
        data_range = max(
            self.projected[:, 0].ptp(), self.projected[:, 1].ptp(), 1e-6
        )
        threshold = data_range * 0.03

        if distances[nearest_idx] < threshold:
            self.point_clicked.emit(nearest_idx)
        else:
            self.empty_clicked.emit(event.xdata, event.ydata)
