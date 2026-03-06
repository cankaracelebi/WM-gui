import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class PlotWidget(FigureCanvasQTAgg):
    """Matplotlib figure embedded in a Qt widget with dark theme."""

    def __init__(self, title: str = "", width: int = 5, height: int = 3, parent=None):
        self.fig = Figure(figsize=(width, height), dpi=80)
        self.fig.patch.set_facecolor("#16213e")
        super().__init__(self.fig)

        self.ax = self.fig.add_subplot(111)
        self._style_axis(self.ax, title)
        self.fig.tight_layout(pad=1.5)

    def _style_axis(self, ax, title: str):
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, color="#e94560", fontsize=11, fontweight="bold")
        ax.tick_params(colors="#a0a0c0", labelsize=8)
        ax.spines["bottom"].set_color("#0f3460")
        ax.spines["left"].set_color("#0f3460")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.label.set_color("#a0a0c0")
        ax.yaxis.label.set_color("#a0a0c0")

    def update_plot(self, data_dict: dict[str, list[float]], xlabel: str = "Step",
                    ylabel: str = "Loss", max_points: int = 500):
        """Update plot with multiple named series. data_dict: {name: [values]}."""
        self.ax.clear()
        self._style_axis(self.ax, self.ax.get_title())

        colors = ["#e94560", "#00d4ff", "#50fa7b", "#ffb86c", "#bd93f9"]
        for i, (name, values) in enumerate(data_dict.items()):
            if not values:
                continue
            # Downsample for performance
            if len(values) > max_points:
                step = len(values) // max_points
                values = values[::step]
            color = colors[i % len(colors)]
            self.ax.plot(values, color=color, linewidth=1.5, label=name, alpha=0.9)

        if data_dict:
            self.ax.legend(fontsize=8, facecolor="#16213e", edgecolor="#0f3460",
                          labelcolor="#e0e0e0", loc="upper right")
        self.ax.set_xlabel(xlabel, fontsize=9)
        self.ax.set_ylabel(ylabel, fontsize=9)
        self.fig.tight_layout(pad=1.5)
        self.draw_idle()
