#!/usr/bin/env python3
"""
World Model Visualization Framework
====================================

An interactive GUI for experimenting with world models.
Train a VAE on visual game environments, watch reconstructions,
explore latent spaces, and let the model dream.

Usage:
    python main.py
"""

import sys
import os
import ctypes.util

# Ensure pygame doesn't open its own window
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def _check_qt_deps():
    """Check for libxcb-cursor0 and suggest install if missing."""
    if os.environ.get("QT_QPA_PLATFORM") in ("offscreen", "minimal"):
        return
    if sys.platform == "linux":
        if ctypes.util.find_library("xcb-cursor") is None:
            print(
                "WARNING: libxcb-cursor0 not found. The GUI may fail to start.\n"
                "Install it with: sudo apt install libxcb-cursor0\n"
                "Or run headless: QT_QPA_PLATFORM=offscreen python main.py\n"
            )


from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from utils.config import AppConfig
from gui.app import DARK_STYLESHEET
from gui.main_window import MainWindow


def main():
    _check_qt_deps()

    # High DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("World Model Visualizer")
    app.setStyleSheet(DARK_STYLESHEET)

    config = AppConfig()
    window = MainWindow(config)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
