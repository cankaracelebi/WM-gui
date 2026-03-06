DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: "Segoe UI", "Ubuntu", sans-serif;
    font-size: 12px;
}

QTabWidget::pane {
    border: 1px solid #0f3460;
    background: #16213e;
    border-radius: 4px;
}

QTabBar::tab {
    background: #0f3460;
    color: #a0a0c0;
    padding: 8px 20px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}

QTabBar::tab:selected {
    background: #16213e;
    color: #e94560;
    font-weight: bold;
}

QTabBar::tab:hover {
    background: #1a1a5e;
    color: #ffffff;
}

QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #1a1a5e, stop:1 #0f3460);
    color: #e0e0e0;
    border: 1px solid #0f3460;
    border-radius: 6px;
    padding: 6px 16px;
    font-weight: bold;
}

QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #2a2a7e, stop:1 #1a4a80);
    border-color: #e94560;
}

QPushButton:pressed {
    background: #e94560;
    color: #ffffff;
}

QPushButton:disabled {
    background: #252545;
    color: #606080;
    border-color: #303050;
}

QSlider::groove:horizontal {
    border: 1px solid #0f3460;
    height: 6px;
    background: #16213e;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #e94560;
    border: 1px solid #c0355e;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #e94560, stop:1 #0f3460);
    border-radius: 3px;
}

QComboBox {
    background: #16213e;
    color: #e0e0e0;
    border: 1px solid #0f3460;
    border-radius: 4px;
    padding: 4px 8px;
}

QComboBox::drop-down {
    border-left: 1px solid #0f3460;
    width: 20px;
}

QComboBox QAbstractItemView {
    background: #16213e;
    color: #e0e0e0;
    selection-background-color: #e94560;
}

QLabel {
    color: #c0c0e0;
}

QGroupBox {
    border: 1px solid #0f3460;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 12px;
    font-weight: bold;
    color: #e94560;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}

QProgressBar {
    border: 1px solid #0f3460;
    border-radius: 4px;
    background: #16213e;
    text-align: center;
    color: #e0e0e0;
    height: 16px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #e94560, stop:1 #0f3460);
    border-radius: 3px;
}

QSpinBox, QDoubleSpinBox {
    background: #16213e;
    color: #e0e0e0;
    border: 1px solid #0f3460;
    border-radius: 4px;
    padding: 2px 6px;
}

QSplitter::handle {
    background: #0f3460;
    width: 2px;
    height: 2px;
}

QStatusBar {
    background: #0f0f1e;
    color: #a0a0c0;
    border-top: 1px solid #0f3460;
}

QScrollBar:vertical {
    background: #1a1a2e;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #0f3460;
    border-radius: 4px;
    min-height: 20px;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
"""
