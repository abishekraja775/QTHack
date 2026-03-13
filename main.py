"""
main.py  —  VNA Simulator v2 entry point

Install dependencies:
    pip install PyQt6 numpy matplotlib scikit-rf

Run:
    python main.py
"""

import sys
from PyQt6.QtWidgets import QApplication
from gui import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("VNA Simulator v2")
    window = MainWindow()
    window.show()
    window.raise_()
    window.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
