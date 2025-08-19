#!/usr/bin/env python3
"""
MineMapper - Advanced Minesweeper Solver

Main application entry point. This launches the PyQt6 GUI application
for the sophisticated minesweeper screenshot solver.

Features:
- Computer vision board detection and cell recognition
- Multi-strategy solving algorithms (CSP, probability, pattern recognition)
- Real-time analysis with progress tracking
- Professional UI with board visualization
- Clipboard screenshot support

Usage:
    python main.py

Requirements:
    - Python 3.8+
    - PyQt6
    - OpenCV
    - NumPy
    - SciPy
"""

import sys
import logging
from pathlib import Path
import traceback

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QIcon

    from src.ui.main_window import MainWindow

except ImportError as e:
    print(f"Import error: {e}")
    print("\nPlease install required dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def setup_logging():
    """Setup application logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "minemapper.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger = logging.getLogger(__name__)
    logger.critical(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )

    # Show error dialog if GUI is available
    try:
        app = QApplication.instance()
        if app:
            error_msg = f"An unexpected error occurred:\n\n{exc_type.__name__}: {exc_value}"
            QMessageBox.critical(None, "Critical Error", error_msg)
    except:
        pass


def main():
    """Main application entry point"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Install global exception handler
    sys.excepthook = handle_exception

    logger.info("Starting MineMapper application...")

    try:
        # Create QApplication
        app = QApplication(sys.argv)
        app.setApplicationName("MineMapper")
        app.setApplicationVersion("1.0")
        app.setOrganizationName("MineMapper")

        # High DPI handling is automatic in PyQt6; avoid deprecated AA_* flags
        # If needed, use QGuiApplication high DPI policies instead

        # Create and show main window
        window = MainWindow()
        window.show()

        logger.info("Application started successfully")

        # Run application
        exit_code = app.exec()

        logger.info(f"Application exited with code {exit_code}")
        return exit_code

    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
        logger.critical(traceback.format_exc())

        try:
            QMessageBox.critical(
                None,
                "Startup Error",
                f"Failed to start MineMapper:\n\n{e}\n\nCheck logs for details."
            )
        except:
            print(f"Critical error: {e}")

        return 1


if __name__ == "__main__":
    sys.exit(main())
