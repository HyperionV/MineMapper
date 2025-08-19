"""
Main Application Window - PyQt6 User Interface

This module implements the main application interface with:
1. Board display and visualization
2. Screenshot input and clipboard handling
3. Real-time analysis logging
4. Control panels and settings
"""

import sys
from typing import Optional, Dict, Any, List, Tuple
import logging
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTextEdit, QSpinBox, QGroupBox,
    QSplitter, QFrame, QScrollArea, QProgressBar, QStatusBar,
    QMenuBar, QMenu, QMessageBox, QFileDialog, QComboBox, QPlainTextEdit
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize, QRect, pyqtSlot
)
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QFont, QPen, QBrush,
    QClipboard, QAction, QKeySequence
)
import numpy as np
import cv2

from ..board_loader.board_detector import BoardDetector
from ..board_loader.html_parser import HTMLBoardParser
from ..solver.game_state import MinesweeperBoard, CellState, CellContent
from ..solver.master_solver import MasterSolver, Move, SolverResult


class BoardWidget(QWidget):
    """Custom widget for displaying the minesweeper board"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.board: Optional[MinesweeperBoard] = None
        self.cell_size = 30
        self.grid_color = QColor(128, 128, 128)
        self.setMinimumSize(400, 400)

        # Colors for different cell types
        self.colors = {
            'unrevealed': QColor(192, 192, 192),
            'revealed': QColor(255, 255, 255),
            'flagged': QColor(255, 100, 100),
            'safe_suggestion': QColor(100, 255, 100),
            'probability': QColor(255, 255, 100),
            'mine_suggestion': QColor(255, 50, 50)
        }

        # Text colors for numbers
        self.number_colors = {
            1: QColor(0, 0, 255),
            2: QColor(0, 128, 0),
            3: QColor(255, 0, 0),
            4: QColor(128, 0, 128),
            5: QColor(128, 0, 0),
            6: QColor(0, 128, 128),
            7: QColor(0, 0, 0),
            8: QColor(128, 128, 128)
        }

        # Suggested moves
        self.suggested_moves: List[Move] = []

    def set_board(self, board: MinesweeperBoard) -> None:
        """Set the board to display"""
        self.board = board
        if board:
            # Adjust widget size based on board dimensions
            width = board.cols * self.cell_size + 1
            height = board.rows * self.cell_size + 1
            self.setMinimumSize(width, height)
        self.update()

    def set_suggested_moves(self, moves: List[Move]) -> None:
        """Set suggested moves to highlight"""
        self.suggested_moves = moves
        self.update()

    def paintEvent(self, event):
        """Custom paint event to draw the board"""
        if not self.board:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate actual cell size based on widget size
        widget_width = self.width()
        widget_height = self.height()
        cell_width = widget_width // self.board.cols
        cell_height = widget_height // self.board.rows
        actual_cell_size = min(cell_width, cell_height)

        # Draw grid and cells
        for row in range(self.board.rows):
            for col in range(self.board.cols):
                cell = self.board.get_cell(row, col)
                if not cell:
                    continue

                # Calculate cell rectangle
                x = col * actual_cell_size
                y = row * actual_cell_size
                rect = QRect(x, y, actual_cell_size, actual_cell_size)

                # Determine cell color
                color = self._get_cell_color(cell, row, col)
                painter.fillRect(rect, color)

                # Draw cell content
                self._draw_cell_content(painter, rect, cell)

                # Draw grid lines
                painter.setPen(QPen(self.grid_color, 1))
                painter.drawRect(rect)

    def _get_cell_color(self, cell, row: int, col: int) -> QColor:
        """Determine the color for a cell"""
        # Check for suggested moves first
        for move in self.suggested_moves:
            if move.row == row and move.col == col:
                if move.action == 'reveal':
                    return self.colors['safe_suggestion']
                elif move.action == 'flag':
                    return self.colors['mine_suggestion']

        # Check probability coloring
        if cell.probability > 0.1:
            # Blend probability color with base color
            alpha = min(cell.probability, 0.8)
            prob_color = self.colors['probability']
            base_color = self.colors['unrevealed'] if cell.is_unrevealed else self.colors['revealed']

            # Simple color blending
            r = int(base_color.red() * (1 - alpha) + prob_color.red() * alpha)
            g = int(base_color.green() * (1 - alpha) +
                    prob_color.green() * alpha)
            b = int(base_color.blue() * (1 - alpha) +
                    prob_color.blue() * alpha)
            return QColor(r, g, b)

        # Standard cell colors
        if cell.is_flagged:
            return self.colors['flagged']
        elif cell.is_revealed:
            return self.colors['revealed']
        else:
            return self.colors['unrevealed']

    def _draw_cell_content(self, painter: QPainter, rect: QRect, cell) -> None:
        """Draw the content of a cell (numbers, flags, etc.)"""
        if cell.is_flagged:
            # Draw flag symbol
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            font = QFont("Arial", max(8, rect.height() // 3))
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "F")

        elif cell.is_revealed and cell.is_numbered:
            # Draw number
            number = cell.number
            color = self.number_colors.get(number, QColor(0, 0, 0))
            painter.setPen(QPen(color, 2))
            font = QFont("Arial", max(8, rect.height() // 2),
                         QFont.Weight.Bold)
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(number))

        elif cell.is_revealed and cell.content == CellContent.EMPTY:
            # Draw empty cell (already colored, no additional content needed)
            pass

        # Draw probability text for unrevealed cells
        elif cell.is_unrevealed and cell.probability > 0.05:
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            font = QFont("Arial", max(6, rect.height() // 4))
            painter.setFont(font)
            prob_text = f"{cell.probability:.0%}"
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, prob_text)


class AnalysisWorker(QThread):
    """Worker thread for performing board analysis"""

    progress = pyqtSignal(str, float)
    finished = pyqtSignal(object)  # SolverResult
    error = pyqtSignal(str)

    def __init__(self, solver: MasterSolver):
        super().__init__()
        self.solver = solver

    def run(self):
        """Run the analysis in background thread"""
        try:
            # Add progress callback
            self.solver.add_progress_callback(self._progress_callback)

            # Perform analysis
            result = self.solver.analyze()
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))

    def _progress_callback(self, message: str, progress: float):
        """Progress callback for solver"""
        self.progress.emit(message, progress)


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MineMapper - Advanced Minesweeper Solver")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize components
        self.board: Optional[MinesweeperBoard] = None
        self.board_detector: Optional[BoardDetector] = None
        self.html_parser: Optional[HTMLBoardParser] = None
        self.solver: Optional[MasterSolver] = None
        self.analysis_worker: Optional[AnalysisWorker] = None

        # Setup logging
        self.setup_logging()

        # Create UI
        self.setup_ui()

        # Initialize computer vision
        self.init_computer_vision()

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def setup_ui(self):
        """Setup the user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - Controls and settings
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)

        # Center panel - Board display
        center_panel = self.create_board_panel()
        splitter.addWidget(center_panel)

        # Right panel - Logs and analysis
        right_panel = self.create_analysis_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([300, 600, 300])

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Paste screenshot or load image")

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Menu bar
        self.create_menu_bar()

    def create_control_panel(self) -> QWidget:
        """Create the control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Board configuration group
        config_group = QGroupBox("Board Configuration")
        config_layout = QVBoxLayout(config_group)

        # Board size inputs
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Rows:"))
        self.rows_spinbox = QSpinBox()
        self.rows_spinbox.setRange(5, 50)
        self.rows_spinbox.setValue(16)
        size_layout.addWidget(self.rows_spinbox)

        size_layout.addWidget(QLabel("Cols:"))
        self.cols_spinbox = QSpinBox()
        self.cols_spinbox.setRange(5, 50)
        self.cols_spinbox.setValue(16)
        size_layout.addWidget(self.cols_spinbox)

        config_layout.addLayout(size_layout)

        # Mine count input
        mines_layout = QHBoxLayout()
        mines_layout.addWidget(QLabel("Mines:"))
        self.mines_spinbox = QSpinBox()
        self.mines_spinbox.setRange(1, 500)
        self.mines_spinbox.setValue(40)
        mines_layout.addWidget(self.mines_spinbox)

        config_layout.addLayout(mines_layout)

        layout.addWidget(config_group)

        # Input mode selection
        input_mode_group = QGroupBox("Input Mode")
        input_mode_layout = QVBoxLayout(input_mode_group)

        self.input_mode_combo = QComboBox()
        self.input_mode_combo.addItems(["Screenshot", "HTML Structure"])
        self.input_mode_combo.currentTextChanged.connect(
            self.on_input_mode_changed)
        input_mode_layout.addWidget(self.input_mode_combo)

        layout.addWidget(input_mode_group)

        # Screenshot input group
        self.screenshot_group = QGroupBox("Screenshot Input")
        screenshot_layout = QVBoxLayout(self.screenshot_group)

        self.paste_button = QPushButton("Paste from Clipboard")
        self.paste_button.clicked.connect(self.paste_screenshot)
        screenshot_layout.addWidget(self.paste_button)

        self.load_button = QPushButton("Load Image File")
        self.load_button.clicked.connect(self.load_image_file)
        screenshot_layout.addWidget(self.load_button)

        layout.addWidget(self.screenshot_group)

        # HTML input group (initially hidden)
        self.html_group = QGroupBox("HTML Structure Input")
        html_layout = QVBoxLayout(self.html_group)

        # Instructions
        html_instructions = QLabel(
            "Paste the HTML structure containing the minesweeper board:")
        html_instructions.setWordWrap(True)
        html_layout.addWidget(html_instructions)

        # HTML text area
        self.html_text_area = QPlainTextEdit()
        self.html_text_area.setPlaceholderText(
            'Paste HTML structure here...\nExample: <div id="AreaBlock" class="pull-left">...</div>')
        self.html_text_area.setMaximumHeight(150)
        html_layout.addWidget(self.html_text_area)

        # Extract button
        self.extract_button = QPushButton("Extract Board")
        self.extract_button.clicked.connect(self.extract_html_board)
        html_layout.addWidget(self.extract_button)

        self.html_group.hide()  # Initially hidden
        layout.addWidget(self.html_group)

        # Analysis group
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout(analysis_group)

        self.analyze_button = QPushButton("Analyze Board")
        self.analyze_button.clicked.connect(self.analyze_board)
        self.analyze_button.setEnabled(False)
        analysis_layout.addWidget(self.analyze_button)

        # Strategy selection
        strategy_layout = QHBoxLayout()
        strategy_layout.addWidget(QLabel("Strategy:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(
            ["Hybrid", "Pattern", "CSP", "Probability"])
        strategy_layout.addWidget(self.strategy_combo)
        analysis_layout.addLayout(strategy_layout)

        layout.addWidget(analysis_group)

        # Statistics group
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)

        self.stats_label = QLabel("No board loaded")
        stats_layout.addWidget(self.stats_label)

        layout.addWidget(stats_group)

        # Add stretch to push everything to top
        layout.addStretch()

        return panel

    def create_board_panel(self) -> QWidget:
        """Create the board display panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Board title
        title = QLabel("Minesweeper Board")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        # Scroll area for board
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # Board widget
        self.board_widget = BoardWidget()
        scroll_area.setWidget(self.board_widget)

        layout.addWidget(scroll_area)

        return panel

    def create_analysis_panel(self) -> QWidget:
        """Create the analysis and logging panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Logs group
        logs_group = QGroupBox("Analysis Logs")
        logs_layout = QVBoxLayout(logs_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(300)
        logs_layout.addWidget(self.log_text)

        layout.addWidget(logs_group)

        # Results group
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)

        layout.addWidget(results_group)

        return panel

    def create_menu_bar(self):
        """Create the menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        load_action = QAction('Load Image...', self)
        load_action.setShortcut(QKeySequence.StandardKey.Open)
        load_action.triggered.connect(self.load_image_file)
        file_menu.addAction(load_action)

        paste_action = QAction('Paste Screenshot', self)
        paste_action.setShortcut(QKeySequence('Ctrl+V'))
        paste_action.triggered.connect(self.paste_screenshot)
        file_menu.addAction(paste_action)

        file_menu.addSeparator()

        exit_action = QAction('Exit', self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Analysis menu
        analysis_menu = menubar.addMenu('Analysis')

        analyze_action = QAction('Analyze Board', self)
        analyze_action.setShortcut(QKeySequence('F5'))
        analyze_action.triggered.connect(self.analyze_board)
        analysis_menu.addAction(analyze_action)

    def init_computer_vision(self):
        """Initialize computer vision components and HTML parser"""
        try:
            templates_dir = Path(__file__).parent.parent.parent / "templates"
            self.board_detector = BoardDetector(templates_dir)
            self.html_parser = HTMLBoardParser()
            self.log_message(
                "Computer vision and HTML parser initialized successfully")
        except Exception as e:
            self.log_error(f"Failed to initialize components: {e}")

    def paste_screenshot(self):
        """Paste screenshot from clipboard"""
        clipboard = QApplication.clipboard()
        pixmap = clipboard.pixmap()

        if pixmap.isNull():
            QMessageBox.warning(self, "Warning", "No image found in clipboard")
            return

        # Convert QPixmap to numpy array
        image = self.qpixmap_to_numpy(pixmap)
        self.process_image(image)

    def load_image_file(self):
        """Load image from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Screenshot", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )

        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                self.process_image(image)
            else:
                QMessageBox.critical(
                    self, "Error", "Failed to load image file")

    def on_input_mode_changed(self, mode: str):
        """Handle input mode change"""
        if mode == "Screenshot":
            self.screenshot_group.show()
            self.html_group.hide()
        elif mode == "HTML Structure":
            self.screenshot_group.hide()
            self.html_group.show()

    def extract_html_board(self):
        """Extract board from HTML structure"""
        try:
            html_content = self.html_text_area.toPlainText().strip()
            if not html_content:
                QMessageBox.warning(
                    self, "Warning", "Please paste HTML structure first")
                return

            self.log_message("Extracting board from HTML structure...")

            # Parse HTML and create board
            expected_mines = self.mines_spinbox.value()
            board = self.html_parser.parse_board_html(
                html_content, expected_mines)
            if board is None:
                raise ValueError("Failed to parse HTML structure")

            self.board = board
            self.solver = MasterSolver(board)

            # Update UI
            self.board_widget.set_board(board)
            self.update_statistics()
            self.analyze_button.setEnabled(True)

            self.log_message(
                f"Board extracted successfully: {board.cols}x{board.rows} with {board.total_mines} mines")

        except Exception as e:
            self.log_error(f"Error extracting HTML board: {e}")
            QMessageBox.warning(self, "Error", f"Failed to extract board: {e}")

    def process_image(self, image: np.ndarray):
        """Process the input image and extract board"""
        if self.board_detector is None:
            QMessageBox.critical(
                self, "Error", "Computer vision not initialized")
            return

        self.log_message("Processing screenshot...")

        try:
            # Get board configuration
            rows = self.rows_spinbox.value()
            cols = self.cols_spinbox.value()
            mines = self.mines_spinbox.value()

            # Analyze the image
            results = self.board_detector.analyze_board(image, (rows, cols))

            if not results['success']:
                QMessageBox.warning(
                    self, "Warning", "Failed to detect board in image")
                return

            # Create board from analysis
            self.board = MinesweeperBoard(rows, cols, mines)

            # Update board with detected cells
            classifications = results['cell_classifications']
            confidence_map = results['confidence_map']

            for row in range(rows):
                for col in range(cols):
                    cell_type = classifications[row][col]
                    confidence = confidence_map[row][col]
                    self.board.update_cell_from_vision(
                        row, col, cell_type, confidence)

            # Update frontier
            self.board.update_frontier()

            # Update UI
            self.board_widget.set_board(self.board)
            self.update_statistics()
            self.analyze_button.setEnabled(True)

            self.log_message(
                f"Board extracted successfully: {rows}x{cols} with {mines} mines")

        except Exception as e:
            self.log_error(f"Failed to process image: {e}")
            QMessageBox.critical(
                self, "Error", f"Failed to process image: {e}")

    def analyze_board(self):
        """Analyze the current board"""
        if not self.board:
            QMessageBox.warning(self, "Warning", "No board loaded")
            return

        # Disable analyze button during analysis
        self.analyze_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Clear previous results
        self.results_text.clear()

        # Create solver
        self.solver = MasterSolver(self.board)

        # Start analysis in worker thread
        self.analysis_worker = AnalysisWorker(self.solver)
        self.analysis_worker.progress.connect(self.update_progress)
        self.analysis_worker.finished.connect(self.analysis_finished)
        self.analysis_worker.error.connect(self.analysis_error)
        self.analysis_worker.start()

        self.log_message("Starting board analysis...")

    @pyqtSlot(str, float)
    def update_progress(self, message: str, progress: float):
        """Update analysis progress"""
        self.progress_bar.setValue(int(progress * 100))
        self.status_bar.showMessage(message)
        self.log_message(message)

    @pyqtSlot(object)
    def analysis_finished(self, result: SolverResult):
        """Handle analysis completion"""
        self.progress_bar.setVisible(False)
        self.analyze_button.setEnabled(True)

        # Display results
        self.display_analysis_results(result)

        # Update board visualization with suggestions
        self.board_widget.set_suggested_moves(result.moves)

        self.status_bar.showMessage("Analysis complete")
        self.log_message(
            f"Analysis completed in {result.analysis_time:.2f} seconds")

    @pyqtSlot(str)
    def analysis_error(self, error_message: str):
        """Handle analysis error"""
        self.progress_bar.setVisible(False)
        self.analyze_button.setEnabled(True)

        self.log_error(f"Analysis failed: {error_message}")
        QMessageBox.critical(self, "Analysis Error",
                             f"Analysis failed: {error_message}")

        self.status_bar.showMessage("Analysis failed")

    def display_analysis_results(self, result: SolverResult):
        """Display analysis results in the UI"""
        results_text = []

        # Summary
        results_text.append("=== ANALYSIS RESULTS ===\n")
        results_text.append(
            f"Analysis time: {result.analysis_time:.2f} seconds")
        results_text.append(
            f"Strategies used: {', '.join(result.strategies_used)}")
        results_text.append(f"Moves found: {len(result.moves)}\n")

        # Suggested moves
        if result.moves:
            results_text.append("=== SUGGESTED MOVES ===")
            for i, move in enumerate(result.moves[:5], 1):  # Show top 5 moves
                action_text = "REVEAL" if move.action == 'reveal' else "FLAG"
                results_text.append(
                    f"{i}. {action_text} ({move.row}, {move.col}) - "
                    f"Confidence: {move.confidence:.3f} - {move.reasoning}"
                )
            results_text.append("")

        # Statistics
        if result.statistics:
            results_text.append("=== STATISTICS ===")
            board_stats = result.statistics.get('board_statistics', {})
            for key, value in board_stats.items():
                results_text.append(
                    f"{key.replace('_', ' ').title()}: {value}")

            pattern_stats = result.statistics.get('pattern_statistics', {})
            if pattern_stats:
                results_text.append("\nPattern Recognition:")
                for key, value in pattern_stats.items():
                    results_text.append(
                        f"  {key.replace('_', ' ').title()}: {value}")

        self.results_text.setPlainText('\n'.join(results_text))

    def update_statistics(self):
        """Update the statistics display"""
        if not self.board:
            self.stats_label.setText("No board loaded")
            return

        stats = self.board.get_game_statistics()
        stats_text = [
            f"Size: {self.board.rows}×{self.board.cols}",
            f"Total mines: {stats['total_mines']}",
            f"Remaining: {stats['remaining_mines']}",
            f"Revealed: {stats['revealed_cells']}",
            f"Flagged: {stats['flagged_cells']}",
            f"Progress: {stats['completion_percentage']:.1f}%"
        ]

        self.stats_label.setText('\n'.join(stats_text))

    def log_message(self, message: str):
        """Add message to log"""
        self.log_text.append(f"[INFO] {message}")
        self.logger.info(message)

    def log_error(self, message: str):
        """Add error message to log"""
        self.log_text.append(f"[ERROR] {message}")
        self.logger.error(message)

    def qpixmap_to_numpy(self, pixmap: QPixmap) -> np.ndarray:
        """Convert QPixmap to numpy array"""
        image = pixmap.toImage()
        width = image.width()
        height = image.height()

        # Convert to RGB format
        image = image.convertToFormat(QImage.Format.Format_RGB888)

        # Get image data
        ptr = image.bits()
        ptr.setsize(height * width * 3)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))

        # Convert RGB to BGR for OpenCV
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
