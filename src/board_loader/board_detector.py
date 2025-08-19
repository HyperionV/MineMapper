"""
Advanced Board Detection and Cell Recognition Module

This module implements sophisticated computer vision techniques for:
1. Automatic board boundary detection from screenshots
2. Grid cell extraction and classification
3. Template matching with multiple algorithms
4. Robust cell state recognition
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from pathlib import Path
import logging


class CellTemplate:
    """Represents a template for a specific cell type"""

    def __init__(self, cell_type: str, template_path: Path):
        self.cell_type = cell_type
        self.template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            raise ValueError(f"Could not load template: {template_path}")
        self.height, self.width = self.template.shape


class BoardDetector:
    """
    Advanced board detection using multiple computer vision techniques

    Combines template matching, edge detection, and contour analysis
    for robust board recognition across different screenshot qualities.
    """

    def __init__(self, templates_dir: Path):
        self.templates_dir = Path(templates_dir)
        self.cell_templates: Dict[str, CellTemplate] = {}
        self.logger = logging.getLogger(__name__)

        # Template matching parameters
        self.match_methods = [
            cv2.TM_CCOEFF_NORMED,
            cv2.TM_CCORR_NORMED,
            cv2.TM_SQDIFF_NORMED
        ]
        # Lower threshold to accommodate varied skins/fonts
        self.match_threshold = 0.55

        self._load_templates()

    def _load_templates(self) -> None:
        """Load all cell templates from the templates directory"""
        template_files = {
            'empty': 'empty.png',
            'flag': 'flag.png',
            '1': '1.png',
            '2': '2.png',
            '3': '3.png',
            '4': '4.png',
            '5': '5.png',
            '6': '6.png',
            '7': '7.png',
            '8': '8.png'
        }

        for cell_type, filename in template_files.items():
            template_path = self.templates_dir / filename
            if template_path.exists():
                try:
                    self.cell_templates[cell_type] = CellTemplate(
                        cell_type, template_path)
                    self.logger.info(f"Loaded template for {cell_type}")
                except Exception as e:
                    self.logger.error(
                        f"Failed to load template {filename}: {e}")
            else:
                self.logger.warning(
                    f"Template file not found: {template_path}")

    def detect_board_boundaries(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect board boundaries using edge detection and contour analysis

        Returns:
            Tuple of (x, y, width, height) or None if detection fails
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(
            image.shape) == 3 else image

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find the largest rectangular contour
        largest_area = 0
        best_rect = None

        for contour in contours:
            # Approximate contour to reduce points
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if contour is roughly rectangular
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h

                # Filter by size and aspect ratio
                if area > largest_area and w > 100 and h > 100:
                    aspect_ratio = w / h
                    if 0.5 < aspect_ratio < 3.0:  # Reasonable aspect ratio for minesweeper
                        largest_area = area
                        best_rect = (x, y, w, h)

        return best_rect

    def extract_grid_cells(self, image: np.ndarray, board_rect: Tuple[int, int, int, int],
                           grid_size: Tuple[int, int]) -> List[List[np.ndarray]]:
        """
        Extract individual cells from the detected board

        Args:
            image: Input image
            board_rect: Board boundaries (x, y, width, height)
            grid_size: Grid dimensions (rows, cols)

        Returns:
            2D list of cell images
        """
        x, y, w, h = board_rect
        rows, cols = grid_size

        # Extract board region and normalize contrast
        board_image = image[y:y+h, x:x+w]
        if len(board_image.shape) == 3:
            _gray_board = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
            _gray_board = cv2.equalizeHist(_gray_board)
            board_image = cv2.cvtColor(_gray_board, cv2.COLOR_GRAY2BGR)

        # Calculate cell dimensions
        cell_height = h // rows
        cell_width = w // cols

        cells = []
        for row in range(rows):
            cell_row = []
            for col in range(cols):
                # Calculate cell boundaries
                cell_y = row * cell_height
                cell_x = col * cell_width

                # Crop inside the cell to avoid grid lines (shrink by 10%)
                pad_h = int(cell_height * 0.1)
                pad_w = int(cell_width * 0.1)
                y0 = max(0, cell_y + pad_h)
                y1 = min(h, cell_y + cell_height - pad_h)
                x0 = max(0, cell_x + pad_w)
                x1 = min(w, cell_x + cell_width - pad_w)
                cell_img = board_image[y0:y1, x0:x1]
                cell_row.append(cell_img)

            cells.append(cell_row)

        return cells

    def classify_cell(self, cell_image: np.ndarray) -> Tuple[str, float]:
        """
        Classify a cell image using template matching

        Args:
            cell_image: Cell image to classify

        Returns:
            Tuple of (cell_type, confidence)
        """
        if cell_image.size == 0:
            return 'unknown', 0.0

        # Convert to grayscale and BGR for color heuristics
        if len(cell_image.shape) == 3:
            cell_bgr = cell_image
            cell_gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            cell_bgr = cv2.cvtColor(cell_image, cv2.COLOR_GRAY2BGR)
            cell_gray = cell_image.copy()
        cell_gray = cv2.GaussianBlur(cell_gray, (3, 3), 0)
        cell_gray = cv2.equalizeHist(cell_gray)

        # Heuristic quick checks (inner crop to avoid borders)
        h, w = cell_gray.shape
        x0, x1 = int(w*0.2), int(w*0.8)
        y0, y1 = int(h*0.2), int(h*0.8)
        inner = cell_gray[y0:y1, x0:x1]
        inner_bgr = cell_bgr[y0:y1, x0:x1]
        mean_val = float(np.mean(inner))
        std_val = float(np.std(inner))

        # Attempt color-based digit detection (robust across skins)
        color_type, color_conf = self._classify_by_color(inner_bgr, mean_val)

        best_match = 'unknown'
        best_confidence = 0.0

        # Try each template
        for cell_type, template in self.cell_templates.items():
            confidences = []

            # Try multiple matching methods for robustness
            for method in self.match_methods:
                try:
                    # Resize cell to match template if needed
                    if cell_gray.shape != template.template.shape:
                        resized_cell = cv2.resize(
                            cell_gray, (template.width, template.height))
                    else:
                        resized_cell = cell_gray

                    # Template matching on normalized inputs
                    tmpl = cv2.equalizeHist(template.template)
                    result = cv2.matchTemplate(
                        resized_cell, tmpl, method)
                    _, max_val, _, _ = cv2.minMaxLoc(result)

                    # Normalize confidence for SQDIFF methods (lower is better)
                    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                        confidence = 1.0 - max_val
                    else:
                        confidence = max_val

                    confidences.append(confidence)

                except Exception as e:
                    self.logger.debug(
                        f"Template matching failed for {cell_type} with method {method}: {e}")
                    continue

            if confidences:
                # Use average confidence across methods
                avg_confidence = np.mean(confidences)

                if avg_confidence > best_confidence and avg_confidence > self.match_threshold:
                    best_confidence = avg_confidence
                    best_match = cell_type

        # If template matching weak, trust color-based when confident
        if color_conf >= 0.7 and (best_confidence < 0.7 or best_match == 'unknown'):
            return color_type, color_conf

        # If nothing matched, decide empty vs unrevealed by brightness/variance
        if best_match == 'unknown':
            if mean_val < 90:  # darker tile → likely unrevealed
                return 'unknown', 0.3
            # low variance bright area with no color → empty
            if color_type == 'empty' or std_val < 15:
                return 'empty', 0.6

        return best_match, best_confidence

    def _classify_by_color(self, inner_bgr: np.ndarray, mean_val: float) -> Tuple[str, float]:
        """Classify using dominant color in HSV (handles colored digits)."""
        hsv = cv2.cvtColor(inner_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # mask colored pixels (ignore gray background)
        color_mask = (s > 60) & (v > 80)
        if np.count_nonzero(color_mask) < max(10, int(0.01 * color_mask.size)):
            # No significant color → probably empty if bright
            if mean_val > 140:
                return 'empty', 0.8
            return 'unknown', 0.2

        # dominant hue
        hues = h[color_mask]
        dom_hue = int(np.median(hues))  # robust to outliers

        # Map hue to number colors (OpenCV H in [0,179])
        def in_range(x, a, b):
            return a <= x <= b

        # Define canonical hue bands per common minesweeper palettes
        if in_range(dom_hue, 95, 130):
            return '1', 0.85  # blue
        if in_range(dom_hue, 45, 85):
            return '2', 0.85  # green
        if in_range(dom_hue, 0, 10) or in_range(dom_hue, 170, 179):
            return '3', 0.85  # red
        if in_range(dom_hue, 135, 160):
            return '4', 0.8   # purple/magenta
        if in_range(dom_hue, 15, 30):
            return '5', 0.7   # orange/brown
        if in_range(dom_hue, 85, 95):
            return '6', 0.7   # cyan/teal edge case
        # 7 and 8 are often black/gray → no strong hue; handled by grayscale
        return 'unknown', 0.3

    def analyze_board(self, image: np.ndarray, grid_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        Complete board analysis pipeline

        Args:
            image: Input screenshot
            grid_size: Expected grid dimensions (rows, cols)

        Returns:
            Dictionary containing board analysis results
        """
        results = {
            'success': False,
            'board_rect': None,
            'cells': None,
            'cell_classifications': None,
            'confidence_map': None
        }

        try:
            # Detect board boundaries
            board_rect = self.detect_board_boundaries(image)
            if board_rect is None:
                self.logger.error("Failed to detect board boundaries")
                return results

            results['board_rect'] = board_rect

            # Extract grid cells
            cells = self.extract_grid_cells(image, board_rect, grid_size)
            results['cells'] = cells

            # Classify each cell
            rows, cols = grid_size
            classifications = [
                [None for _ in range(cols)] for _ in range(rows)]
            confidence_map = [[0.0 for _ in range(cols)] for _ in range(rows)]

            for row in range(rows):
                for col in range(cols):
                    cell_type, confidence = self.classify_cell(cells[row][col])
                    classifications[row][col] = cell_type
                    confidence_map[row][col] = confidence

            results['cell_classifications'] = classifications
            results['confidence_map'] = confidence_map
            results['success'] = True

            self.logger.info(f"Successfully analyzed board: {rows}x{cols}")

        except Exception as e:
            self.logger.error(f"Board analysis failed: {e}")

        return results
