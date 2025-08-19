"""
HTML Parser for Minesweeper Board Extraction
Parses HTML DOM structure to extract exact board state
"""
import re
import logging
from typing import Dict, Tuple, Optional
from bs4 import BeautifulSoup
from ..solver.game_state import MinesweeperBoard, CellState, CellContent


class HTMLBoardParser:
    """Parse minesweeper board from HTML DOM structure"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse_board_html(self, html_content: str, expected_mines: int = None) -> Optional[MinesweeperBoard]:
        """
        Parse HTML content and extract board state

        Args:
            html_content: HTML string containing minesweeper board structure
            expected_mines: Expected total mine count from UI (overrides estimate)

        Returns:
            MinesweeperBoard object or None if parsing fails
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find all cell divs
            cells = soup.find_all('div', class_=lambda x: x and 'cell' in x)

            if not cells:
                self.logger.error("No cell elements found in HTML")
                return None

            # Extract board dimensions and cell data
            max_x, max_y = 0, 0
            cell_data = {}

            for cell_div in cells:
                try:
                    # Extract coordinates
                    x = int(cell_div.get('data-x', 0))
                    y = int(cell_div.get('data-y', 0))
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)

                    # Parse cell state from CSS classes
                    classes = cell_div.get('class', [])
                    cell_state, cell_content = self._parse_cell_classes(
                        classes)

                    cell_data[(x, y)] = (cell_state, cell_content)

                except (ValueError, AttributeError) as e:
                    self.logger.warning(
                        f"Failed to parse cell {cell_div}: {e}")
                    continue

            # Create board
            rows, cols = max_y + 1, max_x + 1
            self.logger.info(f"Detected board size: {cols}x{rows}")

            flagged_count = sum(
                1 for state, _ in cell_data.values() if state == CellState.FLAGGED)

            # Use expected mines from UI if provided, otherwise estimate
            if expected_mines is not None:
                total_mines = expected_mines
                self.logger.info(f"Using expected mine count: {total_mines}")
            else:
                total_mines = flagged_count + max(1, int(rows * cols * 0.15))
                self.logger.info(f"Estimated mine count: {total_mines}")

            board = MinesweeperBoard(rows, cols, total_mines)

            # Populate board with parsed data
            for (x, y), (state, content) in cell_data.items():
                if 0 <= y < rows and 0 <= x < cols:
                    # Note: board uses (row, col) = (y, x)
                    cell = board.get_cell(y, x)
                    cell.state = state
                    cell.content = content

            self.logger.info(
                f"Successfully parsed board: {cols}x{rows} with {flagged_count} flagged cells")
            return board

        except Exception as e:
            self.logger.error(f"Failed to parse HTML board: {e}")
            return None

    def _parse_cell_classes(self, classes: list) -> Tuple[CellState, CellContent]:
        """
        Parse cell state and content from CSS classes

        Expected patterns:
        - hdd_closed: unrevealed cell
        - hdd_closed hdd_flag: flagged cell  
        - hdd_opened hdd_type0: empty revealed cell
        - hdd_opened hdd_type1: revealed cell with number 1
        - hdd_opened hdd_type2: revealed cell with number 2
        - etc.
        """
        classes_str = ' '.join(classes)

        # Check for flagged cell
        if 'hdd_flag' in classes:
            return CellState.FLAGGED, CellContent.UNKNOWN

        # Check for closed/unrevealed cell
        if 'hdd_closed' in classes and 'hdd_flag' not in classes:
            return CellState.UNREVEALED, CellContent.UNKNOWN

        # Check for opened cells
        if 'hdd_opened' in classes:
            # Look for type number
            type_match = re.search(r'hdd_type(\d+)', classes_str)
            if type_match:
                type_num = int(type_match.group(1))
                if type_num == 0:
                    return CellState.REVEALED, CellContent.EMPTY
                elif 1 <= type_num <= 8:
                    # Map to correct enum values: ONE, TWO, THREE, etc.
                    content_map = {
                        1: CellContent.ONE,
                        2: CellContent.TWO,
                        3: CellContent.THREE,
                        4: CellContent.FOUR,
                        5: CellContent.FIVE,
                        6: CellContent.SIX,
                        7: CellContent.SEVEN,
                        8: CellContent.EIGHT
                    }
                    return CellState.REVEALED, content_map[type_num]

            # Default opened cell (assume empty)
            return CellState.REVEALED, CellContent.EMPTY

        # Default fallback
        return CellState.UNREVEALED, CellContent.UNKNOWN

    def estimate_mine_count(self, html_content: str) -> int:
        """
        Estimate total mine count from HTML structure
        This is a fallback - ideally get from game UI
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            cells = soup.find_all('div', class_=lambda x: x and 'cell' in x)

            flagged = len(
                [c for c in cells if 'hdd_flag' in c.get('class', [])])
            total_cells = len(cells)

            # Common minesweeper ratios
            if total_cells <= 81:  # 9x9 beginner
                return max(flagged, 10)
            elif total_cells <= 480:  # 16x30 intermediate
                return max(flagged, 40)
            else:  # expert or custom
                return max(flagged, int(total_cells * 0.15))

        except Exception:
            return max(1, int(len(cells) * 0.15)) if 'cells' in locals() else 50
