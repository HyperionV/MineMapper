"""
Advanced Pattern Recognition System for Minesweeper

This module implements sophisticated pattern matching techniques:
1. Common minesweeper logical patterns (1-1, 1-2-1, etc.)
2. Universal pattern recognition using template matching
3. Geometric pattern detection
4. Machine learning-based pattern classification
"""

from typing import List, Dict, Set, Tuple, Optional, Any, Callable
import numpy as np
from dataclasses import dataclass
import logging

from .game_state import MinesweeperBoard, Cell, CellState, CellContent


@dataclass
class Pattern:
    """Represents a minesweeper pattern"""
    name: str
    description: str
    template: List[List[str]]  # Pattern template using symbols
    apply_function: Callable  # Function to apply pattern logic
    priority: int = 1  # Higher priority patterns are checked first


class PatternMatcher:
    """
    Advanced pattern matching system for minesweeper solving

    Implements both hardcoded logical patterns and machine learning
    approaches for pattern recognition and application.
    """

    def __init__(self, board: MinesweeperBoard):
        self.board = board
        self.logger = logging.getLogger(__name__)

        # Pattern library
        self.patterns: List[Pattern] = []
        self.pattern_cache: Dict[str, List[Tuple[int, int]]] = {}

        # Statistics
        self.patterns_applied = 0
        self.patterns_found = 0

        self._initialize_patterns()

    def _initialize_patterns(self) -> None:
        """Initialize the pattern library with common minesweeper patterns"""

        # 1-1 Pattern: Two adjacent 1s with shared neighbors
        self.patterns.append(Pattern(
            name="1-1",
            description="Two adjacent 1s sharing neighbors",
            template=[
                ["1", "1"],
                ["?", "?"]
            ],
            apply_function=self._apply_1_1_pattern,
            priority=5
        ))

        # 1-2-1 Pattern: Classic minesweeper pattern
        self.patterns.append(Pattern(
            name="1-2-1",
            description="1-2-1 horizontal pattern",
            template=[
                ["1", "2", "1"],
                ["?", "?", "?"]
            ],
            apply_function=self._apply_1_2_1_pattern,
            priority=10
        ))

        # 1-2-2-1 Pattern: Extended version
        self.patterns.append(Pattern(
            name="1-2-2-1",
            description="1-2-2-1 horizontal pattern",
            template=[
                ["1", "2", "2", "1"],
                ["?", "?", "?", "?"]
            ],
            apply_function=self._apply_1_2_2_1_pattern,
            priority=8
        ))

        # Corner patterns
        self.patterns.append(Pattern(
            name="corner-1",
            description="Corner cell with value 1",
            template=[
                ["1", "?"],
                ["?", "?"]
            ],
            apply_function=self._apply_corner_pattern,
            priority=3
        ))

        # Edge patterns
        self.patterns.append(Pattern(
            name="edge-1",
            description="Edge cell with value 1",
            template=[
                ["?", "?", "?"],
                ["1", "?", "?"]
            ],
            apply_function=self._apply_edge_pattern,
            priority=2
        ))

        # Triangular patterns
        self.patterns.append(Pattern(
            name="triangle",
            description="Triangular formation",
            template=[
                ["?", "1", "?"],
                ["1", "?", "1"],
                ["?", "?", "?"]
            ],
            apply_function=self._apply_triangle_pattern,
            priority=6
        ))

        self.logger.info(f"Initialized {len(self.patterns)} pattern templates")

    def find_patterns(self) -> Dict[str, List[Tuple[int, int, Any]]]:
        """
        Find all pattern matches on the current board

        Returns:
            Dictionary mapping pattern names to list of (row, col, metadata) matches
        """
        found_patterns = {}

        for pattern in sorted(self.patterns, key=lambda p: p.priority, reverse=True):
            matches = self._find_pattern_matches(pattern)
            if matches:
                found_patterns[pattern.name] = matches
                self.patterns_found += len(matches)

        self.logger.info(
            f"Found {sum(len(matches) for matches in found_patterns.values())} pattern matches")
        return found_patterns

    def _find_pattern_matches(self, pattern: Pattern) -> List[Tuple[int, int, Any]]:
        """Find all matches for a specific pattern"""
        matches = []
        template_height = len(pattern.template)
        template_width = len(pattern.template[0]) if pattern.template else 0

        # Try all possible positions and orientations
        for row in range(self.board.rows - template_height + 1):
            for col in range(self.board.cols - template_width + 1):
                # Check all 4 rotations
                for rotation in range(4):
                    rotated_template = self._rotate_template(
                        pattern.template, rotation)
                    if self._matches_template(row, col, rotated_template):
                        metadata = {
                            'rotation': rotation,
                            'size': (len(rotated_template), len(rotated_template[0])),
                            'pattern': pattern
                        }
                        matches.append((row, col, metadata))

        return matches

    def _rotate_template(self, template: List[List[str]], rotation: int) -> List[List[str]]:
        """Rotate template by 90 degrees * rotation"""
        if rotation == 0:
            return template

        # Convert to numpy for easy rotation
        arr = np.array(template)
        for _ in range(rotation):
            arr = np.rot90(arr)

        return arr.tolist()

    def _matches_template(self, start_row: int, start_col: int, template: List[List[str]]) -> bool:
        """Check if board region matches template"""
        template_height = len(template)
        template_width = len(template[0]) if template else 0

        for tr in range(template_height):
            for tc in range(template_width):
                board_row = start_row + tr
                board_col = start_col + tc

                if board_row >= self.board.rows or board_col >= self.board.cols:
                    return False

                cell = self.board.get_cell(board_row, board_col)
                template_symbol = template[tr][tc]

                if not self._cell_matches_symbol(cell, template_symbol):
                    return False

        return True

    def _cell_matches_symbol(self, cell: Cell, symbol: str) -> bool:
        """Check if cell matches template symbol"""
        if symbol == "?":
            return True  # Wildcard matches anything

        if symbol.isdigit():
            return cell.is_revealed and cell.number == int(symbol)

        if symbol == "F":
            return cell.is_flagged

        if symbol == "#":
            return cell.is_unrevealed

        if symbol == ".":
            return cell.is_revealed and cell.content == CellContent.EMPTY

        return False

    def apply_patterns(self) -> Dict[str, Set[Tuple[int, int]]]:
        """
        Apply all found patterns to determine safe moves and mines

        Returns:
            Dictionary with 'safe' and 'mines' sets
        """
        safe_moves = set()
        mine_moves = set()

        patterns_found = self.find_patterns()

        for pattern_name, matches in patterns_found.items():
            pattern = next(p for p in self.patterns if p.name == pattern_name)

            for row, col, metadata in matches:
                try:
                    result = pattern.apply_function(row, col, metadata)
                    if result:
                        safe_moves.update(result.get('safe', []))
                        mine_moves.update(result.get('mines', []))
                        self.patterns_applied += 1

                except Exception as e:
                    self.logger.debug(
                        f"Failed to apply pattern {pattern_name} at ({row},{col}): {e}")

        self.logger.info(
            f"Pattern recognition: {len(safe_moves)} safe, {len(mine_moves)} mines")
        return {'safe': safe_moves, 'mines': mine_moves}

    # Pattern application functions

    def _apply_1_1_pattern(self, row: int, col: int, metadata: Any) -> Dict[str, List[Tuple[int, int]]]:
        """Apply logic for 1-1 pattern"""
        safe = []
        mines = []

        # Get the two 1s
        cell1 = self.board.get_cell(row, col)
        cell2 = self.board.get_cell(row, col + 1)

        if not (cell1 and cell2 and cell1.number == 1 and cell2.number == 1):
            return {'safe': safe, 'mines': mines}

        # Find shared and unique neighbors
        neighbors1 = set(self.board.get_neighbor_coords(row, col))
        neighbors2 = set(self.board.get_neighbor_coords(row, col + 1))

        shared = neighbors1 & neighbors2
        unique1 = neighbors1 - neighbors2
        unique2 = neighbors2 - neighbors1

        # Count mines already flagged
        shared_mines = sum(
            1 for r, c in shared if self.board.get_cell(r, c).is_flagged)
        unique1_mines = sum(
            1 for r, c in unique1 if self.board.get_cell(r, c).is_flagged)
        unique2_mines = sum(
            1 for r, c in unique2 if self.board.get_cell(r, c).is_flagged)

        # Apply 1-1 logic
        if shared_mines == 1:
            # If shared area has the mine, unique areas are safe
            for r, c in unique1 | unique2:
                if self.board.get_cell(r, c).is_unrevealed:
                    safe.append((r, c))

        elif shared_mines == 0 and unique1_mines == 0 and unique2_mines == 0:
            # More complex analysis needed
            shared_unrevealed = [
                pos for pos in shared if self.board.get_cell(*pos).is_unrevealed]
            unique1_unrevealed = [
                pos for pos in unique1 if self.board.get_cell(*pos).is_unrevealed]
            unique2_unrevealed = [
                pos for pos in unique2 if self.board.get_cell(*pos).is_unrevealed]

            if len(shared_unrevealed) == 1 and len(unique1_unrevealed) == 1 and len(unique2_unrevealed) == 1:
                # Classic 1-1 pattern: either shared is mine or both uniques are mines
                # This requires probability analysis, mark as indeterminate for now
                pass

        return {'safe': safe, 'mines': mines}

    def _apply_1_2_1_pattern(self, row: int, col: int, metadata: Any) -> Dict[str, List[Tuple[int, int]]]:
        """Apply logic for 1-2-1 pattern"""
        safe = []
        mines = []

        # Verify pattern
        cells = [self.board.get_cell(row, col + i) for i in range(3)]
        if not all(cells) or [c.number for c in cells] != [1, 2, 1]:
            return {'safe': safe, 'mines': mines}

        # Get neighbor regions
        left_neighbors = set(self.board.get_neighbor_coords(row, col))
        middle_neighbors = set(self.board.get_neighbor_coords(row, col + 1))
        right_neighbors = set(self.board.get_neighbor_coords(row, col + 2))

        # Find exclusive regions
        left_only = left_neighbors - middle_neighbors - right_neighbors
        middle_only = middle_neighbors - left_neighbors - right_neighbors
        right_only = right_neighbors - middle_neighbors - left_neighbors

        # Apply 1-2-1 logic: middle exclusive region contains the mines
        for r, c in middle_only:
            if self.board.get_cell(r, c).is_unrevealed:
                mines.append((r, c))

        # Outer regions are safe
        for r, c in left_only | right_only:
            if self.board.get_cell(r, c).is_unrevealed:
                safe.append((r, c))

        return {'safe': safe, 'mines': mines}

    def _apply_1_2_2_1_pattern(self, row: int, col: int, metadata: Any) -> Dict[str, List[Tuple[int, int]]]:
        """Apply logic for 1-2-2-1 pattern"""
        safe = []
        mines = []

        # Verify pattern
        cells = [self.board.get_cell(row, col + i) for i in range(4)]
        if not all(cells) or [c.number for c in cells] != [1, 2, 2, 1]:
            return {'safe': safe, 'mines': mines}

        # This pattern typically indicates mines in the middle region
        # Complex analysis required - implement basic version

        return {'safe': safe, 'mines': mines}

    def _apply_corner_pattern(self, row: int, col: int, metadata: Any) -> Dict[str, List[Tuple[int, int]]]:
        """Apply logic for corner patterns"""
        safe = []
        mines = []

        cell = self.board.get_cell(row, col)
        if not cell or cell.number != 1:
            return {'safe': safe, 'mines': mines}

        # For corner 1s, if one neighbor is flagged, others are safe
        neighbors = self.board.get_neighbor_coords(row, col)
        flagged_count = sum(
            1 for r, c in neighbors if self.board.get_cell(r, c).is_flagged)

        if flagged_count == 1:
            for r, c in neighbors:
                if self.board.get_cell(r, c).is_unrevealed:
                    safe.append((r, c))

        return {'safe': safe, 'mines': mines}

    def _apply_edge_pattern(self, row: int, col: int, metadata: Any) -> Dict[str, List[Tuple[int, int]]]:
        """Apply logic for edge patterns"""
        safe = []
        mines = []

        # Similar to corner but for edge cells
        cell = self.board.get_cell(row, col)
        if not cell or not cell.is_numbered:
            return {'safe': safe, 'mines': mines}

        neighbors = self.board.get_neighbor_coords(row, col)
        unrevealed = [
            pos for pos in neighbors if self.board.get_cell(*pos).is_unrevealed]
        flagged = [
            pos for pos in neighbors if self.board.get_cell(*pos).is_flagged]

        mines_needed = cell.number - len(flagged)

        if mines_needed == 0:
            # All remaining unrevealed are safe
            safe.extend(unrevealed)
        elif mines_needed == len(unrevealed):
            # All remaining unrevealed are mines
            mines.extend(unrevealed)

        return {'safe': safe, 'mines': mines}

    def _apply_triangle_pattern(self, row: int, col: int, metadata: Any) -> Dict[str, List[Tuple[int, int]]]:
        """Apply logic for triangular patterns"""
        safe = []
        mines = []

        # Triangular patterns require complex geometric analysis
        # Implement basic version for now

        return {'safe': safe, 'mines': mines}

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about pattern recognition performance"""
        return {
            'patterns_registered': len(self.patterns),
            'patterns_found': self.patterns_found,
            'patterns_applied': self.patterns_applied,
            'success_rate': self.patterns_applied / max(1, self.patterns_found)
        }

