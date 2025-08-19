"""
Minesweeper Game State and Board Representation

This module provides the core data structures and logic for representing
and manipulating minesweeper game states, optimized for algorithmic solving.
"""

from enum import Enum
from typing import Set, Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np


class CellState(Enum):
    """Represents the state of a minesweeper cell"""
    UNREVEALED = "unrevealed"
    REVEALED = "revealed"
    FLAGGED = "flagged"
    MINE = "mine"


class CellContent(Enum):
    """Represents the content of a minesweeper cell"""
    EMPTY = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    MINE = -1
    UNKNOWN = -2


@dataclass
class Cell:
    """Represents a single minesweeper cell"""
    row: int
    col: int
    state: CellState = CellState.UNREVEALED
    content: CellContent = CellContent.UNKNOWN
    probability: float = 0.0  # Probability of containing a mine
    confidence: float = 0.0   # Confidence in the classification

    @property
    def is_revealed(self) -> bool:
        return self.state == CellState.REVEALED

    @property
    def is_flagged(self) -> bool:
        return self.state == CellState.FLAGGED

    @property
    def is_unrevealed(self) -> bool:
        return self.state == CellState.UNREVEALED

    @property
    def is_numbered(self) -> bool:
        return self.content.value > 0 and self.content.value <= 8

    @property
    def number(self) -> int:
        """Get the number on this cell (0 if empty, -1 if not a number)"""
        if self.content.value >= 0 and self.content.value <= 8:
            return self.content.value
        return -1


class MinesweeperBoard:
    """
    Advanced minesweeper board representation optimized for solving algorithms

    Features:
    - Efficient neighbor computation
    - Constraint tracking
    - Probability management
    - Move validation
    """

    def __init__(self, rows: int, cols: int, total_mines: int):
        self.rows = rows
        self.cols = cols
        self.total_mines = total_mines

        # Initialize board with unrevealed cells
        self.board: List[List[Cell]] = []
        for r in range(rows):
            row = []
            for c in range(cols):
                row.append(Cell(r, c))
            self.board.append(row)

        # Game state tracking
        self.revealed_count = 0
        self.flagged_count = 0
        self.remaining_mines = total_mines

        # Solving state
        self.safe_cells: Set[Tuple[int, int]] = set()
        self.mine_cells: Set[Tuple[int, int]] = set()
        # Unrevealed cells adjacent to revealed
        self.frontier_cells: Set[Tuple[int, int]] = set()

        # Precompute neighbor mappings for efficiency
        self._neighbor_cache: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self._precompute_neighbors()

    def _precompute_neighbors(self) -> None:
        """Precompute neighbor coordinates for all cells"""
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0), (1, 1)]

        for r in range(self.rows):
            for c in range(self.cols):
                neighbors = []
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        neighbors.append((nr, nc))
                self._neighbor_cache[(r, c)] = neighbors

    def get_cell(self, row: int, col: int) -> Optional[Cell]:
        """Get cell at specified coordinates"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.board[row][col]
        return None

    def get_neighbors(self, row: int, col: int) -> List[Cell]:
        """Get all neighboring cells"""
        neighbors = []
        for nr, nc in self._neighbor_cache.get((row, col), []):
            neighbors.append(self.board[nr][nc])
        return neighbors

    def get_neighbor_coords(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get coordinates of all neighboring cells"""
        return self._neighbor_cache.get((row, col), [])

    def update_cell_from_vision(self, row: int, col: int, cell_type: str, confidence: float) -> bool:
        """
        Update cell based on computer vision classification

        Args:
            row, col: Cell coordinates
            cell_type: Detected cell type ('empty', '1'-'8', 'flag', etc.)
            confidence: Classification confidence

        Returns:
            True if cell was successfully updated
        """
        cell = self.get_cell(row, col)
        if cell is None:
            return False

        cell.confidence = confidence

        # Map vision results to game state
        if cell_type == 'flag':
            cell.state = CellState.FLAGGED
            cell.content = CellContent.UNKNOWN
            self.mine_cells.add((row, col))
            self.flagged_count += 1
            self.remaining_mines -= 1

        elif cell_type == 'empty':
            cell.state = CellState.REVEALED
            cell.content = CellContent.EMPTY
            self.revealed_count += 1
            self.safe_cells.add((row, col))

        elif cell_type in ['1', '2', '3', '4', '5', '6', '7', '8']:
            cell.state = CellState.REVEALED
            cell.content = CellContent(int(cell_type))
            self.revealed_count += 1
            self.safe_cells.add((row, col))

        else:  # Unknown or unrevealed
            cell.state = CellState.UNREVEALED
            cell.content = CellContent.UNKNOWN

        return True

    def update_frontier(self) -> None:
        """Update frontier cells (unrevealed cells adjacent to revealed cells)"""
        self.frontier_cells.clear()

        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.board[r][c]
                if cell.is_unrevealed:
                    # Check if adjacent to any revealed cell
                    for neighbor in self.get_neighbors(r, c):
                        if neighbor.is_revealed:
                            self.frontier_cells.add((r, c))
                            break

    def get_constraints(self) -> List[Dict[str, Any]]:
        """
        Generate constraint equations for CSP solving

        Returns:
            List of constraint dictionaries with:
            - 'cell': (row, col) of the numbered cell
            - 'neighbors': list of unrevealed neighbor coordinates
            - 'mines_needed': number of mines still needed around this cell
        """
        constraints = []

        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.board[r][c]

                if cell.is_revealed and cell.is_numbered:
                    # Get unrevealed neighbors
                    unrevealed_neighbors = []
                    flagged_neighbors = 0

                    for nr, nc in self.get_neighbor_coords(r, c):
                        neighbor = self.board[nr][nc]
                        if neighbor.is_unrevealed:
                            unrevealed_neighbors.append((nr, nc))
                        elif neighbor.is_flagged:
                            flagged_neighbors += 1

                    # Calculate remaining mines needed
                    mines_needed = cell.number - flagged_neighbors

                    if unrevealed_neighbors and mines_needed >= 0:
                        constraints.append({
                            'cell': (r, c),
                            'neighbors': unrevealed_neighbors,
                            'mines_needed': mines_needed,
                            'total_neighbors': len(unrevealed_neighbors)
                        })

        return constraints

    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if a move is valid (cell exists and is unrevealed)"""
        cell = self.get_cell(row, col)
        return cell is not None and cell.is_unrevealed

    def get_safe_moves(self) -> List[Tuple[int, int]]:
        """Get list of definitely safe moves"""
        safe_moves = []
        for r, c in self.safe_cells:
            if self.is_valid_move(r, c):
                safe_moves.append((r, c))
        return safe_moves

    def get_mine_moves(self) -> List[Tuple[int, int]]:
        """Get list of definite mine locations to flag"""
        mine_moves = []
        for r, c in self.mine_cells:
            if self.is_valid_move(r, c):
                mine_moves.append((r, c))
        return mine_moves

    def get_probability_moves(self, min_probability: float = 0.0) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get moves with their mine probabilities

        Args:
            min_probability: Minimum probability threshold

        Returns:
            List of ((row, col), probability) tuples
        """
        probability_moves = []

        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.board[r][c]
                if cell.is_unrevealed and cell.probability >= min_probability:
                    probability_moves.append(((r, c), cell.probability))

        # Sort by probability (ascending - lower probability = safer)
        probability_moves.sort(key=lambda x: x[1])
        return probability_moves

    def get_game_statistics(self) -> Dict[str, Any]:
        """Get current game statistics"""
        total_cells = self.rows * self.cols
        unrevealed_cells = total_cells - self.revealed_count - self.flagged_count

        return {
            'total_cells': total_cells,
            'revealed_cells': self.revealed_count,
            'flagged_cells': self.flagged_count,
            'unrevealed_cells': unrevealed_cells,
            'total_mines': self.total_mines,
            'remaining_mines': self.remaining_mines,
            'frontier_size': len(self.frontier_cells),
            'completion_percentage': (self.revealed_count / (total_cells - self.total_mines)) * 100
        }

    def to_array(self, property_name: str = 'content') -> np.ndarray:
        """
        Convert board to numpy array for analysis

        Args:
            property_name: Cell property to extract ('content', 'state', 'probability')
        """
        array = np.zeros((self.rows, self.cols), dtype=float)

        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.board[r][c]
                if property_name == 'content':
                    array[r, c] = cell.content.value
                elif property_name == 'state':
                    array[r, c] = hash(cell.state.value)
                elif property_name == 'probability':
                    array[r, c] = cell.probability
                else:
                    array[r, c] = 0

        return array

    def __str__(self) -> str:
        """String representation of the board"""
        lines = []
        for row in self.board:
            line = []
            for cell in row:
                if cell.is_flagged:
                    line.append('F')
                elif cell.is_revealed:
                    if cell.content == CellContent.EMPTY:
                        line.append('.')
                    elif cell.is_numbered:
                        line.append(str(cell.number))
                    else:
                        line.append('?')
                else:
                    line.append('#')
            lines.append(' '.join(line))
        return '\n'.join(lines)

