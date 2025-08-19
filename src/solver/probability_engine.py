"""
Advanced Probability Analysis Engine for Minesweeper

This module implements sophisticated probability calculation techniques:
1. Bayesian inference for mine probability estimation
2. Monte Carlo sampling for complex scenarios
3. Information theory approaches for optimal move selection
4. Global constraint integration
"""

from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
from collections import defaultdict, Counter
import random
import math
import logging

from .game_state import MinesweeperBoard
from .csp_solver import CSPSolver


class ProbabilityEngine:
    """
    Advanced probability analysis engine for minesweeper solving

    Features:
    - Exact probability calculation from CSP solutions
    - Monte Carlo sampling for large solution spaces
    - Bayesian inference with global constraints
    - Information theory metrics for move selection
    """

    def __init__(self, board: MinesweeperBoard):
        self.board = board
        self.csp_solver = CSPSolver(board)
        self.logger = logging.getLogger(__name__)

        # Probability computation parameters
        self.max_solutions_exact = 10000
        self.monte_carlo_samples = 50000
        self.min_probability_threshold = 0.001

        # Cache for expensive computations
        self._probability_cache: Dict[str, Dict[Tuple[int, int], float]] = {}

    def calculate_exact_probabilities(self) -> Dict[Tuple[int, int], float]:
        """
        Calculate exact mine probabilities from CSP solution enumeration

        Returns:
            Dictionary mapping cell coordinates to mine probabilities
        """
        # Get all valid solutions
        solutions = self.csp_solver.enumerate_solutions(
            self.max_solutions_exact)

        if not solutions:
            self.logger.warning(
                "No CSP solutions found for probability calculation")
            return {}

        # Count mine occurrences for each cell
        mine_counts = defaultdict(int)
        total_solutions = len(solutions)

        for solution in solutions:
            for cell, is_mine in solution.items():
                if is_mine:
                    mine_counts[cell] += 1

        # Calculate probabilities
        probabilities = {}
        for cell in self.csp_solver.variables:
            prob = mine_counts[cell] / total_solutions
            if prob > self.min_probability_threshold:
                probabilities[cell] = prob
            else:
                probabilities[cell] = 0.0

        self.logger.info(
            f"Calculated exact probabilities for {len(probabilities)} cells")
        return probabilities

    def monte_carlo_probabilities(self) -> Dict[Tuple[int, int], float]:
        """
        Estimate probabilities using Monte Carlo sampling

        This method is used when the solution space is too large for
        exact enumeration.
        """
        self.csp_solver.setup_csp()

        if not self.csp_solver.variables:
            return {}

        mine_counts = defaultdict(int)
        valid_samples = 0

        for _ in range(self.monte_carlo_samples):
            # Generate random assignment
            assignment = {}
            for var in self.csp_solver.variables:
                assignment[var] = random.choice([0, 1])

            # Check if assignment satisfies constraints
            if self._is_valid_assignment(assignment):
                valid_samples += 1
                for cell, is_mine in assignment.items():
                    if is_mine:
                        mine_counts[cell] += 1

        if valid_samples == 0:
            self.logger.warning(
                "No valid samples found in Monte Carlo simulation")
            return {}

        # Calculate probabilities
        probabilities = {}
        for cell in self.csp_solver.variables:
            probabilities[cell] = mine_counts[cell] / valid_samples

        self.logger.info(
            f"Monte Carlo: {valid_samples} valid samples from {self.monte_carlo_samples}")
        return probabilities

    def _is_valid_assignment(self, assignment: Dict[Tuple[int, int], int]) -> bool:
        """Check if assignment satisfies all constraints"""
        for constraint in self.csp_solver.constraints:
            total = sum(assignment.get(var, 0) for var in constraint.variables)
            if total != constraint.rhs:
                return False

        # Check global mine count constraint
        total_mines = sum(assignment.values())
        remaining_cells = len([(r, c) for r in range(self.board.rows)
                               for c in range(self.board.cols)
                               if self.board.get_cell(r, c).is_unrevealed
                               and (r, c) not in assignment])

        # Ensure total mines doesn't exceed board limit
        current_flagged = self.board.flagged_count
        if total_mines + current_flagged > self.board.total_mines:
            return False

        return True

    def bayesian_inference(self) -> Dict[Tuple[int, int], float]:
        """
        Apply Bayesian inference to incorporate global constraints

        This method adjusts local probabilities based on global
        mine count constraints.
        """
        # Get local probabilities
        local_probs = self.calculate_exact_probabilities()
        if not local_probs:
            local_probs = self.monte_carlo_probabilities()

        if not local_probs:
            return {}

        # Apply global constraint adjustment
        total_unrevealed = len([cell for r in range(self.board.rows)
                                for c in range(self.board.cols)
                                if self.board.get_cell(r, c).is_unrevealed])

        remaining_mines = self.board.remaining_mines

        if total_unrevealed == 0:
            return {}

        # Calculate expected mines from local probabilities
        expected_mines = sum(local_probs.values())

        # Adjust probabilities to match global constraint
        if expected_mines > 0 and abs(expected_mines - remaining_mines) > 1e-6:
            adjustment_factor = remaining_mines / expected_mines

            # Apply adjustment with bounds checking
            adjusted_probs = {}
            for cell, prob in local_probs.items():
                adjusted_prob = min(1.0, max(0.0, prob * adjustment_factor))
                adjusted_probs[cell] = adjusted_prob

            self.logger.info(
                f"Applied Bayesian adjustment: factor = {adjustment_factor:.3f}")
            return adjusted_probs

        return local_probs

    def information_theory_analysis(self, probabilities: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
        """
        Calculate information gain for each potential move

        Uses entropy and information theory to determine which moves
        provide the most information about the game state.
        """
        information_gains = {}

        for cell, prob in probabilities.items():
            if prob <= 0 or prob >= 1:
                information_gains[cell] = 0.0
                continue

            # Calculate entropy
            entropy = -prob * math.log2(prob) - \
                (1 - prob) * math.log2(1 - prob)

            # Weight by expected information gain
            # Higher entropy = more uncertainty = more potential information
            information_gains[cell] = entropy

        return information_gains

    def calculate_move_values(self) -> Dict[str, List[Tuple[Tuple[int, int], float]]]:
        """
        Calculate comprehensive move values using multiple criteria

        Returns:
            Dictionary with different move ranking criteria
        """
        # Get probabilities
        probabilities = self.bayesian_inference()

        if not probabilities:
            return {'safe': [], 'risky': [], 'information': []}

        # Update board cell probabilities
        for (r, c), prob in probabilities.items():
            cell = self.board.get_cell(r, c)
            if cell:
                cell.probability = prob

        # Calculate information gains
        information_gains = self.information_theory_analysis(probabilities)

        # Categorize moves
        safe_moves = []
        risky_moves = []
        information_moves = []

        for cell, prob in probabilities.items():
            # Safe moves (low mine probability)
            if prob < 0.1:
                safe_moves.append((cell, prob))

            # Risky moves (high mine probability)
            elif prob > 0.7:
                # Invert for flagging priority
                risky_moves.append((cell, 1.0 - prob))

            # Information-rich moves
            info_gain = information_gains.get(cell, 0.0)
            if info_gain > 0.1:
                information_moves.append((cell, info_gain))

        # Sort moves
        safe_moves.sort(key=lambda x: x[1])  # Ascending probability
        risky_moves.sort(key=lambda x: x[1], reverse=True)  # Descending safety
        # Descending information
        information_moves.sort(key=lambda x: x[1], reverse=True)

        return {
            'safe': safe_moves,
            'risky': risky_moves,
            'information': information_moves
        }

    def get_best_move(self, strategy: str = 'safe') -> Optional[Tuple[Tuple[int, int], float, str]]:
        """
        Get the best move according to specified strategy

        Args:
            strategy: 'safe', 'information', or 'balanced'

        Returns:
            Tuple of (cell, value, description) or None
        """
        move_values = self.calculate_move_values()

        if strategy == 'safe' and move_values['safe']:
            cell, prob = move_values['safe'][0]
            return (cell, prob, f"Safe move (mine probability: {prob:.3f})")

        elif strategy == 'information' and move_values['information']:
            cell, info_gain = move_values['information'][0]
            prob = self.board.get_cell(*cell).probability
            return (cell, info_gain, f"Information move (entropy: {info_gain:.3f}, mine prob: {prob:.3f})")

        elif strategy == 'balanced':
            # Balance safety and information gain
            best_score = -1
            best_move = None

            for moves, weight in [(move_values['safe'], 2.0), (move_values['information'], 1.0)]:
                for i, (cell, value) in enumerate(moves[:5]):  # Top 5 moves
                    # Score based on rank and weight
                    score = weight * (len(moves) - i) / len(moves)
                    if score > best_score:
                        best_score = score
                        prob = self.board.get_cell(*cell).probability
                        best_move = (
                            cell, value, f"Balanced move (score: {score:.3f}, mine prob: {prob:.3f})")

            return best_move

        return None

    def analyze_position(self) -> Dict[str, Any]:
        """
        Comprehensive position analysis

        Returns:
            Dictionary with complete probability analysis
        """
        results = {
            'probabilities': {},
            'move_values': {},
            'best_moves': {},
            'statistics': {},
            'method_used': 'none'
        }

        try:
            # Calculate probabilities
            if len(self.csp_solver.variables or []) <= self.max_solutions_exact:
                probabilities = self.calculate_exact_probabilities()
                results['method_used'] = 'exact'
            else:
                probabilities = self.monte_carlo_probabilities()
                results['method_used'] = 'monte_carlo'

            if probabilities:
                # Apply Bayesian inference
                probabilities = self.bayesian_inference()
                results['probabilities'] = probabilities

                # Calculate move values
                move_values = self.calculate_move_values()
                results['move_values'] = move_values

                # Get best moves for different strategies
                results['best_moves'] = {
                    'safe': self.get_best_move('safe'),
                    'information': self.get_best_move('information'),
                    'balanced': self.get_best_move('balanced')
                }

                # Calculate statistics
                probs = list(probabilities.values())
                results['statistics'] = {
                    'total_cells': len(probs),
                    'mean_probability': np.mean(probs) if probs else 0,
                    'std_probability': np.std(probs) if probs else 0,
                    'min_probability': min(probs) if probs else 0,
                    'max_probability': max(probs) if probs else 0,
                    'safe_moves_count': len(move_values.get('safe', [])),
                    'risky_moves_count': len(move_values.get('risky', [])),
                    'information_moves_count': len(move_values.get('information', []))
                }

        except Exception as e:
            self.logger.error(f"Probability analysis failed: {e}")

        return results
