"""
Advanced Constraint Satisfaction Problem (CSP) Solver for Minesweeper

This module implements sophisticated CSP algorithms including:
1. Deterministic Solution Search (DSS) with dependency separation
2. Constraint propagation and reduction
3. Gaussian elimination for linear systems
4. Backtracking with intelligent pruning
"""

from typing import List, Dict, Set, Tuple, Optional, Any
import numpy as np
from itertools import combinations, product
import logging
from dataclasses import dataclass

from .game_state import MinesweeperBoard


@dataclass
class Constraint:
    """Represents a linear constraint in the CSP"""
    variables: List[Tuple[int, int]]  # Cell coordinates
    coefficients: List[int]           # Coefficients (usually all 1s)
    rhs: int                         # Right-hand side (mines needed)
    source_cell: Tuple[int, int]     # Cell that generated this constraint

    def __post_init__(self):
        assert len(self.variables) == len(self.coefficients)

    def evaluate(self, assignment: Dict[Tuple[int, int], int]) -> Optional[int]:
        """Evaluate constraint with partial assignment"""
        total = 0
        unknown_vars = 0

        for var, coeff in zip(self.variables, self.coefficients):
            if var in assignment:
                total += coeff * assignment[var]
            else:
                unknown_vars += 1

        if unknown_vars == 0:
            return total - self.rhs  # Should be 0 for satisfied constraint
        return None

    def is_satisfied(self, assignment: Dict[Tuple[int, int], int]) -> bool:
        """Check if constraint is satisfied by assignment"""
        result = self.evaluate(assignment)
        return result is not None and result == 0

    def get_unassigned_variables(self, assignment: Dict[Tuple[int, int], int]) -> List[Tuple[int, int]]:
        """Get variables not yet assigned"""
        return [var for var in self.variables if var not in assignment]


class CSPSolver:
    """
    Advanced CSP solver implementing multiple algorithms for minesweeper solving

    Features:
    - Deterministic Solution Search (DSS) with dependency separation
    - Constraint propagation and arc consistency
    - Gaussian elimination for linear constraint systems
    - Intelligent backtracking with pruning
    """

    def __init__(self, board: MinesweeperBoard):
        self.board = board
        self.logger = logging.getLogger(__name__)

        # CSP state
        self.variables: Set[Tuple[int, int]] = set()
        self.constraints: List[Constraint] = []
        self.domains: Dict[Tuple[int, int], Set[int]] = {}

        # Solution tracking
        self.solutions: List[Dict[Tuple[int, int], int]] = []
        self.max_solutions = 10000  # Limit to prevent memory issues

    def setup_csp(self) -> None:
        """Setup CSP from current board state"""
        self.variables.clear()
        self.constraints.clear()
        self.domains.clear()

        # Get constraints from board
        board_constraints = self.board.get_constraints()

        for constraint_data in board_constraints:
            neighbors = constraint_data['neighbors']
            mines_needed = constraint_data['mines_needed']
            source_cell = constraint_data['cell']

            if neighbors and mines_needed >= 0:
                # Add variables
                for var in neighbors:
                    self.variables.add(var)
                    if var not in self.domains:
                        self.domains[var] = {0, 1}  # 0 = safe, 1 = mine

                # Create constraint
                constraint = Constraint(
                    variables=neighbors,
                    coefficients=[1] * len(neighbors),
                    rhs=mines_needed,
                    source_cell=source_cell
                )
                self.constraints.append(constraint)

        self.logger.info(
            f"CSP setup: {len(self.variables)} variables, {len(self.constraints)} constraints")

    def propagate_constraints(self) -> bool:
        """
        Apply constraint propagation to reduce domains

        Returns:
            False if inconsistency detected, True otherwise
        """
        changed = True
        while changed:
            changed = False

            for constraint in self.constraints:
                # Check if constraint can be simplified
                min_possible = 0
                max_possible = 0
                unassigned = []

                for var in constraint.variables:
                    if len(self.domains[var]) == 1:
                        # Variable is assigned
                        value = next(iter(self.domains[var]))
                        min_possible += value
                        max_possible += value
                    else:
                        # Variable is unassigned
                        unassigned.append(var)
                        min_possible += 0  # Minimum value
                        max_possible += 1  # Maximum value

                # Check for inconsistency
                if constraint.rhs < min_possible or constraint.rhs > max_possible:
                    return False

                # Apply constraint propagation
                if constraint.rhs == min_possible:
                    # All unassigned variables must be 0 (safe)
                    for var in unassigned:
                        if 1 in self.domains[var]:
                            self.domains[var].remove(1)
                            changed = True

                elif constraint.rhs == max_possible:
                    # All unassigned variables must be 1 (mine)
                    for var in unassigned:
                        if 0 in self.domains[var]:
                            self.domains[var].remove(0)
                            changed = True

                # Check for empty domains
                for var in self.variables:
                    if not self.domains[var]:
                        return False

        return True

    def solve_deterministic(self) -> Dict[str, Set[Tuple[int, int]]]:
        """
        Find deterministic solutions using constraint propagation

        Returns:
            Dictionary with 'safe' and 'mines' sets
        """
        self.setup_csp()

        if not self.propagate_constraints():
            self.logger.warning(
                "Constraint propagation detected inconsistency")
            return {'safe': set(), 'mines': set()}

        safe_cells = set()
        mine_cells = set()

        # Extract determined variables
        for var in self.variables:
            if len(self.domains[var]) == 1:
                value = next(iter(self.domains[var]))
                if value == 0:
                    safe_cells.add(var)
                else:
                    mine_cells.add(var)

        self.logger.info(
            f"Deterministic solution: {len(safe_cells)} safe, {len(mine_cells)} mines")
        return {'safe': safe_cells, 'mines': mine_cells}

    def enumerate_solutions(self, max_solutions: Optional[int] = None) -> List[Dict[Tuple[int, int], int]]:
        """
        Enumerate all valid solutions using backtracking

        Args:
            max_solutions: Maximum number of solutions to find

        Returns:
            List of solution dictionaries
        """
        if max_solutions is None:
            max_solutions = self.max_solutions

        self.setup_csp()

        if not self.propagate_constraints():
            return []

        self.solutions.clear()

        # Get unassigned variables
        unassigned = [var for var in self.variables if len(
            self.domains[var]) > 1]

        if not unassigned:
            # All variables are assigned
            solution = {}
            for var in self.variables:
                solution[var] = next(iter(self.domains[var]))
            return [solution]

        # Start backtracking
        assignment = {}
        for var in self.variables:
            if len(self.domains[var]) == 1:
                assignment[var] = next(iter(self.domains[var]))

        self._backtrack(assignment, unassigned, max_solutions)

        self.logger.info(f"Found {len(self.solutions)} solutions")
        return self.solutions

    def _backtrack(self, assignment: Dict[Tuple[int, int], int],
                   unassigned: List[Tuple[int, int]], max_solutions: int) -> bool:
        """Recursive backtracking search"""
        if len(self.solutions) >= max_solutions:
            return True

        if not unassigned:
            # Complete assignment found
            if self._is_valid_assignment(assignment):
                self.solutions.append(assignment.copy())
            return False

        # Choose next variable (use MRV heuristic)
        var = self._choose_variable(unassigned, assignment)
        unassigned.remove(var)

        # Try each value in domain
        for value in sorted(self.domains[var]):
            assignment[var] = value

            if self._is_consistent(assignment, var):
                if self._backtrack(assignment, unassigned, max_solutions):
                    return True

            del assignment[var]

        unassigned.append(var)
        return False

    def _choose_variable(self, unassigned: List[Tuple[int, int]],
                         assignment: Dict[Tuple[int, int], int]) -> Tuple[int, int]:
        """Choose next variable using Minimum Remaining Values heuristic"""
        def remaining_values(var):
            count = 0
            for value in self.domains[var]:
                test_assignment = assignment.copy()
                test_assignment[var] = value
                if self._is_consistent(test_assignment, var):
                    count += 1
            return count

        return min(unassigned, key=remaining_values)

    def _is_consistent(self, assignment: Dict[Tuple[int, int], int],
                       var: Tuple[int, int]) -> bool:
        """Check if assignment is consistent with constraints"""
        for constraint in self.constraints:
            if var in constraint.variables:
                # Check this constraint
                total = 0
                unassigned_count = 0

                for v in constraint.variables:
                    if v in assignment:
                        total += assignment[v]
                    else:
                        unassigned_count += 1

                # Check bounds
                if unassigned_count == 0:
                    if total != constraint.rhs:
                        return False
                else:
                    min_possible = total
                    max_possible = total + unassigned_count
                    if constraint.rhs < min_possible or constraint.rhs > max_possible:
                        return False

        return True

    def _is_valid_assignment(self, assignment: Dict[Tuple[int, int], int]) -> bool:
        """Check if complete assignment satisfies all constraints"""
        for constraint in self.constraints:
            if not constraint.is_satisfied(assignment):
                return False
        return True

    def gaussian_elimination(self) -> Dict[str, Set[Tuple[int, int]]]:
        """
        Solve linear system using Gaussian elimination

        This method converts constraints to matrix form and applies
        Gaussian elimination to find solutions.
        """
        self.setup_csp()

        if not self.variables or not self.constraints:
            return {'safe': set(), 'mines': set()}

        # Create variable mapping
        var_list = sorted(list(self.variables))
        var_to_idx = {var: i for i, var in enumerate(var_list)}

        # Build constraint matrix
        A = []
        b = []

        for constraint in self.constraints:
            row = [0] * len(var_list)
            for var, coeff in zip(constraint.variables, constraint.coefficients):
                if var in var_to_idx:
                    row[var_to_idx[var]] = coeff
            A.append(row)
            b.append(constraint.rhs)

        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)

        if A.size == 0:
            return {'safe': set(), 'mines': set()}

        try:
            # Augmented matrix
            augmented = np.column_stack([A, b])

            # Forward elimination
            rows, cols = augmented.shape
            rank = 0

            for col in range(min(rows, cols - 1)):
                # Find pivot
                pivot_row = rank
                for row in range(rank + 1, rows):
                    if abs(augmented[row, col]) > abs(augmented[pivot_row, col]):
                        pivot_row = row

                if abs(augmented[pivot_row, col]) < 1e-10:
                    continue

                # Swap rows
                if pivot_row != rank:
                    augmented[[rank, pivot_row]] = augmented[[pivot_row, rank]]

                # Eliminate
                for row in range(rows):
                    if row != rank and abs(augmented[row, col]) > 1e-10:
                        factor = augmented[row, col] / augmented[rank, col]
                        augmented[row] -= factor * augmented[rank]

                rank += 1

            # Extract solutions
            safe_cells = set()
            mine_cells = set()

            for i in range(rank):
                # Find leading variable
                leading_var = None
                for j in range(len(var_list)):
                    if abs(augmented[i, j]) > 1e-10:
                        leading_var = j
                        break

                if leading_var is not None:
                    # Check if this variable is determined
                    non_zero_count = sum(1 for j in range(len(var_list))
                                         if abs(augmented[i, j]) > 1e-10)

                    if non_zero_count == 1:
                        # Variable is determined
                        value = augmented[i, -1] / augmented[i, leading_var]
                        if abs(value) < 1e-10:
                            safe_cells.add(var_list[leading_var])
                        elif abs(value - 1) < 1e-10:
                            mine_cells.add(var_list[leading_var])

            self.logger.info(
                f"Gaussian elimination: {len(safe_cells)} safe, {len(mine_cells)} mines")
            return {'safe': safe_cells, 'mines': mine_cells}

        except Exception as e:
            self.logger.error(f"Gaussian elimination failed: {e}")
            return {'safe': set(), 'mines': set()}

    def solve(self) -> Dict[str, Any]:
        """
        Main solving method combining multiple approaches

        Returns:
            Dictionary with solving results
        """
        results = {
            'safe_moves': set(),
            'mine_moves': set(),
            'solutions': [],
            'method_used': 'none'
        }

        try:
            # Try deterministic solving first
            deterministic = self.solve_deterministic()
            if deterministic['safe'] or deterministic['mines']:
                results.update(deterministic)
                results['method_used'] = 'deterministic'
                return results

            # Try Gaussian elimination
            gaussian = self.gaussian_elimination()
            if gaussian['safe'] or gaussian['mines']:
                results.update(gaussian)
                results['method_used'] = 'gaussian'
                return results

            # Enumerate solutions for probability analysis
            solutions = self.enumerate_solutions(1000)
            if solutions:
                results['solutions'] = solutions
                results['method_used'] = 'enumeration'

        except Exception as e:
            self.logger.error(f"CSP solving failed: {e}")

        return results

