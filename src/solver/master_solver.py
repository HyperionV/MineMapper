"""
Master Solver - Multi-Strategy Coordination for Minesweeper

This module coordinates multiple solving strategies to achieve optimal
performance across different board configurations and game states.
"""

from typing import Dict, List, Tuple, Set, Optional, Any, Callable
from enum import Enum
import logging
import time
from dataclasses import dataclass

from .game_state import MinesweeperBoard
from .csp_solver import CSPSolver
from .probability_engine import ProbabilityEngine
from .pattern_recognition import PatternMatcher


class SolverStrategy(Enum):
    """Available solving strategies"""
    DETERMINISTIC = "deterministic"
    CSP = "csp"
    PROBABILITY = "probability"
    PATTERN = "pattern"
    HYBRID = "hybrid"


@dataclass
class Move:
    """Represents a suggested move"""
    row: int
    col: int
    action: str  # 'reveal' or 'flag'
    confidence: float
    reasoning: str
    strategy: str
    metadata: Dict[str, Any] = None


@dataclass
class SolverResult:
    """Complete solver analysis result"""
    moves: List[Move]
    statistics: Dict[str, Any]
    analysis_time: float
    strategies_used: List[str]
    board_state: Dict[str, Any]


class MasterSolver:
    """
    Master solver that coordinates multiple solving strategies

    Features:
    - Multi-strategy approach with intelligent fallback
    - Performance monitoring and strategy selection
    - Comprehensive move analysis and ranking
    - Real-time progress tracking
    """

    def __init__(self, board: MinesweeperBoard):
        self.board = board
        self.logger = logging.getLogger(__name__)

        # Initialize sub-solvers
        self.csp_solver = CSPSolver(board)
        self.probability_engine = ProbabilityEngine(board)
        self.pattern_matcher = PatternMatcher(board)

        # Solver configuration
        self.strategy_priority = [
            SolverStrategy.PATTERN,
            SolverStrategy.DETERMINISTIC,
            SolverStrategy.CSP,
            SolverStrategy.PROBABILITY
        ]

        # Performance tracking
        self.solver_stats = {
            strategy.value: {
                'calls': 0,
                'successes': 0,
                'total_time': 0.0,
                'moves_found': 0
            } for strategy in SolverStrategy
        }

        # Analysis callbacks for UI updates
        self.progress_callbacks: List[Callable[[str, float], None]] = []

    def add_progress_callback(self, callback: Callable[[str, float], None]) -> None:
        """Add callback for progress updates"""
        self.progress_callbacks.append(callback)

    def _notify_progress(self, message: str, progress: float) -> None:
        """Notify all progress callbacks"""
        for callback in self.progress_callbacks:
            try:
                callback(message, progress)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")

    def analyze(self, strategy: SolverStrategy = SolverStrategy.HYBRID) -> SolverResult:
        """
        Perform comprehensive board analysis

        Args:
            strategy: Primary solving strategy to use

        Returns:
            Complete solver result with moves and analysis
        """
        start_time = time.time()
        self._notify_progress("Starting analysis...", 0.0)

        all_moves = []
        strategies_used = []

        try:
            if strategy == SolverStrategy.HYBRID:
                # Use hybrid approach with all strategies
                all_moves, strategies_used = self._hybrid_solve()
            else:
                # Use single strategy
                moves = self._apply_strategy(strategy)
                if moves:
                    all_moves.extend(moves)
                    strategies_used.append(strategy.value)

            # Rank and filter moves
            final_moves = self._rank_moves(all_moves)

            # Compile statistics
            analysis_time = time.time() - start_time
            statistics = self._compile_statistics(
                strategies_used, analysis_time)
            board_state = self._get_board_state()

            self._notify_progress("Analysis complete", 1.0)

            return SolverResult(
                moves=final_moves,
                statistics=statistics,
                analysis_time=analysis_time,
                strategies_used=strategies_used,
                board_state=board_state
            )

        except Exception as e:
            self.logger.error(f"Solver analysis failed: {e}")
            return SolverResult(
                moves=[],
                statistics={'error': str(e)},
                analysis_time=time.time() - start_time,
                strategies_used=[],
                board_state={}
            )

    def _hybrid_solve(self) -> Tuple[List[Move], List[str]]:
        """Apply hybrid solving strategy using multiple approaches"""
        all_moves = []
        strategies_used = []

        # Phase 1: Pattern Recognition (fast, high confidence)
        self._notify_progress("Applying pattern recognition...", 0.1)
        pattern_moves = self._apply_strategy(SolverStrategy.PATTERN)
        if pattern_moves:
            all_moves.extend(pattern_moves)
            strategies_used.append(SolverStrategy.PATTERN.value)

        # Phase 2: Deterministic CSP solving
        self._notify_progress("Running deterministic analysis...", 0.3)
        csp_moves = self._apply_strategy(SolverStrategy.DETERMINISTIC)
        if csp_moves:
            all_moves.extend(csp_moves)
            strategies_used.append(SolverStrategy.DETERMINISTIC.value)

        # Phase 3: Advanced CSP if deterministic failed
        if not csp_moves:
            self._notify_progress("Running advanced CSP analysis...", 0.5)
            advanced_csp = self._apply_strategy(SolverStrategy.CSP)
            if advanced_csp:
                all_moves.extend(advanced_csp)
                strategies_used.append(SolverStrategy.CSP.value)

        # Phase 4: Probability analysis for uncertain situations
        self._notify_progress("Calculating probabilities...", 0.7)
        prob_moves = self._apply_strategy(SolverStrategy.PROBABILITY)
        if prob_moves:
            all_moves.extend(prob_moves)
            strategies_used.append(SolverStrategy.PROBABILITY.value)

        self._notify_progress("Finalizing analysis...", 0.9)
        return all_moves, strategies_used

    def _apply_strategy(self, strategy: SolverStrategy) -> List[Move]:
        """Apply a specific solving strategy"""
        strategy_start = time.time()
        self.solver_stats[strategy.value]['calls'] += 1

        try:
            if strategy == SolverStrategy.PATTERN:
                return self._pattern_solve()
            elif strategy == SolverStrategy.DETERMINISTIC:
                return self._deterministic_solve()
            elif strategy == SolverStrategy.CSP:
                return self._csp_solve()
            elif strategy == SolverStrategy.PROBABILITY:
                return self._probability_solve()
            else:
                return []

        except Exception as e:
            self.logger.error(f"Strategy {strategy.value} failed: {e}")
            return []

        finally:
            strategy_time = time.time() - strategy_start
            self.solver_stats[strategy.value]['total_time'] += strategy_time

    def _pattern_solve(self) -> List[Move]:
        """Apply pattern recognition strategy"""
        results = self.pattern_matcher.apply_patterns()
        moves = []

        # Convert pattern results to moves
        for row, col in results.get('safe', []):
            moves.append(Move(
                row=row, col=col, action='reveal',
                confidence=0.95, reasoning="Pattern recognition - safe cell",
                strategy="pattern"
            ))

        for row, col in results.get('mines', []):
            moves.append(Move(
                row=row, col=col, action='flag',
                confidence=0.95, reasoning="Pattern recognition - mine cell",
                strategy="pattern"
            ))

        if moves:
            self.solver_stats['pattern']['successes'] += 1
            self.solver_stats['pattern']['moves_found'] += len(moves)

        return moves

    def _deterministic_solve(self) -> List[Move]:
        """Apply deterministic CSP solving"""
        results = self.csp_solver.solve_deterministic()
        moves = []

        for row, col in results.get('safe', []):
            moves.append(Move(
                row=row, col=col, action='reveal',
                confidence=1.0, reasoning="Deterministic solution - guaranteed safe",
                strategy="deterministic"
            ))

        for row, col in results.get('mines', []):
            moves.append(Move(
                row=row, col=col, action='flag',
                confidence=1.0, reasoning="Deterministic solution - guaranteed mine",
                strategy="deterministic"
            ))

        if moves:
            self.solver_stats['deterministic']['successes'] += 1
            self.solver_stats['deterministic']['moves_found'] += len(moves)

        return moves

    def _csp_solve(self) -> List[Move]:
        """Apply advanced CSP solving with enumeration"""
        csp_results = self.csp_solver.solve()
        moves = []

        # Extract deterministic moves from CSP solutions
        if csp_results.get('solutions'):
            # Analyze solutions to find guaranteed moves
            solutions = csp_results['solutions']
            if len(solutions) > 1:
                # Find variables that have the same value in all solutions
                consistent_vars = {}
                for var in solutions[0].keys():
                    values = set(sol[var] for sol in solutions)
                    if len(values) == 1:
                        consistent_vars[var] = values.pop()

                # Convert to moves
                for (row, col), value in consistent_vars.items():
                    if value == 0:
                        moves.append(Move(
                            row=row, col=col, action='reveal',
                            confidence=1.0, reasoning="CSP enumeration - consistent safe",
                            strategy="csp"
                        ))
                    else:
                        moves.append(Move(
                            row=row, col=col, action='flag',
                            confidence=1.0, reasoning="CSP enumeration - consistent mine",
                            strategy="csp"
                        ))

        if moves:
            self.solver_stats['csp']['successes'] += 1
            self.solver_stats['csp']['moves_found'] += len(moves)

        return moves

    def _probability_solve(self) -> List[Move]:
        """Apply probability-based solving"""
        prob_results = self.probability_engine.analyze_position()
        moves = []

        # Get best moves from probability analysis
        best_moves = prob_results.get('best_moves', {})

        # Add safe moves
        safe_move = best_moves.get('safe')
        if safe_move:
            cell, prob, description = safe_move
            moves.append(Move(
                row=cell[0], col=cell[1], action='reveal',
                confidence=1.0 - prob, reasoning=description,
                strategy="probability"
            ))

        # Add information moves if no safe moves
        if not moves:
            info_move = best_moves.get('information')
            if info_move:
                cell, info_gain, description = info_move
                prob = self.board.get_cell(*cell).probability
                moves.append(Move(
                    row=cell[0], col=cell[1], action='reveal',
                    confidence=1.0 - prob, reasoning=description,
                    strategy="probability"
                ))

        # Add balanced move as fallback
        if not moves:
            balanced_move = best_moves.get('balanced')
            if balanced_move:
                cell, score, description = balanced_move
                prob = self.board.get_cell(*cell).probability
                moves.append(Move(
                    row=cell[0], col=cell[1], action='reveal',
                    confidence=1.0 - prob, reasoning=description,
                    strategy="probability"
                ))

        if moves:
            self.solver_stats['probability']['successes'] += 1
            self.solver_stats['probability']['moves_found'] += len(moves)

        return moves

    def _rank_moves(self, moves: List[Move]) -> List[Move]:
        """Rank and filter moves by quality"""
        if not moves:
            return []

        # Remove duplicates while preserving best confidence
        move_dict = {}
        for move in moves:
            key = (move.row, move.col, move.action)
            if key not in move_dict or move.confidence > move_dict[key].confidence:
                move_dict[key] = move

        unique_moves = list(move_dict.values())

        # Sort by confidence (descending) and then by strategy priority
        strategy_weights = {
            'deterministic': 100,
            'pattern': 90,
            'csp': 80,
            'probability': 70
        }

        def move_score(move):
            return (move.confidence, strategy_weights.get(move.strategy, 0))

        unique_moves.sort(key=move_score, reverse=True)

        # Limit to top moves
        return unique_moves[:10]

    def _compile_statistics(self, strategies_used: List[str], analysis_time: float) -> Dict[str, Any]:
        """Compile comprehensive statistics"""
        board_stats = self.board.get_game_statistics()
        pattern_stats = self.pattern_matcher.get_pattern_statistics()

        return {
            'analysis_time': analysis_time,
            'strategies_used': strategies_used,
            'board_statistics': board_stats,
            'pattern_statistics': pattern_stats,
            'solver_performance': self.solver_stats.copy(),
            'total_moves_generated': sum(
                stats['moves_found'] for stats in self.solver_stats.values()
            )
        }

    def _get_board_state(self) -> Dict[str, Any]:
        """Get current board state summary"""
        return {
            'dimensions': (self.board.rows, self.board.cols),
            'total_mines': self.board.total_mines,
            'remaining_mines': self.board.remaining_mines,
            'revealed_cells': self.board.revealed_count,
            'flagged_cells': self.board.flagged_count,
            'frontier_size': len(self.board.frontier_cells)
        }

    def get_next_move(self) -> Optional[Move]:
        """Get the single best next move"""
        result = self.analyze()
        return result.moves[0] if result.moves else None

    def get_solver_performance(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        performance = {}

        for strategy, stats in self.solver_stats.items():
            if stats['calls'] > 0:
                performance[strategy] = {
                    'success_rate': stats['successes'] / stats['calls'],
                    'average_time': stats['total_time'] / stats['calls'],
                    'moves_per_success': stats['moves_found'] / max(1, stats['successes']),
                    **stats
                }

        return performance

