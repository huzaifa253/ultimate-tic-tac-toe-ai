"""
minimax_ai.py
Genuine Minimax search with Alpha-Beta Pruning, Move Ordering, and a multi-factor
Ultimate Tic-Tac-Toe Heuristic Evaluation Function.
"""

import math
import time
from typing import Dict, List, Optional, Tuple

from game_logic import (
    DRAW,
    EMPTY,
    PLAYER_O,
    PLAYER_X,
    WIN_LINES,
    UltimateTicTacToe,
)

# Terminal score dominates every possible non-terminal heuristic score
TERMINAL_SCORE: float = 100_000.0

# Strategic positional weights for boards and cells (center > corners > edges)
POSITION_WEIGHTS: Tuple[float, ...] = (
    3.0, 2.0, 3.0,
    2.0, 4.0, 2.0,
    3.0, 2.0, 3.0,
)

DIFFICULTY_DEPTHS: Dict[str, int] = {
    "Easy": 2,
    "Medium": 3,
    "Hard": 4,
}


class MinimaxAI:
    """
    Minimax AI agent with Alpha-Beta pruning and move ordering for Ultimate Tic-Tac-Toe.
    """

    def __init__(
        self,
        difficulty: str = "Hard",
        custom_depth: Optional[int] = None,
    ) -> None:
        if difficulty not in DIFFICULTY_DEPTHS:
            raise ValueError(f"Invalid difficulty {difficulty!r}. Choose from {list(DIFFICULTY_DEPTHS)}")
        self.difficulty: str = difficulty
        self.max_depth: int = custom_depth if custom_depth is not None else DIFFICULTY_DEPTHS[difficulty]
        self.nodes_evaluated: int = 0
        self.cutoffs: int = 0
        self.last_stats: Dict[str, object] = {
            "algorithm": "Minimax",
            "difficulty": self.difficulty,
            "search_depth": self.max_depth,
            "nodes_evaluated": 0,
            "cutoffs": 0,
            "thinking_time": 0.0,
            "chosen_move": None,
            "eval_score": 0.0,
        }

    def get_config_summary(self) -> str:
        """Return a human-readable summary of the search configuration."""
        return f"Minimax ({self.difficulty}, depth={self.max_depth}, alpha-beta=True)"

    def get_move(self, state: UltimateTicTacToe) -> Optional[Tuple[int, int]]:
        """
        Compute and return the best legal move for state.current_player using
        Alpha-Beta Minimax search up to self.max_depth.
        Does not mutate the passed state.
        """
        if state.is_terminal():
            return None

        legal_moves = state.get_legal_moves()
        if not legal_moves:
            return None

        start_time = time.perf_counter()
        self.nodes_evaluated = 0
        self.cutoffs = 0
        root_player = state.current_player

        # Adaptive depth boost in late-game when branching factor is tiny
        effective_depth = self.max_depth
        if self.difficulty == "Hard" and len(legal_moves) <= 6 and state.move_count >= 25:
            effective_depth = self.max_depth + 1

        ordered_moves = self._order_moves(state, legal_moves, root_player)

        best_score = -math.inf
        best_move: Tuple[int, int] = ordered_moves[0]
        alpha = -math.inf
        beta = math.inf

        for move in ordered_moves:
            child_state = state.copy()
            child_state.apply_move(move)
            score = self._alphabeta(
                state=child_state,
                depth=effective_depth - 1,
                alpha=alpha,
                beta=beta,
                maximizing=False,
                root_player=root_player,
            )
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score

        elapsed = time.perf_counter() - start_time
        self.last_stats = {
            "algorithm": "Minimax",
            "difficulty": self.difficulty,
            "search_depth": effective_depth,
            "nodes_evaluated": self.nodes_evaluated,
            "cutoffs": self.cutoffs,
            "thinking_time": elapsed,
            "chosen_move": best_move,
            "eval_score": round(best_score, 2),
        }
        return best_move

    def _alphabeta(
        self,
        state: UltimateTicTacToe,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        root_player: str,
    ) -> float:
        """Recursive Minimax with Alpha-Beta Pruning."""
        self.nodes_evaluated += 1

        if state.is_terminal():
            return self._score_terminal(state, root_player, depth)

        if depth <= 0:
            return self.evaluate(state, root_player)

        legal_moves = state.get_legal_moves()
        ordered_moves = self._order_moves(state, legal_moves, state.current_player)

        if maximizing:
            value = -math.inf
            for move in ordered_moves:
                child = state.copy()
                child.apply_move(move)
                value = max(
                    value,
                    self._alphabeta(child, depth - 1, alpha, beta, False, root_player),
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    self.cutoffs += 1
                    break
            return value
        else:
            value = math.inf
            for move in ordered_moves:
                child = state.copy()
                child.apply_move(move)
                value = min(
                    value,
                    self._alphabeta(child, depth - 1, alpha, beta, True, root_player),
                )
                beta = min(beta, value)
                if alpha >= beta:
                    self.cutoffs += 1
                    break
            return value

    def search_comparison(
        self,
        state: UltimateTicTacToe,
        depth: int,
        use_alpha_beta: bool = True,
    ) -> Tuple[float, Optional[Tuple[int, int]], int, int]:
        """
        Helper method used by unit tests to verify that Alpha-Beta pruning returns
        the exact same minimax value as plain unpruned Minimax while evaluating fewer nodes.
        """
        self.nodes_evaluated = 0
        self.cutoffs = 0
        root_player = state.current_player
        legal_moves = self._order_moves(state, state.get_legal_moves(), root_player)
        if not legal_moves:
            return 0.0, None, 0, 0

        best_score = -math.inf
        best_move = legal_moves[0]
        alpha = -math.inf
        beta = math.inf

        for move in legal_moves:
            child = state.copy()
            child.apply_move(move)
            if use_alpha_beta:
                score = self._alphabeta(child, depth - 1, alpha, beta, False, root_player)
                alpha = max(alpha, score)
            else:
                score = self._plain_minimax(child, depth - 1, False, root_player)
            if score > best_score:
                best_score = score
                best_move = move

        return best_score, best_move, self.nodes_evaluated, self.cutoffs

    def _plain_minimax(
        self,
        state: UltimateTicTacToe,
        depth: int,
        maximizing: bool,
        root_player: str,
    ) -> float:
        """Unpruned recursive Minimax for testing equivalence against Alpha-Beta."""
        self.nodes_evaluated += 1
        if state.is_terminal():
            return self._score_terminal(state, root_player, depth)
        if depth <= 0:
            return self.evaluate(state, root_player)

        legal_moves = self._order_moves(state, state.get_legal_moves(), state.current_player)
        if maximizing:
            val = -math.inf
            for m in legal_moves:
                child = state.copy()
                child.apply_move(m)
                val = max(val, self._plain_minimax(child, depth - 1, False, root_player))
            return val
        else:
            val = math.inf
            for m in legal_moves:
                child = state.copy()
                child.apply_move(m)
                val = min(val, self._plain_minimax(child, depth - 1, True, root_player))
            return val

    @staticmethod
    def _score_terminal(state: UltimateTicTacToe, root_player: str, depth: int) -> float:
        """Score terminal game states, preferring faster wins and slower losses."""
        winner = state.get_winner()
        if winner == root_player:
            return TERMINAL_SCORE + depth * 100.0
        if winner == DRAW or winner is None:
            return 0.0
        return -TERMINAL_SCORE - depth * 100.0

    def _order_moves(
        self,
        state: UltimateTicTacToe,
        moves: List[Tuple[int, int]],
        player: str,
    ) -> List[Tuple[int, int]]:
        """
        Order candidate moves to maximize early alpha-beta cutoffs:
        1. Moves that immediately win or block a small board.
        2. Moves that avoid giving the opponent a free choice (completed target board).
        3. High-weight cells and boards (center > corner > edge).
        """
        opponent = UltimateTicTacToe.get_opponent(player)

        def move_priority(move: Tuple[int, int]) -> float:
            b_idx, c_idx = move
            board = state.small_boards[b_idx]
            priority = POSITION_WEIGHTS[b_idx] * 2.0 + POSITION_WEIGHTS[c_idx]

            # Check if this move completes a 3-in-a-row for player or blocks opponent
            for a, b, c in WIN_LINES:
                if c_idx in (a, b, c):
                    line_vals = [board[a], board[b], board[c]]
                    if line_vals.count(player) == 2 and line_vals.count(EMPTY) == 1:
                        priority += 50.0
                    elif line_vals.count(opponent) == 2 and line_vals.count(EMPTY) == 1:
                        priority += 40.0

            # Penalty for sending opponent to an already finished board (giving free choice)
            if state.macro_board[c_idx] != EMPTY and c_idx != b_idx:
                priority -= 25.0

            return priority

        return sorted(moves, key=move_priority, reverse=True)

    def evaluate(self, state: UltimateTicTacToe, root_player: str) -> float:
        """
        Heuristic evaluation function from the perspective of root_player.
        Considers:
        - Macro-board won boards weighted by strategic importance.
        - Macro-board two-in-a-row threats and open winning lines.
        - Small-board cell control, two-in-a-row threats, and open lines inside unfinished boards.
        - Routing/tempo advantage: whether the active turn has free choice or is forced into
          a favorable/unfavorable small board.
        """
        if state.is_terminal():
            return self._score_terminal(state, root_player, 0)

        opponent = UltimateTicTacToe.get_opponent(root_player)
        score = 0.0

        # 1. Macro-board ownership & small-board internal structure
        small_evals: List[float] = [0.0] * 9
        for b_idx in range(9):
            macro_status = state.macro_board[b_idx]
            weight = POSITION_WEIGHTS[b_idx]
            if macro_status == root_player:
                score += 120.0 * weight
                small_evals[b_idx] = 1.0
            elif macro_status == opponent:
                score -= 120.0 * weight
                small_evals[b_idx] = -1.0
            elif macro_status == DRAW:
                small_evals[b_idx] = 0.0
            else:
                local_val = self._evaluate_small_board(
                    state.small_boards[b_idx], root_player, opponent
                )
                score += local_val * (weight * 0.5)
                small_evals[b_idx] = max(-0.8, min(0.8, local_val / 45.0))

        # 2. Macro-board winning lines and two-in-a-row threats
        for a, b, c in WIN_LINES:
            line = [state.macro_board[a], state.macro_board[b], state.macro_board[c]]
            if DRAW in line:
                continue
            root_count = line.count(root_player)
            opp_count = line.count(opponent)

            if root_count > 0 and opp_count == 0:
                if root_count == 2:
                    score += 320.0
                elif root_count == 1:
                    score += 65.0
            elif opp_count > 0 and root_count == 0:
                if opp_count == 2:
                    score -= 320.0
                elif opp_count == 1:
                    score -= 65.0

            # Fractional line synergy from unfinished boards
            if root_count == 0 or opp_count == 0:
                synergy = small_evals[a] + small_evals[b] + small_evals[c]
                score += synergy * 30.0

        # 3. Forced-board routing and free-choice tempo evaluation
        turn_sign = 1.0 if state.current_player == root_player else -1.0
        if state.active_board is None:
            # Having free board selection is a strong tactical advantage
            score += turn_sign * 45.0
        else:
            # Being sent into a board where one player already dominates
            target_bias = small_evals[state.active_board]
            score += target_bias * 25.0 + turn_sign * 10.0

        return score

    @staticmethod
    def _evaluate_small_board(
        cells: List[str], root_player: str, opponent: str
    ) -> float:
        """Evaluate an unfinished 3x3 small board from root_player's perspective."""
        local_score = 0.0

        # Positional cell control
        for idx, cell in enumerate(cells):
            if cell == root_player:
                local_score += POSITION_WEIGHTS[idx] * 1.5
            elif cell == opponent:
                local_score -= POSITION_WEIGHTS[idx] * 1.5

        # Two-in-a-row threats and open lines
        for a, b, c in WIN_LINES:
            line = [cells[a], cells[b], cells[c]]
            r_cnt = line.count(root_player)
            o_cnt = line.count(opponent)
            if r_cnt == 2 and o_cnt == 0:
                local_score += 18.0
            elif o_cnt == 2 and r_cnt == 0:
                local_score -= 18.0
            elif r_cnt == 1 and o_cnt == 0:
                local_score += 3.5
            elif o_cnt == 1 and r_cnt == 0:
                local_score -= 3.5

        return local_score