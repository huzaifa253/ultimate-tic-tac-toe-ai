"""
mcts_ai.py
Genuine Monte Carlo Tree Search (MCTS) implementation featuring Selection (UCT),
Expansion, Simulation (Rollout), and Backpropagation with player-perspective reward tracking.
"""

import math
import random
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

DIFFICULTY_SIMULATIONS: Dict[str, int] = {
    "Easy": 150,
    "Medium": 450,
    "Hard": 1000,
}


class MCTSNode:
    """
    A node in the Monte Carlo Search Tree.
    Reward perspective:
    - self.player_just_moved is the player who made `self.move` to reach this node.
    - self.wins accumulates rewards from the perspective of `self.player_just_moved`
      (1.0 for a win by `player_just_moved`, 0.5 for a draw, 0.0 for a loss).
    - During UCT selection at a parent node, the player to move next is the parent's
      `state.current_player`, which equals each child's `player_just_moved`. Therefore,
      maximizing child.wins / child.visits always selects moves favored by the player whose
      turn it is at the parent node.
    """

    def __init__(
        self,
        state: UltimateTicTacToe,
        parent: Optional["MCTSNode"] = None,
        move: Optional[Tuple[int, int]] = None,
        player_just_moved: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.state: UltimateTicTacToe = state
        self.parent: Optional["MCTSNode"] = parent
        self.move: Optional[Tuple[int, int]] = move
        self.player_just_moved: Optional[str] = player_just_moved
        self.children: List["MCTSNode"] = []
        self.visits: int = 0
        self.wins: float = 0.0

        legal = state.get_legal_moves()
        if rng is not None:
            rng.shuffle(legal)
        self.untried_moves: List[Tuple[int, int]] = legal

    def is_fully_expanded(self) -> bool:
        """Return True if all legal moves from this node have been expanded."""
        return len(self.untried_moves) == 0

    def is_terminal_node(self) -> bool:
        """Return True if the game state at this node is terminal."""
        return self.state.is_terminal()

    def uct_select_child(self, exploration_constant: float) -> "MCTSNode":
        """Select the child with the highest Upper Confidence Bound for Trees (UCT) value."""
        log_parent_visits = math.log(max(1, self.visits))
        best_child: Optional[MCTSNode] = None
        best_uct = -math.inf

        for child in self.children:
            if child.visits == 0:
                uct_val = math.inf
            else:
                exploitation = child.wins / child.visits
                exploration = exploration_constant * math.sqrt(log_parent_visits / child.visits)
                uct_val = exploitation + exploration

            if uct_val > best_uct:
                best_uct = uct_val
                best_child = child

        assert best_child is not None
        return best_child


class MCTSAI:
    """
    Monte Carlo Tree Search AI agent for Ultimate Tic-Tac-Toe.
    Implements Selection, Expansion, Simulation, and Backpropagation.
    """

    def __init__(
        self,
        difficulty: str = "Hard",
        custom_simulations: Optional[int] = None,
        exploration_constant: float = 1.41421356,
        seed: Optional[int] = None,
    ) -> None:
        if difficulty not in DIFFICULTY_SIMULATIONS:
            raise ValueError(
                f"Invalid difficulty {difficulty!r}. Choose from {list(DIFFICULTY_SIMULATIONS)}"
            )
        self.difficulty: str = difficulty
        self.simulation_budget: int = (
            custom_simulations
            if custom_simulations is not None
            else DIFFICULTY_SIMULATIONS[difficulty]
        )
        self.exploration_constant: float = exploration_constant
        self.seed: Optional[int] = seed
        self.rng: random.Random = random.Random(seed)
        self.last_stats: Dict[str, object] = {
            "algorithm": "MCTS",
            "difficulty": self.difficulty,
            "simulations": 0,
            "nodes_created": 0,
            "thinking_time": 0.0,
            "chosen_move": None,
            "root_move_win_rate": 0.0,
            "root_child_visits": {},
            "root_child_win_rates": {},
        }

    def get_config_summary(self) -> str:
        """Return a human-readable summary of the MCTS configuration."""
        return (
            f"MCTS ({self.difficulty}, simulations={self.simulation_budget}, "
            f"C={self.exploration_constant:.2f})"
        )

    def get_move(self, state: UltimateTicTacToe) -> Optional[Tuple[int, int]]:
        """
        Execute MCTS for `self.simulation_budget` iterations and return the selected move.
        Does not mutate the input `state`.
        """
        if state.is_terminal():
            return None

        legal_moves = state.get_legal_moves()
        if not legal_moves:
            return None

        start_time = time.perf_counter()
        root_player = state.current_player
        root_node = MCTSNode(
            state=state.copy(),
            parent=None,
            move=None,
            player_just_moved=UltimateTicTacToe.get_opponent(root_player),
            rng=self.rng,
        )
        nodes_created = 1

        for _ in range(self.simulation_budget):
            node = root_node

            # 1. SELECTION
            while not node.is_terminal_node() and node.is_fully_expanded():
                node = node.uct_select_child(self.exploration_constant)

            # 2. EXPANSION
            if not node.is_terminal_node() and not node.is_fully_expanded():
                move = node.untried_moves.pop()
                next_state = node.state.copy()
                moving_player = next_state.current_player
                next_state.apply_move(move)
                child_node = MCTSNode(
                    state=next_state,
                    parent=node,
                    move=move,
                    player_just_moved=moving_player,
                    rng=self.rng,
                )
                node.children.append(child_node)
                node = child_node
                nodes_created += 1

            # 3. SIMULATION (Rollout)
            winner = self._simulate_rollout(node.state)

            # 4. BACKPROPAGATION
            curr: Optional[MCTSNode] = node
            while curr is not None:
                curr.visits += 1
                if winner == DRAW:
                    curr.wins += 0.5
                elif winner == curr.player_just_moved:
                    curr.wins += 1.0
                curr = curr.parent

        # Robust child selection: choose the root child with the highest visit count
        # (breaking ties by highest win rate from the root player's perspective).
        best_child = max(
            root_node.children,
            key=lambda c: (c.visits, (c.wins / c.visits) if c.visits > 0 else 0.0),
        )
        chosen_move = best_child.move
        best_win_rate = (best_child.wins / best_child.visits) if best_child.visits > 0 else 0.0

        visit_map: Dict[Tuple[int, int], int] = {}
        win_rate_map: Dict[Tuple[int, int], float] = {}
        for child in root_node.children:
            if child.move is not None:
                visit_map[child.move] = child.visits
                win_rate_map[child.move] = (
                    round((child.wins / child.visits) * 100.0, 1) if child.visits > 0 else 0.0
                )

        elapsed = time.perf_counter() - start_time
        self.last_stats = {
            "algorithm": "MCTS",
            "difficulty": self.difficulty,
            "simulations": self.simulation_budget,
            "nodes_created": nodes_created,
            "thinking_time": elapsed,
            "chosen_move": chosen_move,
            "root_move_win_rate": round(best_win_rate * 100.0, 1),
            "root_child_visits": visit_map,
            "root_child_win_rates": win_rate_map,
        }
        return chosen_move

    def _simulate_rollout(self, state: UltimateTicTacToe) -> str:
        """
        Perform a legal simulation rollout from `state` until a terminal result is reached.
        Uses fast copying and light tactical awareness on Medium/Hard difficulty (taking
        immediate small-board wins when forced to a single board) while remaining fast and unbiased.
        """
        sim_state = state.copy()
        smart_rollout = self.difficulty in ("Medium", "Hard")

        while not sim_state.is_terminal():
            legal_moves = sim_state.get_legal_moves()
            if not legal_moves:
                break

            selected_move: Optional[Tuple[int, int]] = None
            if smart_rollout and sim_state.active_board is not None and len(legal_moves) <= 9:
                selected_move = self._find_immediate_small_board_win(sim_state, legal_moves)

            if selected_move is None:
                selected_move = self.rng.choice(legal_moves)

            sim_state.apply_move(selected_move)

        return sim_state.get_winner() or DRAW

    @staticmethod
    def _find_immediate_small_board_win(
        state: UltimateTicTacToe, legal_moves: List[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        """Check if the active player has an immediate winning cell in the forced small board."""
        b_idx = legal_moves[0][0]
        board = state.small_boards[b_idx]
        player = state.current_player

        for a, b, c in WIN_LINES:
            vals = [board[a], board[b], board[c]]
            if vals.count(player) == 2 and vals.count(EMPTY) == 1:
                empty_cell = a if board[a] == EMPTY else (b if board[b] == EMPTY else c)
                candidate = (b_idx, empty_cell)
                if candidate in legal_moves:
                    return candidate
        return None