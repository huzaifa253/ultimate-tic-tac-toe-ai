"""
game_logic.py
Core rules and state management for Ultimate Tic-Tac-Toe.
Completely independent of Pygame or any UI framework.
"""

from typing import List, Optional, Tuple

EMPTY: str = ""
PLAYER_X: str = "X"
PLAYER_O: str = "O"
DRAW: str = "D"

WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


class UltimateTicTacToe:
    """
    Represents the complete state and rule engine of an Ultimate Tic-Tac-Toe game.
    - small_boards: 9 lists of 9 strings each, representing cells 0..8 inside boards 0..8.
    - macro_board: list of 9 strings representing the status of each small board
      (EMPTY, PLAYER_X, PLAYER_O, or DRAW).
    - current_player: PLAYER_X or PLAYER_O.
    - active_board: int (0..8) if forced to play in a specific small board, or None if free choice.
    - winner: None while ongoing, or PLAYER_X, PLAYER_O, or DRAW when terminal.
    - move_count: total number of legal moves applied so far.
    - last_move: Optional tuple (board_idx, cell_idx) of the most recent move.
    - winning_line: Optional tuple of 3 macro indices if the game was won by a line.
    """

    def __init__(self) -> None:
        self.small_boards: List[List[str]] = [[EMPTY] * 9 for _ in range(9)]
        self.macro_board: List[str] = [EMPTY] * 9
        self.current_player: str = PLAYER_X
        self.active_board: Optional[int] = None
        self.winner: Optional[str] = None
        self.move_count: int = 0
        self.last_move: Optional[Tuple[int, int]] = None
        self.winning_line: Optional[Tuple[int, int, int]] = None

    def reset(self) -> None:
        """Reset the game to its initial state."""
        self.small_boards = [[EMPTY] * 9 for _ in range(9)]
        self.macro_board = [EMPTY] * 9
        self.current_player = PLAYER_X
        self.active_board = None
        self.winner = None
        self.move_count = 0
        self.last_move = None
        self.winning_line = None

    def copy(self) -> "UltimateTicTacToe":
        """Create a fast, independent deep copy of the game state for AI search."""
        clone = UltimateTicTacToe.__new__(UltimateTicTacToe)
        clone.small_boards = [board[:] for board in self.small_boards]
        clone.macro_board = self.macro_board[:]
        clone.current_player = self.current_player
        clone.active_board = self.active_board
        clone.winner = self.winner
        clone.move_count = self.move_count
        clone.last_move = self.last_move
        clone.winning_line = self.winning_line
        return clone

    @staticmethod
    def get_opponent(player: str) -> str:
        """Return the opposing player symbol."""
        if player == PLAYER_X:
            return PLAYER_O
        if player == PLAYER_O:
            return PLAYER_X
        raise ValueError(f"Invalid player symbol: {player!r}")

    @staticmethod
    def check_3x3_winner(cells: List[str]) -> Tuple[Optional[str], Optional[Tuple[int, int, int]]]:
        """
        Check a 9-element list for a 3-in-a-row winner or draw.
        Ignores DRAW markers when checking for 3-in-a-row (crucial for the macro board).
        Returns (status, winning_line) where status is PLAYER_X, PLAYER_O, DRAW, or None.
        """
        for a, b, c in WIN_LINES:
            val = cells[a]
            if val in (PLAYER_X, PLAYER_O) and val == cells[b] == cells[c]:
                return val, (a, b, c)
        if all(cell != EMPTY for cell in cells):
            return DRAW, None
        return None, None

    def check_small_board_status(self, board_idx: int) -> Optional[str]:
        """Evaluate and return the status of a specific small board (0..8)."""
        if not isinstance(board_idx, int) or not (0 <= board_idx < 9):
            raise ValueError(f"Invalid board_idx: {board_idx}")
        status, _ = self.check_3x3_winner(self.small_boards[board_idx])
        return status

    def is_terminal(self) -> bool:
        """Return True if the overall game has ended in a win or draw."""
        return self.winner is not None

    def get_winner(self) -> Optional[str]:
        """Return PLAYER_X, PLAYER_O, DRAW, or None if the game is still ongoing."""
        return self.winner

    def get_Valid_boards(self) -> List[int]:
        """Return the list of small-board indices where the current player is allowed to move."""
        if self.winner is not None:
            return []
        if self.active_board is not None and self.macro_board[self.active_board] == EMPTY:
            return [self.active_board]
        return [b for b in range(9) if self.macro_board[b] == EMPTY]

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """
        Return all legal moves as a list of (board_idx, cell_idx) tuples.
        Returns an empty list if the game is already terminal.
        """
        if self.winner is not None:
            return []

        if self.active_board is not None and self.macro_board[self.active_board] == EMPTY:
            board = self.small_boards[self.active_board]
            return [(self.active_board, c) for c in range(9) if board[c] == EMPTY]

        moves: List[Tuple[int, int]] = []
        for b in range(9):
            if self.macro_board[b] == EMPTY:
                board = self.small_boards[b]
                for c in range(9):
                    if board[c] == EMPTY:
                        moves.append((b, c))
        return moves

    def is_valid_move(self, move: Tuple[int, int]) -> bool:
        """Validate whether a move tuple (board_idx, cell_idx) is legal in the current state."""
        if self.winner is not None:
            return False
        if not isinstance(move, (tuple, list)) or len(move) != 2:
            return False
        board_idx, cell_idx = move
        if not isinstance(board_idx, int) or not isinstance(cell_idx, int):
            return False
        if isinstance(board_idx, bool) or isinstance(cell_idx, bool):
            return False
        if not (0 <= board_idx < 9 and 0 <= cell_idx < 9):
            return False
        if self.macro_board[board_idx] != EMPTY:
            return False
        if self.active_board is not None and self.macro_board[self.active_board] == EMPTY:
            if board_idx != self.active_board:
                return False
        if self.small_boards[board_idx][cell_idx] != EMPTY:
            return False
        return True

    def apply_move(self, move: Tuple[int, int]) -> bool:
        """
        Apply a move (board_idx, cell_idx) to the current state.
        Raises ValueError if the move is illegal or the game is already over.
        Returns True on successful application.
        """
        if not self.is_valid_move(move):
            raise ValueError(
                f"Illegal move {move!r} for player {self.current_player} "
                f"(active_board={self.active_board}, winner={self.winner})"
            )

        board_idx, cell_idx = move
        self.small_boards[board_idx][cell_idx] = self.current_player
        self.move_count += 1
        self.last_move = (board_idx, cell_idx)

        # Update small board status if it just finished
        small_status, _ = self.check_3x3_winner(self.small_boards[board_idx])
        if small_status is not None:
            self.macro_board[board_idx] = small_status

        # Check overall macro board status
        macro_status, macro_line = self.check_3x3_winner(self.macro_board)
        if macro_status is not None:
            self.winner = macro_status
            self.winning_line = macro_line
            self.active_board = None
        else:
            # Determine next forced board based on cell_idx
            if self.macro_board[cell_idx] == EMPTY:
                self.active_board = cell_idx
            else:
                self.active_board = None

            # Defensive check: if no legal moves remain anywhere, declare DRAW
            if not any(
                self.macro_board[b] == EMPTY and EMPTY in self.small_boards[b]
                for b in range(9)
            ):
                self.winner = DRAW
                self.active_board = None

        # Switch active turn
        self.current_player = self.get_opponent(self.current_player)
        return True

    def count_small_boards_won(self) -> Tuple[int, int, int]:
        """Return (x_wins, o_wins, draws) on the macro board."""
        x_wins = sum(1 for status in self.macro_board if status == PLAYER_X)
        o_wins = sum(1 for status in self.macro_board if status == PLAYER_O)
        draws = sum(1 for status in self.macro_board if status == DRAW)
        return x_wins, o_wins, draws