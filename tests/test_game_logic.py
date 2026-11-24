"""
tests/test_game_logic.py
Comprehensive unit tests covering Ultimate Tic-Tac-Toe rules, Minimax with Alpha-Beta
Pruning, Monte Carlo Tree Search (MCTS), and the 100-game Tournament engine.
"""

import unittest
from unittest.mock import patch

from game_logic import (
    DRAW,
    EMPTY,
    PLAYER_O,
    PLAYER_X,
    UltimateTicTacToe,
)
from mcts_ai import MCTSAI
from minimax_ai import TERMINAL_SCORE, MinimaxAI
from tournament import TournamentRunner


class TestUltimateTicTacToeLogic(unittest.TestCase):
    """Tests for Ultimate Tic-Tac-Toe board representation, routing, wins, draws, and immutability."""

    def test_initial_board_state(self) -> None:
        game = UltimateTicTacToe()
        self.assertEqual(game.current_player, PLAYER_X)
        self.assertIsNone(game.active_board)
        self.assertIsNone(game.winner)
        self.assertFalse(game.is_terminal())
        self.assertEqual(game.move_count, 0)
        self.assertEqual(len(game.get_legal_moves()), 81)
        self.assertEqual(game.macro_board, [EMPTY] * 9)

    def test_apply_move_and_forced_board_routing(self) -> None:
        game = UltimateTicTacToe()
        game.apply_move((4, 2))
        self.assertEqual(game.small_boards[4][2], PLAYER_X)
        self.assertEqual(game.current_player, PLAYER_O)
        self.assertEqual(game.active_board, 2)

        legal_moves = game.get_legal_moves()
        self.assertEqual(len(legal_moves), 9)
        self.assertTrue(all(b == 2 for b, _ in legal_moves))

        # Playing in a non-forced board must be rejected
        self.assertFalse(game.is_valid_move((0, 0)))
        with self.assertRaises(ValueError):
            game.apply_move((0, 0))

    def test_occupied_cell_rejection(self) -> None:
        game = UltimateTicTacToe()
        game.apply_move((4, 4))
        self.assertFalse(game.is_valid_move((4, 4)))
        with self.assertRaises(ValueError):
            game.apply_move((4, 4))

    def test_free_choice_routing_when_destination_board_is_complete(self) -> None:
        game = UltimateTicTacToe()
        # Complete small board 2 for X
        game.small_boards[2][0] = PLAYER_X
        game.small_boards[2][1] = PLAYER_X
        game.small_boards[2][2] = PLAYER_X
        game.macro_board[2] = PLAYER_X

        # Now X plays in board 4, cell 2 -> sends O to board 2, which is already won
        game.apply_move((4, 2))
        self.assertIsNone(game.active_board)
        legal_moves = game.get_legal_moves()
        # All cells in unfinished boards (8 boards * 9 cells - 1 occupied cell in board 4) = 71
        self.assertEqual(len(legal_moves), 71)
        self.assertTrue(all(b != 2 for b, _ in legal_moves))

    def test_small_board_win_detection_and_move_prevention(self) -> None:
        game = UltimateTicTacToe()
        game.small_boards[0][0] = PLAYER_X
        game.small_boards[0][1] = PLAYER_X
        game.active_board = 0
        game.current_player = PLAYER_X
        game.apply_move((0, 2))

        self.assertEqual(game.macro_board[0], PLAYER_X)
        # Even though cell 3 in board 0 is empty, board 0 is finished
        self.assertFalse(game.is_valid_move((0, 3)))

    def test_small_board_draw_detection(self) -> None:
        game = UltimateTicTacToe()
        # Construct a drawn small board 1 leaving cell 8 empty:
        # X O X
        # X O O
        # O X .
        pattern = [
            PLAYER_X, PLAYER_O, PLAYER_X,
            PLAYER_X, PLAYER_O, PLAYER_O,
            PLAYER_O, PLAYER_X, EMPTY,
        ]
        game.small_boards[1] = pattern[:]
        game.active_board = 1
        game.current_player = PLAYER_X
        game.apply_move((1, 8))

        self.assertEqual(game.macro_board[1], DRAW)
        self.assertIsNone(game.winner)

    def test_overall_win_and_terminal_move_prevention(self) -> None:
        game = UltimateTicTacToe()
        game.macro_board[0] = PLAYER_X
        game.macro_board[1] = PLAYER_X
        game.small_boards[2][0] = PLAYER_X
        game.small_boards[2][1] = PLAYER_X
        game.active_board = 2
        game.current_player = PLAYER_X

        game.apply_move((2, 2))
        self.assertTrue(game.is_terminal())
        self.assertEqual(game.get_winner(), PLAYER_X)
        self.assertEqual(game.winning_line, (0, 1, 2))
        self.assertEqual(game.get_legal_moves(), [])

        with self.assertRaises(ValueError):
            game.apply_move((3, 0))

    def test_overall_draw_when_macro_board_fills_without_line(self) -> None:
        game = UltimateTicTacToe()
        # Macro board pattern with no 3-in-a-row:
        # X O X
        # X O O
        # O X . (board 8 about to be drawn)
        macro_setup = [
            PLAYER_X, PLAYER_O, PLAYER_X,
            PLAYER_X, PLAYER_O, PLAYER_O,
            PLAYER_O, PLAYER_X, EMPTY,
        ]
        game.macro_board = macro_setup[:]
        game.small_boards[8] = [
            PLAYER_X, PLAYER_O, PLAYER_X,
            PLAYER_X, PLAYER_O, PLAYER_O,
            PLAYER_O, PLAYER_X, EMPTY,
        ]
        game.active_board = 8
        game.current_player = PLAYER_X
        game.apply_move((8, 8))

        self.assertTrue(game.is_terminal())
        self.assertEqual(game.get_winner(), DRAW)
        self.assertEqual(game.get_legal_moves(), [])

    def test_three_drawn_small_boards_do_not_count_as_win(self) -> None:
        game = UltimateTicTacToe()
        game.macro_board[0] = DRAW
        game.macro_board[1] = DRAW
        game.macro_board[2] = DRAW
        status, line = game.check_3x3_winner(game.macro_board)
        self.assertIsNone(status)
        self.assertIsNone(line)

    def test_invalid_indices_and_malformed_moves(self) -> None:
        game = UltimateTicTacToe()
        invalid_inputs = [
            (-1, 0),
            (0, -1),
            (9, 0),
            (0, 9),
            (True, 0),
            ("0", 0),
            (0,),
            (0, 1, 2),
            None,
        ]
        for bad_move in invalid_inputs:
            self.assertFalse(game.is_valid_move(bad_move))  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                game.apply_move(bad_move)  # type: ignore[arg-type]

    def test_state_copy_immutability(self) -> None:
        game = UltimateTicTacToe()
        game.apply_move((4, 4))
        snapshot = game.copy()
        snapshot.apply_move((4, 0))

        self.assertEqual(game.move_count, 1)
        self.assertEqual(snapshot.move_count, 2)
        self.assertEqual(game.small_boards[4][0], EMPTY)
        self.assertEqual(snapshot.small_boards[4][0], PLAYER_O)


class TestMinimaxAI(unittest.TestCase):
    """Tests for Minimax with alpha-beta pruning, evaluation, and state preservation."""

    def test_returns_legal_move_and_preserves_state(self) -> None:
        game = UltimateTicTacToe()
        game.apply_move((4, 4))
        before_boards = [b[:] for b in game.small_boards]
        before_macro = game.macro_board[:]

        ai = MinimaxAI(difficulty="Easy")
        move = ai.get_move(game)

        self.assertIsNotNone(move)
        self.assertTrue(game.is_valid_move(move))  # type: ignore[arg-type]
        self.assertEqual(game.small_boards, before_boards)
        self.assertEqual(game.macro_board, before_macro)
        self.assertGreater(ai.last_stats["nodes_evaluated"], 0)

    def test_recognizes_immediate_winning_move(self) -> None:
        game = UltimateTicTacToe()
        game.macro_board[0] = PLAYER_X
        game.macro_board[1] = PLAYER_X
        game.small_boards[2] = [
            PLAYER_X, PLAYER_X, EMPTY,
            PLAYER_O, PLAYER_O, EMPTY,
            EMPTY, EMPTY, EMPTY,
        ]
        game.active_board = 2
        game.current_player = PLAYER_X

        ai = MinimaxAI(difficulty="Easy")
        move = ai.get_move(game)
        self.assertEqual(move, (2, 2))
        self.assertGreaterEqual(ai.last_stats["eval_score"], TERMINAL_SCORE)

    def test_avoids_immediate_loss(self) -> None:
        game = UltimateTicTacToe()
        # O threatens to win the whole game on board 2 cell 2 if X sends O there or doesn't block
        game.macro_board[0] = PLAYER_O
        game.macro_board[1] = PLAYER_O
        game.small_boards[2] = [
            PLAYER_O, PLAYER_O, EMPTY,
            PLAYER_X, EMPTY, EMPTY,
            EMPTY, EMPTY, EMPTY,
        ]
        game.active_board = 2
        game.current_player = PLAYER_X

        ai = MinimaxAI(difficulty="Medium")
        move = ai.get_move(game)
        # X must block cell 2 in board 2 to prevent O from winning board 2 and the game
        self.assertEqual(move, (2, 2))

    def test_terminal_state_returns_none(self) -> None:
        game = UltimateTicTacToe()
        game.winner = PLAYER_X
        ai = MinimaxAI(difficulty="Easy")
        self.assertIsNone(ai.get_move(game))

    def test_alpha_beta_matches_plain_minimax_and_prunes_branches(self) -> None:
        game = UltimateTicTacToe()
        # Set up a tactical mid-game state with constrained branching
        moves_seq = [(4, 4), (4, 0), (0, 4), (4, 8), (8, 4), (4, 2), (2, 4), (4, 6)]
        for m in moves_seq:
            game.apply_move(m)

        ai = MinimaxAI(difficulty="Medium", custom_depth=3)
        ab_score, ab_move, ab_nodes, ab_cutoffs = ai.search_comparison(game, depth=3, use_alpha_beta=True)
        plain_score, plain_move, plain_nodes, _ = ai.search_comparison(game, depth=3, use_alpha_beta=False)

        self.assertAlmostEqual(ab_score, plain_score, places=5)
        self.assertIsNotNone(ab_move)
        self.assertIsNotNone(plain_move)
        self.assertGreater(ab_cutoffs, 0)
        self.assertLess(ab_nodes, plain_nodes)


class TestMCTSAI(unittest.TestCase):
    """Tests for Monte Carlo Tree Search stages, statistics, reproducibility, and state safety."""

    def test_returns_legal_move_and_preserves_state(self) -> None:
        game = UltimateTicTacToe()
        game.apply_move((4, 4))
        before_boards = [b[:] for b in game.small_boards]

        ai = MCTSAI(difficulty="Easy", custom_simulations=40, seed=42)
        move = ai.get_move(game)

        self.assertIsNotNone(move)
        self.assertTrue(game.is_valid_move(move))  # type: ignore[arg-type]
        self.assertEqual(game.small_boards, before_boards)

        stats = ai.last_stats
        self.assertEqual(stats["simulations"], 40)
        self.assertEqual(sum(stats["root_child_visits"].values()), 40)
        self.assertGreater(stats["nodes_created"], 1)

    def test_reproducible_with_fixed_seed(self) -> None:
        game = UltimateTicTacToe()
        game.apply_move((4, 2))

        ai1 = MCTSAI(difficulty="Easy", custom_simulations=50, seed=12345)
        ai2 = MCTSAI(difficulty="Easy", custom_simulations=50, seed=12345)

        move1 = ai1.get_move(game)
        move2 = ai2.get_move(game)

        self.assertEqual(move1, move2)
        self.assertEqual(ai1.last_stats["root_child_visits"], ai2.last_stats["root_child_visits"])

    def test_finds_immediate_winning_move(self) -> None:
        game = UltimateTicTacToe()
        game.macro_board[0] = PLAYER_O
        game.macro_board[4] = PLAYER_O
        game.small_boards[8] = [
            PLAYER_O, PLAYER_O, EMPTY,
            PLAYER_X, PLAYER_X, EMPTY,
            EMPTY, EMPTY, EMPTY,
        ]
        game.active_board = 8
        game.current_player = PLAYER_O

        ai = MCTSAI(difficulty="Medium", custom_simulations=120, seed=7)
        move = ai.get_move(game)
        self.assertEqual(move, (8, 2))

    def test_terminal_state_returns_none(self) -> None:
        game = UltimateTicTacToe()
        game.winner = DRAW
        ai = MCTSAI(difficulty="Easy", custom_simulations=10)
        self.assertIsNone(ai.get_move(game))


class TestTournamentRunner(unittest.TestCase):
    """Tests for 100-game schedule generation, accounting, denominators, and failure isolation."""

    def test_100_game_schedule_balance(self) -> None:
        runner = TournamentRunner()
        schedule = runner.generate_schedule()

        self.assertEqual(len(schedule), 100)
        for idx in range(50):
            self.assertEqual(schedule[idx]["game_number"], idx + 1)
            self.assertEqual(schedule[idx]["x_agent"], "Minimax")
            self.assertEqual(schedule[idx]["o_agent"], "MCTS")

        for idx in range(50, 100):
            self.assertEqual(schedule[idx]["game_number"], idx + 1)
            self.assertEqual(schedule[idx]["x_agent"], "MCTS")
            self.assertEqual(schedule[idx]["o_agent"], "Minimax")

    def test_accounting_and_rates_with_controlled_outcomes(self) -> None:
        runner = TournamentRunner(minimax_difficulty="Easy", mcts_difficulty="Easy")

        # Mock _play_single_game to simulate controlled results across 100 games:
        # Games 1..30 (Minimax=X): X wins (30 Minimax wins as X)
        # Games 31..40 (Minimax=X): O wins (10 MCTS wins as O)
        # Games 41..50 (Minimax=X): Draw (10 Draws)
        # Games 51..70 (MCTS=X): X wins (20 MCTS wins as X)
        # Games 71..85 (MCTS=X): O wins (15 Minimax wins as O)
        # Games 86..100 (MCTS=X): Draw (15 Draws)
        def fake_play(assignment: dict) -> dict:
            g = assignment["game_number"]
            if g <= 30:
                winner = PLAYER_X
            elif g <= 40:
                winner = PLAYER_O
            elif g <= 50:
                winner = DRAW
            elif g <= 70:
                winner = PLAYER_X
            elif g <= 85:
                winner = PLAYER_O
            else:
                winner = DRAW
            return {
                "completed": True,
                "winner": winner,
                "moves": 25,
                "minimax_time": 0.05,
                "minimax_moves": 13,
                "mcts_time": 0.05,
                "mcts_moves": 12,
                "error": None,
            }

        with patch.object(runner, "_play_single_game", side_effect=fake_play):
            results = runner.run_tournament()

        self.assertEqual(results["completed_games"], 100)
        self.assertEqual(results["failed_games"], 0)
        self.assertEqual(results["minimax_wins"], 45)
        self.assertEqual(results["mcts_wins"], 30)
        self.assertEqual(results["draws"], 25)

        self.assertEqual(results["minimax_as_x"]["wins"], 30)
        self.assertEqual(results["minimax_as_x"]["losses"], 10)
        self.assertEqual(results["minimax_as_x"]["draws"], 10)

        self.assertEqual(results["mcts_as_x"]["wins"], 20)
        self.assertEqual(results["mcts_as_x"]["losses"], 15)
        self.assertEqual(results["mcts_as_x"]["draws"], 15)

        self.assertAlmostEqual(results["minimax_win_rate"], 45.0)
        self.assertAlmostEqual(results["mcts_win_rate"], 30.0)
        self.assertAlmostEqual(results["draw_rate"], 25.0)
        # Score rate: (45 + 0.5*25) / 100 = 57.5% for Minimax; (30 + 0.5*25) / 100 = 42.5% for MCTS
        self.assertAlmostEqual(results["minimax_score_rate"], 57.5)
        self.assertAlmostEqual(results["mcts_score_rate"], 42.5)

    def test_failed_games_excluded_from_completed_denominator(self) -> None:
        runner = TournamentRunner(minimax_difficulty="Easy", mcts_difficulty="Easy")

        def fake_play_with_failures(assignment: dict) -> dict:
            g = assignment["game_number"]
            if g in (1, 2, 3, 4):
                return {
                    "completed": False,
                    "winner": None,
                    "moves": 4,
                    "minimax_time": 0.0,
                    "minimax_moves": 0,
                    "mcts_time": 0.0,
                    "mcts_moves": 0,
                    "error": "Simulated exception",
                }
            return {
                "completed": True,
                "winner": PLAYER_X,
                "moves": 20,
                "minimax_time": 0.02,
                "minimax_moves": 10,
                "mcts_time": 0.02,
                "mcts_moves": 10,
                "error": None,
            }

        with patch.object(runner, "_play_single_game", side_effect=fake_play_with_failures):
            results = runner.run_tournament()

        self.assertEqual(results["completed_games"], 96)
        self.assertEqual(results["failed_games"], 4)
        # 46 wins for Minimax (games 5..50) and 50 wins for MCTS (games 51..100)
        self.assertEqual(results["minimax_wins"], 46)
        self.assertEqual(results["mcts_wins"], 50)
        self.assertAlmostEqual(results["minimax_win_rate"], (46 / 96) * 100.0)
        self.assertAlmostEqual(results["mcts_win_rate"], (50 / 96) * 100.0)


if __name__ == "__main__":
    unittest.main()