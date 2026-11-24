"""
main.py
Entry point for Ultimate Tic-Tac-Toe: Minimax vs. Monte Carlo Tree Search.
Launches the Pygame GUI by default, or runs the CLI tournament when `--tournament` is passed.
"""

import argparse
import sys

from tournament import TournamentRunner, print_cli_tournament_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ultimate Tic-Tac-Toe: Minimax vs. Monte Carlo Tree Search"
    )
    parser.add_argument(
        "--tournament",
        action="store_true",
        help="Run the 100-game Minimax vs. MCTS tournament in headless CLI mode.",
    )
    parser.add_argument(
        "--minimax-difficulty",
        choices=["Easy", "Medium", "Hard"],
        default="Hard",
        help="Difficulty setting for Minimax in CLI tournament mode (default: Hard).",
    )
    parser.add_argument(
        "--mcts-difficulty",
        choices=["Easy", "Medium", "Hard"],
        default="Hard",
        help="Difficulty setting for MCTS in CLI tournament mode (default: Hard).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.tournament:
        runner = TournamentRunner(
            minimax_difficulty=args.minimax_difficulty,
            mcts_difficulty=args.mcts_difficulty,
        )
        print(
            f"Starting 100-game tournament: {runner.minimax_config_str} vs. {runner.mcts_config_str}..."
        )
        results = runner.run_tournament(
            progress_callback=lambda done, snap: print(
                f"  Completed {done:3d}/100 games | "
                f"Minimax Wins: {snap['minimax_wins']} | "
                f"MCTS Wins: {snap['mcts_wins']} | "
                f"Draws: {snap['draws']}"
            )
        )
        print_cli_tournament_report(results)
        return 0 if results["completed_games"] == 100 else 1

    from gui import UltimateTTTGUI

    app = UltimateTTTGUI()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())