"""
tournament.py
Automated 100-Game Tournament comparing Minimax (with Alpha-Beta Pruning) against
Monte Carlo Tree Search (MCTS) under balanced X/O assignments:
- Games 1-50: Minimax plays as X, MCTS plays as O.
- Games 51-100: MCTS plays as X, Minimax plays as O.
Includes in-memory Matplotlib chart generation using the headless Agg backend.
"""

import time
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from game_logic import DRAW, PLAYER_O, PLAYER_X, UltimateTicTacToe
from mcts_ai import MCTSAI
from minimax_ai import MinimaxAI


class TournamentRunner:
    """
    Executes and tracks a 100-game tournament between Minimax and MCTS.
    All statistics and Matplotlib charts are maintained purely in memory.
    """

    def __init__(
        self,
        minimax_difficulty: str = "Hard",
        mcts_difficulty: str = "Hard",
        custom_minimax_depth: Optional[int] = None,
        custom_mcts_simulations: Optional[int] = None,
        base_seed: int = 2026,
    ) -> None:
        self.minimax_difficulty = minimax_difficulty
        self.mcts_difficulty = mcts_difficulty
        self.custom_minimax_depth = custom_minimax_depth
        self.custom_mcts_simulations = custom_mcts_simulations
        self.base_seed = base_seed

        # Reference agents to document fixed tournament configuration
        ref_mm = MinimaxAI(difficulty=minimax_difficulty, custom_depth=custom_minimax_depth)
        ref_mcts = MCTSAI(
            difficulty=mcts_difficulty,
            custom_simulations=custom_mcts_simulations,
            seed=base_seed,
        )
        self.minimax_config_str: str = ref_mm.get_config_summary()
        self.mcts_config_str: str = ref_mcts.get_config_summary()

        # Live tracking counters
        self.completed_games: int = 0
        self.failed_games: int = 0
        self.minimax_wins: int = 0
        self.mcts_wins: int = 0
        self.draws: int = 0

        self.minimax_as_x: Dict[str, int] = {"wins": 0, "losses": 0, "draws": 0}
        self.minimax_as_o: Dict[str, int] = {"wins": 0, "losses": 0, "draws": 0}
        self.mcts_as_x: Dict[str, int] = {"wins": 0, "losses": 0, "draws": 0}
        self.mcts_as_o: Dict[str, int] = {"wins": 0, "losses": 0, "draws": 0}

        self.total_minimax_time: float = 0.0
        self.total_minimax_moves: int = 0
        self.total_mcts_time: float = 0.0
        self.total_mcts_moves: int = 0
        self.start_time: float = 0.0
        self._abort_requested: bool = False
        self.game_log: List[Dict[str, object]] = []

    def abort(self) -> None:
        """Signal the tournament loop to stop safely after the current move."""
        self._abort_requested = True

    @staticmethod
    def generate_schedule() -> List[Dict[str, object]]:
        """
        Generate the mandatory 100-game tournament assignment schedule:
        - Games 1-50: Minimax = X, MCTS = O
        - Games 51-100: MCTS = X, Minimax = O
        """
        schedule: List[Dict[str, object]] = []
        for game_num in range(1, 101):
            if game_num <= 50:
                schedule.append(
                    {"game_number": game_num, "x_agent": "Minimax", "o_agent": "MCTS"}
                )
            else:
                schedule.append(
                    {"game_number": game_num, "x_agent": "MCTS", "o_agent": "Minimax"}
                )
        return schedule

    def _play_single_game(self, assignment: Dict[str, object]) -> Dict[str, object]:
        """Play one complete game of Ultimate Tic-Tac-Toe between Minimax and MCTS."""
        game_num = int(assignment["game_number"])
        x_agent_name = str(assignment["x_agent"])

        game = UltimateTicTacToe()
        mm_agent = MinimaxAI(
            difficulty=self.minimax_difficulty, custom_depth=self.custom_minimax_depth
        )
        mcts_agent = MCTSAI(
            difficulty=self.mcts_difficulty,
            custom_simulations=self.custom_mcts_simulations,
            seed=self.base_seed + game_num,
        )

        mm_time = 0.0
        mm_moves = 0
        mcts_time = 0.0
        mcts_moves = 0

        try:
            while not game.is_terminal():
                if self._abort_requested:
                    return {
                        "completed": False,
                        "winner": None,
                        "moves": game.move_count,
                        "minimax_time": mm_time,
                        "minimax_moves": mm_moves,
                        "mcts_time": mcts_time,
                        "mcts_moves": mcts_moves,
                        "error": "Aborted by user",
                    }

                current_symbol = game.current_player
                is_minimax_turn = (
                    (current_symbol == PLAYER_X and x_agent_name == "Minimax")
                    or (current_symbol == PLAYER_O and x_agent_name == "MCTS")
                )

                t0 = time.perf_counter()
                if is_minimax_turn:
                    move = mm_agent.get_move(game)
                    mm_time += time.perf_counter() - t0
                    mm_moves += 1
                else:
                    move = mcts_agent.get_move(game)
                    mcts_time += time.perf_counter() - t0
                    mcts_moves += 1

                if move is None or not game.is_valid_move(move):
                    raise RuntimeError(
                        f"Illegal move {move!r} returned by "
                        f"{'Minimax' if is_minimax_turn else 'MCTS'} in game {game_num}"
                    )

                game.apply_move(move)

            return {
                "completed": True,
                "winner": game.get_winner(),
                "moves": game.move_count,
                "minimax_time": mm_time,
                "minimax_moves": mm_moves,
                "mcts_time": mcts_time,
                "mcts_moves": mcts_moves,
                "error": None,
            }

        except Exception as exc:
            return {
                "completed": False,
                "winner": None,
                "moves": game.move_count,
                "minimax_time": mm_time,
                "minimax_moves": mm_moves,
                "mcts_time": mcts_time,
                "mcts_moves": mcts_moves,
                "error": str(exc),
            }

    def run_tournament(
        self,
        progress_callback: Optional[Callable[[int, Dict[str, object]], None]] = None,
    ) -> Dict[str, object]:
        """Run all 100 scheduled games and return the complete verified dictionary of statistics."""
        self.start_time = time.perf_counter()
        schedule = self.generate_schedule()

        for assignment in schedule:
            if self._abort_requested:
                break

            outcome = self._play_single_game(assignment)
            self.game_log.append({**assignment, **outcome})

            if not outcome["completed"]:
                if not self._abort_requested:
                    self.failed_games += 1
                continue

            self.completed_games += 1
            winner = outcome["winner"]
            x_agent = assignment["x_agent"]

            self.total_minimax_time += float(outcome["minimax_time"])
            self.total_minimax_moves += int(outcome["minimax_moves"])
            self.total_mcts_time += float(outcome["mcts_time"])
            self.total_mcts_moves += int(outcome["mcts_moves"])

            if x_agent == "Minimax":
                # Games 1-50: Minimax is X, MCTS is O
                if winner == PLAYER_X:
                    self.minimax_wins += 1
                    self.minimax_as_x["wins"] += 1
                    self.mcts_as_o["losses"] += 1
                elif winner == PLAYER_O:
                    self.mcts_wins += 1
                    self.mcts_as_o["wins"] += 1
                    self.minimax_as_x["losses"] += 1
                else:
                    self.draws += 1
                    self.minimax_as_x["draws"] += 1
                    self.mcts_as_o["draws"] += 1
            else:
                # Games 51-100: MCTS is X, Minimax is O
                if winner == PLAYER_X:
                    self.mcts_wins += 1
                    self.mcts_as_x["wins"] += 1
                    self.minimax_as_o["losses"] += 1
                elif winner == PLAYER_O:
                    self.minimax_wins += 1
                    self.minimax_as_o["wins"] += 1
                    self.mcts_as_x["losses"] += 1
                else:
                    self.draws += 1
                    self.mcts_as_x["draws"] += 1
                    self.minimax_as_o["draws"] += 1

            if progress_callback is not None:
                progress_callback(self.completed_games, self.get_live_snapshot())

        return self.get_live_snapshot()

    def get_live_snapshot(self) -> Dict[str, object]:
        """Compute current tournament metrics using completed_games as the denominator."""
        denom = max(1, self.completed_games)
        has_completed = self.completed_games > 0

        mm_win_rate = (self.minimax_wins / denom) * 100.0 if has_completed else 0.0
        mcts_win_rate = (self.mcts_wins / denom) * 100.0 if has_completed else 0.0
        draw_rate = (self.draws / denom) * 100.0 if has_completed else 0.0

        mm_points = self.minimax_wins + 0.5 * self.draws
        mcts_points = self.mcts_wins + 0.5 * self.draws
        mm_score_rate = (mm_points / denom) * 100.0 if has_completed else 0.0
        mcts_score_rate = (mcts_points / denom) * 100.0 if has_completed else 0.0

        mm_avg_ms = (
            (self.total_minimax_time / self.total_minimax_moves) * 1000.0
            if self.total_minimax_moves > 0
            else 0.0
        )
        mcts_avg_ms = (
            (self.total_mcts_time / self.total_mcts_moves) * 1000.0
            if self.total_mcts_moves > 0
            else 0.0
        )
        elapsed = time.perf_counter() - self.start_time if self.start_time > 0 else 0.0

        return {
            "completed_games": self.completed_games,
            "failed_games": self.failed_games,
            "aborted": self._abort_requested,
            "minimax_wins": self.minimax_wins,
            "mcts_wins": self.mcts_wins,
            "draws": self.draws,
            "minimax_win_rate": mm_win_rate,
            "mcts_win_rate": mcts_win_rate,
            "draw_rate": draw_rate,
            "minimax_score_rate": mm_score_rate,
            "mcts_score_rate": mcts_score_rate,
            "minimax_as_x": dict(self.minimax_as_x),
            "minimax_as_o": dict(self.minimax_as_o),
            "mcts_as_x": dict(self.mcts_as_x),
            "mcts_as_o": dict(self.mcts_as_o),
            "minimax_avg_move_ms": mm_avg_ms,
            "mcts_avg_move_ms": mcts_avg_ms,
            "duration_sec": elapsed,
            "minimax_config": self.minimax_config_str,
            "mcts_config": self.mcts_config_str,
        }

    @staticmethod
    def build_matplotlib_figure(results: Dict[str, object]) -> plt.Figure:
        """
        Build a two-panel Matplotlib figure from actual recorded tournament results:
        - Left panel: Overall Wins, Draws, and Score Rates.
        - Right panel: Wins split by playing as X (First) vs. O (Second).
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 4.6), dpi=100)
        fig.patch.set_facecolor("#162244")

        for ax in (ax1, ax2):
            ax.set_facecolor("#0b132b")
            ax.tick_params(colors="#f1f5f9", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#415a91")

        # Panel 1: Overall Outcomes
        categories = ["Minimax Wins", "MCTS Wins", "Draws"]
        counts = [
            int(results.get("minimax_wins", 0)),
            int(results.get("mcts_wins", 0)),
            int(results.get("draws", 0)),
        ]
        colors = ["#4cc9f0", "#f77f66", "#94a3b8"]
        bars1 = ax1.bar(categories, counts, color=colors, width=0.55)
        ax1.set_title(
            f"Overall Outcomes (N={results.get('completed_games', 0)})",
            color="#f1f5f9",
            fontsize=11,
            fontweight="bold",
        )
        ax1.set_ylabel("Games", color="#f1f5f9", fontsize=10)
        ax1.set_ylim(0, max(100, max(counts, default=10) + 12))

        for bar in bars1:
            h = int(bar.get_height())
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 2,
                f"{h}",
                ha="center",
                va="bottom",
                color="#f1f5f9",
                fontsize=9,
                fontweight="bold",
            )

        # Panel 2: Wins by Symbol (X vs O)
        labels = ["As Player X", "As Player O"]
        mm_split = [
            int(results["minimax_as_x"]["wins"]),
            int(results["minimax_as_o"]["wins"]),
        ]
        mcts_split = [
            int(results["mcts_as_x"]["wins"]),
            int(results["mcts_as_o"]["wins"]),
        ]

        x_pos = [0, 1]
        width = 0.32
        b_mm = ax2.bar([p - width / 2 for p in x_pos], mm_split, width=width, label="Minimax", color="#4cc9f0")
        b_mc = ax2.bar([p + width / 2 for p in x_pos], mcts_split, width=width, label="MCTS", color="#f77f66")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, color="#f1f5f9")
        ax2.set_title("Wins Split by Symbol (50 Games Each)", color="#f1f5f9", fontsize=11, fontweight="bold")
        ax2.set_ylim(0, 55)
        legend = ax2.legend(facecolor="#162244", edgecolor="#415a91", fontsize=8)
        for text in legend.get_texts():
            text.set_color("#f1f5f9")

        for bar in list(b_mm) + list(b_mc):
            h = int(bar.get_height())
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 1,
                f"{h}",
                ha="center",
                va="bottom",
                color="#f1f5f9",
                fontsize=9,
            )

        fig.tight_layout()
        return fig

    def create_pygame_chart_surface(
        self, results: Dict[str, object], size: Tuple[int, int] = (660, 440)
    ) -> Optional["object"]:
        """
        Render the Matplotlib figure into an in-memory RGBA buffer and convert it
        into a Pygame Surface so it embeds directly inside the Pygame results screen
        without opening a conflicting secondary GUI window or writing files to disk.
        """
        import pygame

        fig = self.build_matplotlib_figure(results)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        raw_rgba = bytes(canvas.buffer_rgba())
        w, h = canvas.get_width_height()
        surf = pygame.image.frombuffer(raw_rgba, (w, h), "RGBA")
        plt.close(fig)
        if (w, h) != size:
            surf = pygame.transform.smoothscale(surf, size)
        return surf


def print_cli_tournament_report(results: Dict[str, object]) -> None:
    """Print a formatted tournament report to standard output."""
    print("\n" + "=" * 68)
    print(" ULTIMATE TIC-TAC-TOE: 100-GAME AI TOURNAMENT REPORT")
    print("=" * 68)
    print(f"Minimax Configuration : {results['minimax_config']}")
    print(f"MCTS Configuration    : {results['mcts_config']}")
    print(f"Completed Games       : {results['completed_games']} / 100")
    print(f"Failed/Aborted Games  : {results['failed_games']}")
    print("-" * 68)
    print(f"Minimax Wins          : {results['minimax_wins']} ({results['minimax_win_rate']:.2f}%)")
    print(f"MCTS Wins             : {results['mcts_wins']} ({results['mcts_win_rate']:.2f}%)")
    print(f"Draws                 : {results['draws']} ({results['draw_rate']:.2f}%)")
    print(f"Minimax Score Rate    : {results['minimax_score_rate']:.2f}% (1=Win, 0.5=Draw)")
    print(f"MCTS Score Rate       : {results['mcts_score_rate']:.2f}% (1=Win, 0.5=Draw)")
    print("-" * 68)
    print(
        f"Minimax as X (G1-50)  : {results['minimax_as_x']['wins']}W - "
        f"{results['minimax_as_x']['losses']}L - {results['minimax_as_x']['draws']}D"
    )
    print(
        f"Minimax as O (G51-100): {results['minimax_as_o']['wins']}W - "
        f"{results['minimax_as_o']['losses']}L - {results['minimax_as_o']['draws']}D"
    )
    print(
        f"MCTS as X (G51-100)   : {results['mcts_as_x']['wins']}W - "
        f"{results['mcts_as_x']['losses']}L - {results['mcts_as_x']['draws']}D"
    )
    print(
        f"MCTS as O (G1-50)     : {results['mcts_as_o']['wins']}W - "
        f"{results['mcts_as_o']['losses']}L - {results['mcts_as_o']['draws']}D"
    )
    print("-" * 68)
    print(f"Minimax Avg Move Time : {results['minimax_avg_move_ms']:.2f} ms")
    print(f"MCTS Avg Move Time    : {results['mcts_avg_move_ms']:.2f} ms")
    print(f"Total Duration        : {results['duration_sec']:.2f} seconds")
    print("=" * 68 + "\n")