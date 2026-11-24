"""
gui.py
Pygame GUI for Ultimate Tic-Tac-Toe (Human vs. AI, AI vs. AI, 100-game tournament).

Visual redesign:
- Two hand-tuned themes: "Onyx & Gold" (black / graphite, dark) and "Ivory & Charcoal"
  (warm light, strong dark borders), with emerald (X) / rose (O) / gold (highlight) accents.
- Light / Dark toggle on every screen (button, or press T). The choice is remembered
  in ui_settings.json and the tournament chart is re-rendered in the active theme.
- Card-style main menu with a logo, soft shadows, animated hover lift.
- Anti-aliased X / O glyphs with pop-in animations, ripples, pulsing active-board glow,
  hover ghost preview, animated winning line, game-over overlay, undo, move log, telemetry.
"""

import json
import math
import os
import random
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import pygame

from game_logic import (
    DRAW,
    EMPTY,
    PLAYER_O,
    PLAYER_X,
    UltimateTicTacToe,
)
from mcts_ai import MCTSAI
from minimax_ai import MinimaxAI
from tournament import TournamentRunner

Color = Tuple[int, int, int]

# ----------------------------------------------------------------------
# Theme system (dark "Midnight Violet" / light "Warm Paper")
# ----------------------------------------------------------------------
DARK_THEME: Dict[str, object] = dict(
    name="dark", label="Onyx & Gold",
    bg_top=(7, 7, 8), bg_bottom=(26, 26, 28),
    panel_bg=(22, 22, 25), panel_border=(78, 75, 68),
    board_bg=(13, 13, 15), sub_bg=(27, 27, 30),
    active_board_bg=(46, 40, 24), hover_cell_bg=(62, 57, 44),
    grid_thin=(104, 101, 94), grid_thick=(190, 160, 84),
    color_x=(52, 211, 153), color_o=(246, 112, 130), color_draw=(156, 154, 150),
    text_primary=(250, 249, 245), text_muted=(216, 212, 202), text_dim=(166, 162, 152),
    btn_default=(38, 38, 42), btn_hover=(58, 57, 60),
    btn_active=(214, 176, 62), btn_active_border=(250, 224, 134),
    accent_gold=(226, 186, 72),
    text_on_accent=(22, 17, 4), text_on_color=(10, 10, 10),
    scrim=(0, 0, 0), scrim_alpha=178,
    shadow_rgb=(0, 0, 0), shadow_k=1.0,
    capture_mix=0.24, decor_alpha=22,
    grid_w=2, macro_w=4, border_w=3, vignette=(0, 0, 0, 150),
)

LIGHT_THEME: Dict[str, object] = dict(
    name="light", label="Ivory & Charcoal",
    bg_top=(248, 245, 238), bg_bottom=(233, 227, 213),
    panel_bg=(255, 254, 250), panel_border=(150, 140, 118),
    board_bg=(220, 212, 193), sub_bg=(255, 254, 250),
    active_board_bg=(255, 236, 186), hover_cell_bg=(240, 226, 190),
    grid_thin=(128, 118, 98), grid_thick=(44, 38, 28),
    color_x=(0, 122, 92), color_o=(198, 36, 66), color_draw=(118, 112, 102),
    text_primary=(20, 18, 14), text_muted=(58, 53, 43), text_dim=(98, 92, 80),
    btn_default=(238, 232, 218), btn_hover=(222, 214, 194),
    btn_active=(40, 36, 28), btn_active_border=(184, 142, 36),
    accent_gold=(138, 90, 0),
    text_on_accent=(255, 250, 235), text_on_color=(255, 255, 255),
    scrim=(248, 245, 238), scrim_alpha=212,
    shadow_rgb=(70, 52, 20), shadow_k=0.5,
    capture_mix=0.17, decor_alpha=30,
    grid_w=3, macro_w=5, border_w=4, vignette=(110, 90, 50, 60),
)

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui_settings.json")


class Theme:
    """Mutable, globally shared colour palette. Every draw call reads from `T`."""

    def __init__(self) -> None:
        self.apply("dark")

    def apply(self, name: str) -> None:
        data = LIGHT_THEME if name == "light" else DARK_THEME
        for key, value in data.items():
            setattr(self, key, value)

    @property
    def is_dark(self) -> bool:
        return getattr(self, "name", "dark") == "dark"

    def chart_theme(self) -> Dict[str, str]:
        """Hex colour dictionary understood by TournamentRunner's Matplotlib charts."""
        def hx(c: Color) -> str:
            return "#%02x%02x%02x" % (c[0], c[1], c[2])
        return {
            "figure_bg": hx(self.panel_bg),   # type: ignore[arg-type]
            "axes_bg": hx(self.board_bg),     # type: ignore[arg-type]
            "text": hx(self.text_primary),    # type: ignore[arg-type]
            "muted": hx(self.text_muted),     # type: ignore[arg-type]
            "spine": hx(self.panel_border),   # type: ignore[arg-type]
            "minimax": hx(self.color_x),      # type: ignore[arg-type]
            "mcts": hx(self.color_o),         # type: ignore[arg-type]
            "draw": hx(self.color_draw),      # type: ignore[arg-type]
        }


T = Theme()


def load_saved_theme() -> str:
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
            return "light" if json.load(fh).get("theme") == "light" else "dark"
    except (OSError, ValueError):
        return "dark"


def save_theme(name: str) -> None:
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
            json.dump({"theme": name}, fh)
    except OSError:
        pass


SIDEBAR_W = 400
HEADER_H = 92
MIN_W, MIN_H = 1100, 740
BOARD_PAD = 0.08  # fraction of a small board used as padding around its 3x3 grid


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def clamp01(t: float) -> float:
    return max(0.0, min(1.0, t))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_color(c1: Color, c2: Color, t: float) -> Color:
    return (int(lerp(c1[0], c2[0], t)), int(lerp(c1[1], c2[1], t)), int(lerp(c1[2], c2[2], t)))


def ease_out_cubic(t: float) -> float:
    t = clamp01(t)
    return 1 - (1 - t) ** 3


def ease_out_back(t: float) -> float:
    t = clamp01(t)
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def draw_alpha_rect(
    surface: pygame.Surface,
    color: Tuple[int, ...],
    rect: pygame.Rect,
    radius: int = 8,
    width: int = 0,
) -> None:
    """Draw a (possibly translucent) rounded rectangle."""
    rect = pygame.Rect(rect)
    if rect.width <= 0 or rect.height <= 0:
        return
    layer = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(layer, color, layer.get_rect(), width=width, border_radius=radius)
    surface.blit(layer, rect.topleft)


_SHADOW_CACHE: Dict[Tuple[int, int, int, int, str], pygame.Surface] = {}


def draw_shadow(surface: pygame.Surface, rect: pygame.Rect, radius: int = 12, spread: int = 10) -> None:
    """Soft drop shadow. The shadow is rendered once per size/theme and then simply blitted."""
    rect = pygame.Rect(rect)
    key = (rect.w, rect.h, radius, spread, str(T.name))
    layer = _SHADOW_CACHE.get(key)
    pad = spread + 8
    if layer is None:
        if len(_SHADOW_CACHE) > 80:
            _SHADOW_CACHE.clear()
        layer = pygame.Surface((rect.w + 2 * pad, rect.h + 2 * pad), pygame.SRCALPHA)
        r, g, b = T.shadow_rgb  # type: ignore[misc]
        base = pygame.Rect(pad, pad, rect.w, rect.h)
        for i in range(spread, 0, -2):
            alpha = int((10 + (spread - i) * 1.2) * T.shadow_k)  # type: ignore[operator]
            if alpha > 0:
                draw_alpha_rect(layer, (r, g, b, alpha), base.inflate(i * 2, i * 2).move(0, 4), radius + i)
        _SHADOW_CACHE[key] = layer
    surface.blit(layer, (rect.x - pad, rect.y - pad))


_SYMBOL_CACHE: Dict[Tuple[str, int, Color, float], pygame.Surface] = {}


def symbol_surface(symbol: str, size: int, color: Color, thick: float = 0.11) -> pygame.Surface:
    """Return an anti-aliased X or O glyph (supersampled 4x, cached)."""
    size = max(2, int(size))
    key = (symbol, size, color, round(thick, 3))
    cached = _SYMBOL_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_SYMBOL_CACHE) > 600:
        _SYMBOL_CACHE.clear()

    ss = 4
    s = size * ss
    big = pygame.Surface((s, s), pygame.SRCALPHA)
    th = max(2, int(s * thick))
    if symbol == PLAYER_X:
        m = int(s * 0.2)
        for p1, p2 in (((m, m), (s - m, s - m)), ((s - m, m), (m, s - m))):
            pygame.draw.line(big, color, p1, p2, th)
            pygame.draw.circle(big, color, p1, th // 2)
            pygame.draw.circle(big, color, p2, th // 2)
    else:
        r = int(s * 0.31)
        center = (s // 2, s // 2)
        pygame.draw.circle(big, color, center, r + th // 2)
        pygame.draw.circle(big, (0, 0, 0, 0), center, max(1, r - th // 2))
    surf = pygame.transform.smoothscale(big, (size, size))
    _SYMBOL_CACHE[key] = surf
    return surf


def draw_equals(surface: pygame.Surface, center: Tuple[int, int], size: int, color: Color) -> None:
    """Draw the '=' glyph used for drawn small boards."""
    w = int(size * 0.7)
    h = max(4, int(size * 0.11))
    gap = int(size * 0.17)
    for dy in (-gap, gap):
        r = pygame.Rect(0, 0, w, h)
        r.center = (center[0], center[1] + dy)
        pygame.draw.rect(surface, color, r, border_radius=h // 2)


def render_chart_surface(results: Dict[str, object], size: Tuple[int, int], theme: Dict[str, str]) -> pygame.Surface:
    """Draw the tournament result charts at exactly `size` pixels (no scaling => crisp, readable text)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    th = theme
    dpi = 100
    w, h = size
    fig = Figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.patch.set_facecolor(th["figure_bg"])
    wide = (w / max(1, h)) > 1.7
    ax1, ax2 = fig.subplots(1, 2) if wide else fig.subplots(2, 1)

    for ax in (ax1, ax2):
        ax.set_facecolor(th["figure_bg"])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ---- Panel 1: overall results (horizontal bars, big labels) ----
    vals = [int(results.get("minimax_wins", 0)), int(results.get("mcts_wins", 0)), int(results.get("draws", 0))]
    names = ["Minimax", "MCTS", "Draws"]
    cols = [th["minimax"], th["mcts"], th["draw"]]
    n = max(1, sum(vals))
    ys = [2, 1, 0]
    ax1.barh(ys, vals, color=cols, height=0.62)
    xmax = max(max(vals), 1) * 1.55
    ax1.set_xlim(0, xmax)
    ax1.set_ylim(-0.6, 2.6)
    ax1.set_yticks(ys)
    ax1.set_yticklabels(names, fontsize=13, fontweight="bold", color=th["text"])
    ax1.tick_params(axis="y", length=0, pad=8)
    ax1.xaxis.set_visible(False)
    for y, v in zip(ys, vals):
        ax1.text(v + xmax * 0.025, y, f"{v}  ({v / n * 100:.0f}%)", va="center", ha="left",
                 fontsize=13, fontweight="bold", color=th["text"])
    ax1.set_title(f"Overall results  -  {int(results.get('completed_games', 0))} games",
                  loc="left", fontsize=14, fontweight="bold", color=th["text"], pad=10)

    # ---- Panel 2: wins by side (grouped bars) ----
    mm = [int(results["minimax_as_x"]["wins"]), int(results["minimax_as_o"]["wins"])]   # type: ignore[index]
    mc = [int(results["mcts_as_x"]["wins"]), int(results["mcts_as_o"]["wins"])]         # type: ignore[index]
    xs, bw = [0, 1], 0.34
    b1 = ax2.bar([x - bw / 2 - 0.01 for x in xs], mm, width=bw, color=th["minimax"])
    b2 = ax2.bar([x + bw / 2 + 0.01 for x in xs], mc, width=bw, color=th["mcts"])
    ymax = max(mm + mc + [5]) * 1.4
    ax2.set_ylim(0, ymax)
    ax2.set_xlim(-0.6, 1.6)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(["Playing first (X)", "Playing second (O)"], fontsize=12, fontweight="bold", color=th["text"])
    ax2.tick_params(axis="x", length=0, pad=6)
    ax2.yaxis.set_visible(False)
    for bar in list(b1) + list(b2):
        v = int(bar.get_height())
        ax2.text(bar.get_x() + bar.get_width() / 2, v + ymax * 0.02, str(v), ha="center", va="bottom",
                 fontsize=13, fontweight="bold", color=th["text"])
    ax2.set_title("Wins by side", loc="left", fontsize=14, fontweight="bold", color=th["text"], pad=10)
    leg = ax2.legend(handles=[Patch(color=th["minimax"], label="Minimax"), Patch(color=th["mcts"], label="MCTS")],
                     loc="upper right", fontsize=11, frameon=False, ncol=2)
    for t in leg.get_texts():
        t.set_color(th["text"])
        t.set_fontweight("bold")

    fig.tight_layout(pad=1.4, h_pad=2.2, w_pad=2.2)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    cw, ch = canvas.get_width_height()
    return pygame.image.frombuffer(bytes(canvas.buffer_rgba()), (cw, ch), "RGBA")


def draw_theme_icon(surface: pygame.Surface, kind: str, center: Tuple[int, int], size: int, color: Color) -> None:
    """Anti-aliased sun / moon glyph (supersampled)."""
    ss = 4
    s = size * ss
    big = pygame.Surface((s, s), pygame.SRCALPHA)
    c = s // 2
    if kind == "sun":
        pygame.draw.circle(big, color, (c, c), int(s * 0.22))
        for k in range(8):
            ang = k * math.pi / 4
            p1 = (c + math.cos(ang) * s * 0.34, c + math.sin(ang) * s * 0.34)
            p2 = (c + math.cos(ang) * s * 0.47, c + math.sin(ang) * s * 0.47)
            pygame.draw.line(big, color, p1, p2, max(2, int(s * 0.07)))
    else:
        pygame.draw.circle(big, color, (c, c), int(s * 0.38))
        pygame.draw.circle(big, (0, 0, 0, 0), (int(c + s * 0.17), int(c - s * 0.13)), int(s * 0.32))
    glyph = pygame.transform.smoothscale(big, (size, size))
    surface.blit(glyph, glyph.get_rect(center=center))


class Button:
    """Clickable button with animated hover. Supports plain, card (accent stripe) and icon-pill styles."""

    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        callback: Callable[[], None],
        active: bool = False,
        subtitle: str = "",
        enabled_fn: Optional[Callable[[], bool]] = None,
        accent: Optional[Color] = None,
        icon: Optional[str] = None,
    ) -> None:
        self.rect = rect
        self.text = text
        self.callback = callback
        self.active = active
        self.subtitle = subtitle
        self.enabled_fn = enabled_fn
        self.accent = accent
        self.icon = icon
        self.hovered = False
        self.hover_t = 0.0

    def is_enabled(self) -> bool:
        return self.enabled_fn is None or self.enabled_fn()

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        small_font: Optional[pygame.font.Font] = None,
    ) -> None:
        enabled = self.is_enabled()
        target = 1.0 if (self.hovered and enabled) else 0.0
        self.hover_t += (target - self.hover_t) * 0.3

        if self.active:
            bg, border = T.btn_active, T.btn_active_border
        else:
            bg = lerp_color(T.btn_default, T.btn_hover, self.hover_t)
            border = lerp_color(T.panel_border, T.grid_thick, self.hover_t)
        if not enabled:
            bg = lerp_color(bg, T.bg_top, 0.55)
            border = lerp_color(border, T.bg_top, 0.5)

        rect = self.rect
        card = self.accent is not None
        if card:
            rect = self.rect.move(0, -int(round(2 * self.hover_t)))
            draw_shadow(surface, rect, 12, 6)
            bg = lerp_color(T.panel_bg, T.btn_hover, self.hover_t * 0.6)
            border = lerp_color(T.panel_border, T.accent_gold, self.hover_t)  # type: ignore[arg-type]

        pygame.draw.rect(surface, bg, rect, border_radius=12 if card else 10)
        pygame.draw.rect(surface, border, rect, width=2, border_radius=12 if card else 10)

        txt_col = T.text_on_accent if self.active else (T.text_primary if enabled else T.text_dim)
        label = font.render(self.text, True, txt_col)

        if card:
            if self.subtitle and small_font is not None:
                sub = small_font.render(self.subtitle, True, T.text_muted)
                total_h = label.get_height() + sub.get_height() + 2
                top = rect.centery - total_h // 2
                surface.blit(label, label.get_rect(midtop=(rect.centerx, top)))
                surface.blit(sub, sub.get_rect(midtop=(rect.centerx, top + label.get_height() + 2)))
            else:
                surface.blit(label, label.get_rect(center=rect.center))
            return

        if self.icon:
            isz = 16
            gap = 8
            total = isz + gap + label.get_width()
            x0 = rect.centerx - total // 2
            icol = T.accent_gold if self.icon == "sun" else T.btn_active
            if self.active:
                icol = T.text_on_accent
            draw_theme_icon(surface, self.icon, (x0 + isz // 2, rect.centery), isz + 4, icol)  # type: ignore[arg-type]
            surface.blit(label, label.get_rect(midleft=(x0 + isz + gap, rect.centery)))
            return

        if self.subtitle and small_font is not None:
            sub = small_font.render(self.subtitle, True, T.text_muted if enabled else T.text_dim)
            total_h = label.get_height() + sub.get_height() + 2
            top = rect.centery - total_h // 2
            surface.blit(label, label.get_rect(midtop=(rect.centerx, top)))
            surface.blit(sub, sub.get_rect(midtop=(rect.centerx, top + label.get_height() + 2)))
        else:
            surface.blit(label, label.get_rect(center=rect.center))


class UltimateTTTGUI:
    """Main Pygame application controller managing screens, background threads, and rendering."""

    def __init__(self, width: int = 1200, height: int = 780) -> None:
        pygame.init()
        T.apply(load_saved_theme())
        pygame.display.set_caption("Ultimate Tic-Tac-Toe: Minimax vs. MCTS")
        self.width = max(MIN_W, width)
        self.height = max(MIN_H, height)
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        # Fonts
        face = "segoeui,helveticaneue,helvetica,arial,dejavusans"
        self.font_title = pygame.font.SysFont(face, 38, bold=True)
        self.font_subtitle = pygame.font.SysFont(face, 21, bold=True)
        self.font_body = pygame.font.SysFont(face, 18, bold=True)
        self.font_small = pygame.font.SysFont(face, 15, bold=True)
        self.font_tiny = pygame.font.SysFont(face, 13, bold=True)
        self.font_big = pygame.font.SysFont(face, 44, bold=True)

        # App navigation state: "MENU", "HVAI", "AIVAI", "TOURNAMENT"
        self.mode: str = "MENU"
        self.running: bool = True

        # Game state & synchronization
        self.game = UltimateTicTacToe()
        self.move_history: List[Tuple[int, int]] = []
        self.session_id: int = 0
        self.ai_thinking: bool = False
        self.pending_ai_move: Optional[Tuple[int, int, Tuple[int, int], Dict[str, object]]] = None
        self.state_lock = threading.Lock()

        # Human vs AI settings
        self.hvai_human_symbol: str = PLAYER_X
        self.hvai_ai_type: str = "Minimax"
        self.hvai_difficulty: str = "Hard"
        self.last_ai_stats: Dict[str, object] = {}

        # AI vs AI settings
        self.aivai_x_type: str = "Minimax"
        self.aivai_x_diff: str = "Hard"
        self.aivai_o_type: str = "MCTS"
        self.aivai_o_diff: str = "Hard"
        self.aivai_paused: bool = False
        self.aivai_delay: float = 0.35
        self.last_move_timestamp: float = 0.0
        self.stats_x: Dict[str, object] = {}
        self.stats_o: Dict[str, object] = {}

        # Tournament state
        self.tournament_runner: Optional[TournamentRunner] = None
        self.tournament_thread: Optional[threading.Thread] = None
        self.tournament_results: Optional[Dict[str, object]] = None
        self.tournament_chart_surface: Optional[pygame.Surface] = None
        self.tournament_minimax_diff: str = "Hard"
        self.tournament_mcts_diff: str = "Hard"
        self._chart_cache: Optional[Tuple[Tuple[int, int, int], pygame.Surface]] = None
        self._chart_busy: bool = False
        self._last_resize: float = 0.0
        self._chart_rendered_for: Optional[Tuple[Tuple[int, int], str]] = None

        # Animation / presentation state
        self.cell_anim: Dict[Tuple[int, int], float] = {}
        self.macro_anim: Dict[int, float] = {}
        self.win_anim_start: Optional[float] = None
        self.overlay_dismissed: bool = False
        self._prev_move_count: int = 0
        self._prev_macro: List[str] = [EMPTY] * 9
        self.mouse_pos: Tuple[int, int] = (0, 0)
        self._cursor_hand: bool = False
        self._bg_cache: Optional[Tuple[Tuple[int, int], pygame.Surface]] = None
        rng = random.Random(7)
        self._decor = [
            (rng.random(), rng.random(), rng.choice((PLAYER_X, PLAYER_O)), rng.uniform(34, 90), rng.uniform(0, 6.28))
            for _ in range(16)
        ]

        # Layout
        self.sidebar_rect = pygame.Rect(0, 0, 0, 0)
        self.board_area = pygame.Rect(0, 0, 0, 0)
        self.board_rect = pygame.Rect(0, 0, 0, 0)
        self.sidebar_labels: List[Tuple[str, int, int, Color]] = []
        self.sidebar_stats_y: int = 0
        self.sidebar_buttons_top: int = 0
        self.menu_title_y: int = 120

        self.buttons: List[Button] = []
        self.overlay_buttons: List[Button] = []
        self._build_ui_buttons()

    # ------------------------------------------------------------------
    # Layout & UI construction
    # ------------------------------------------------------------------
    def _layout(self) -> None:
        self.sidebar_rect = pygame.Rect(
            self.width - SIDEBAR_W - 20, 20, SIDEBAR_W, self.height - 40
        )
        self.board_area = pygame.Rect(20, 20, self.width - SIDEBAR_W - 60, self.height - 40)
        avail_w = self.board_area.w
        avail_h = self.board_area.h - HEADER_H
        size = max(300, min(avail_w, avail_h))
        x = self.board_area.x + (avail_w - size) // 2
        y = self.board_area.y + HEADER_H + (avail_h - size) // 2
        self.board_rect = pygame.Rect(x, y, size, size)

    def _build_ui_buttons(self) -> None:
        """Rebuild interactive buttons for the active screen and current window size."""
        self.buttons.clear()
        self.sidebar_labels = []
        self._layout()

        if self.mode == "MENU":
            bw, gap = 480, 14
            specs = [
                ("Human vs. AI", "Challenge Minimax or MCTS yourself", 66, lambda: self._switch_mode("HVAI"), T.color_x),
                ("AI vs. AI", "Watch two agents battle it out", 66, lambda: self._switch_mode("AIVAI"), T.color_o),
                ("100-Game Tournament", "Automated benchmark with live results & charts", 66,
                 lambda: self._switch_mode("TOURNAMENT"), T.accent_gold),
                ("Exit", "", 48, self._quit_app, None),
            ]
            total = sum(s[2] for s in specs) + gap * (len(specs) - 1)
            y = max(260, (self.height - total) // 2 + 70)
            self.menu_title_y = y - 110
            x = (self.width - bw) // 2
            for label, sub, h, cb, accent in specs:
                self.buttons.append(Button(pygame.Rect(x, y, bw, h), label, cb, subtitle=sub, accent=accent))  # type: ignore[arg-type]
                y += h + gap
            self.buttons.append(self._theme_button(pygame.Rect(self.width - 20 - 104, 20, 104, 36)))
            return

        if self.mode in ("HVAI", "AIVAI"):
            sb = self.sidebar_rect
            px = sb.x + 20
            inner = sb.w - 40            # 360
            col2 = (inner - 10) // 2     # 175
            col3 = (inner - 16) // 3     # 114
            self.sidebar_buttons_top = sb.bottom - 30 - 52

        if self.mode == "HVAI":
            y = sb.y + 62
            self.sidebar_labels.append(("AI ALGORITHM", px, y, T.text_muted))
            for i, algo in enumerate(("Minimax", "MCTS")):
                self.buttons.append(Button(
                    pygame.Rect(px + i * (col2 + 10), y + 20, col2, 34), algo,
                    lambda a=algo: self._set_hvai_algo(a), active=(self.hvai_ai_type == algo)))
            y += 70
            self.sidebar_labels.append(("DIFFICULTY", px, y, T.text_muted))
            for i, diff in enumerate(("Easy", "Medium", "Hard")):
                self.buttons.append(Button(
                    pygame.Rect(px + i * (col3 + 8), y + 20, col3, 34), diff,
                    lambda d=diff: self._set_hvai_diff(d), active=(self.hvai_difficulty == diff)))
            y += 70
            self.sidebar_labels.append(("YOU PLAY AS", px, y, T.text_muted))
            for i, (sym, txt) in enumerate(((PLAYER_X, "X  (moves first)"), (PLAYER_O, "O  (moves second)"))):
                self.buttons.append(Button(
                    pygame.Rect(px + i * (col2 + 10), y + 20, col2, 34), txt,
                    lambda s=sym: self._set_hvai_symbol(s), active=(self.hvai_human_symbol == sym)))
            y += 74
            self.sidebar_stats_y = y

            by = self.sidebar_buttons_top
            col3b = (inner - 16) // 3
            self.buttons.append(Button(pygame.Rect(px, by, col3b, 42), "Undo", self._undo_move,
                                       enabled_fn=self._can_undo))
            self.buttons.append(Button(pygame.Rect(px + col3b + 8, by, col3b, 42), "Restart", self._restart_game))
            self.buttons.append(Button(pygame.Rect(px + 2 * (col3b + 8), by, col3b, 42), "Menu",
                                       lambda: self._switch_mode("MENU")))
            self.buttons.append(self._theme_button(pygame.Rect(sb.right - 20 - 100, sb.y + 14, 100, 32)))

        elif self.mode == "AIVAI":
            y = sb.y + 62
            for sym, col in ((PLAYER_X, T.color_x), (PLAYER_O, T.color_o)):
                is_x = sym == PLAYER_X
                cur_type = self.aivai_x_type if is_x else self.aivai_o_type
                cur_diff = self.aivai_x_diff if is_x else self.aivai_o_diff
                self.sidebar_labels.append((f"PLAYER {sym}", px, y, col))
                for i, algo in enumerate(("Minimax", "MCTS")):
                    setter = self._set_aivai_x if is_x else self._set_aivai_o
                    self.buttons.append(Button(
                        pygame.Rect(px + i * (col2 + 10), y + 20, col2, 30), algo,
                        lambda a=algo, d=cur_diff, f=setter: f(a, d), active=(cur_type == algo)))
                for i, diff in enumerate(("Easy", "Medium", "Hard")):
                    setter = self._set_aivai_x if is_x else self._set_aivai_o
                    self.buttons.append(Button(
                        pygame.Rect(px + i * (col3 + 8), y + 56, col3, 28), diff,
                        lambda d=diff, t=cur_type, f=setter: f(t, d), active=(cur_diff == diff)))
                y += 100
            self.sidebar_labels.append(("MOVE DELAY", px, y, T.text_muted))
            for i, (lbl, val) in enumerate((("0.1 s", 0.1), ("0.35 s", 0.35), ("0.8 s", 0.8))):
                self.buttons.append(Button(
                    pygame.Rect(px + i * (col3 + 8), y + 20, col3, 28), lbl,
                    lambda v=val: self._set_aivai_delay(v), active=(abs(self.aivai_delay - val) < 0.01)))
            y += 64
            self.sidebar_stats_y = y

            by = self.sidebar_buttons_top
            self.buttons.append(Button(
                pygame.Rect(px, by - 46, inner, 36), "Resume" if self.aivai_paused else "Pause",
                self._toggle_aivai_pause, active=self.aivai_paused))
            self.buttons.append(Button(pygame.Rect(px, by, col2, 42), "Restart Match", self._restart_game))
            self.buttons.append(Button(pygame.Rect(px + col2 + 10, by, col2, 42), "Main Menu",
                                       lambda: self._switch_mode("MENU")))
            self.buttons.append(self._theme_button(pygame.Rect(sb.right - 20 - 100, sb.y + 14, 100, 32)))

        elif self.mode == "TOURNAMENT":
            is_running = self.tournament_thread is not None and self.tournament_thread.is_alive()
            diff_w, diff_gap = 92, 8
            group_w = 3 * diff_w + 2 * diff_gap
            menu_btn = pygame.Rect(self.width - 60 - 130, 106, 130, 42)
            start_btn = pygame.Rect(menu_btn.x - 10 - 180, 106, 180, 42)
            x0 = 60
            x1 = x0 + group_w + 36
            for i, diff in enumerate(("Easy", "Medium", "Hard")):
                self.buttons.append(Button(
                    pygame.Rect(x0 + i * (diff_w + diff_gap), 112, diff_w, 34), diff,
                    lambda d=diff: self._set_tournament_diff(d, self.tournament_mcts_diff),
                    active=(self.tournament_minimax_diff == diff),
                    enabled_fn=lambda: not self._tournament_running()))
                self.buttons.append(Button(
                    pygame.Rect(x1 + i * (diff_w + diff_gap), 112, diff_w, 34), diff,
                    lambda d=diff: self._set_tournament_diff(self.tournament_minimax_diff, d),
                    active=(self.tournament_mcts_diff == diff),
                    enabled_fn=lambda: not self._tournament_running()))
            if not is_running:
                self.buttons.append(Button(start_btn, "Start 100 Games", self._start_tournament_background, active=True))
            else:
                self.buttons.append(Button(start_btn, "Stop Tournament", self._stop_tournament))
            self.buttons.append(Button(menu_btn, "Main Menu", self._return_menu_from_tournament))
            self.buttons.append(self._theme_button(pygame.Rect(self.width - 60 - 104, 26, 104, 36)))
            self._tourn_group_x = (x0, x1)

    # ------------------------------------------------------------------
    # State Setters & Mode Transitions
    # ------------------------------------------------------------------
    def _theme_button(self, rect: pygame.Rect) -> Button:
        """Pill showing the mode you will switch TO (sun = go light, moon = go dark)."""
        to_light = T.is_dark
        return Button(rect, "Light" if to_light else "Dark", self._toggle_theme,
                      icon="sun" if to_light else "moon")

    def _toggle_theme(self) -> None:
        new = "light" if T.is_dark else "dark"
        T.apply(new)
        save_theme(new)
        self._bg_cache = None
        self._chart_cache = None
        self._build_ui_buttons()
        self._rerender_chart()

    def _chart_card_rect(self) -> pygame.Rect:
        top = 246
        return pygame.Rect(485, top, self.width - 545, self.height - top - 30)

    def _chart_size(self) -> Tuple[int, int]:
        r = self._chart_card_rect()
        return (max(320, r.w - 24), max(280, r.h - 24))

    def _rerender_chart(self) -> None:
        """Re-draw the Matplotlib tournament chart for the current theme / window size (off the UI thread)."""
        res = self.tournament_results
        if res is None or self._chart_busy:
            return
        self._chart_busy = True
        theme = T.chart_theme()
        size = self._chart_size()

        def work() -> None:
            try:
                surf = render_chart_surface(res, size, theme)
                with self.state_lock:
                    self.tournament_chart_surface = surf
                    self._chart_rendered_for = (size, str(T.name))
            finally:
                self._chart_busy = False

        threading.Thread(target=work, daemon=True).start()

    def _tournament_running(self) -> bool:
        return self.tournament_thread is not None and self.tournament_thread.is_alive()

    def _switch_mode(self, new_mode: str) -> None:
        self.session_id += 1
        self.ai_thinking = False
        self.pending_ai_move = None
        self.mode = new_mode
        if new_mode in ("HVAI", "AIVAI"):
            self._restart_game()
        elif new_mode == "TOURNAMENT":
            self._build_ui_buttons()
            if self.tournament_runner is None and self.tournament_results is None:
                self._start_tournament_background()
        else:
            self._build_ui_buttons()

    def _restart_game(self) -> None:
        self.session_id += 1
        self.ai_thinking = False
        self.pending_ai_move = None
        self.game.reset()
        self.move_history = []
        self.last_ai_stats = {}
        self.stats_x = {}
        self.stats_o = {}
        self.aivai_paused = False
        self.overlay_dismissed = False
        self.last_move_timestamp = time.perf_counter()
        self._build_ui_buttons()

    def _set_hvai_algo(self, algo: str) -> None:
        self.hvai_ai_type = algo
        self._restart_game()

    def _set_hvai_diff(self, diff: str) -> None:
        self.hvai_difficulty = diff
        self._restart_game()

    def _set_hvai_symbol(self, symbol: str) -> None:
        self.hvai_human_symbol = symbol
        self._restart_game()

    def _set_aivai_x(self, algo: str, diff: str) -> None:
        self.aivai_x_type = algo
        self.aivai_x_diff = diff
        self._restart_game()

    def _set_aivai_o(self, algo: str, diff: str) -> None:
        self.aivai_o_type = algo
        self.aivai_o_diff = diff
        self._restart_game()

    def _set_aivai_delay(self, delay: float) -> None:
        self.aivai_delay = delay
        self._build_ui_buttons()

    def _toggle_aivai_pause(self) -> None:
        self.aivai_paused = not self.aivai_paused
        self._build_ui_buttons()

    def _can_undo(self) -> bool:
        if self.mode != "HVAI" or self.ai_thinking:
            return False
        n = 2 if self.game.current_player == self.hvai_human_symbol else 1
        return len(self.move_history) >= n

    def _undo_move(self) -> None:
        """Take back the human's last move (and the AI reply, if any)."""
        if not self._can_undo():
            return
        n = 2 if self.game.current_player == self.hvai_human_symbol else 1
        hist = self.move_history[:-n]
        self.session_id += 1
        self.ai_thinking = False
        self.pending_ai_move = None
        self.game.reset()
        for mv in hist:
            self.game.apply_move(mv)
        self.move_history = hist
        self.overlay_dismissed = False
        self.last_move_timestamp = time.perf_counter()

    def _set_tournament_diff(self, mm_diff: str, mcts_diff: str) -> None:
        if self._tournament_running():
            return
        self.tournament_minimax_diff = mm_diff
        self.tournament_mcts_diff = mcts_diff
        self._build_ui_buttons()

    def _start_tournament_background(self) -> None:
        if self._tournament_running():
            return
        self.tournament_results = None
        self.tournament_chart_surface = None
        self._chart_cache = None
        self.tournament_runner = TournamentRunner(
            minimax_difficulty=self.tournament_minimax_diff,
            mcts_difficulty=self.tournament_mcts_diff,
        )

        def worker() -> None:
            assert self.tournament_runner is not None
            res = self.tournament_runner.run_tournament()
            size, theme = self._chart_size(), T.chart_theme()
            chart_surf = render_chart_surface(res, size, theme)
            with self.state_lock:
                self.tournament_results = res
                self.tournament_chart_surface = chart_surf
                self._chart_rendered_for = (size, str(T.name))

        self.tournament_thread = threading.Thread(target=worker, daemon=True)
        self.tournament_thread.start()
        self._build_ui_buttons()

    def _stop_tournament(self) -> None:
        if self.tournament_runner is not None:
            self.tournament_runner.abort()
        self._build_ui_buttons()

    def _return_menu_from_tournament(self) -> None:
        if self.tournament_runner is not None:
            self.tournament_runner.abort()
        self._switch_mode("MENU")

    def _quit_app(self) -> None:
        if self.tournament_runner is not None:
            self.tournament_runner.abort()
        self.running = False

    # ------------------------------------------------------------------
    # Background AI Execution for Gameplay Modes
    # ------------------------------------------------------------------
    def _create_agent(self, algo_type: str, difficulty: str) -> object:
        if algo_type == "Minimax":
            return MinimaxAI(difficulty=difficulty)
        return MCTSAI(difficulty=difficulty)

    def _trigger_ai_turn_if_needed(self) -> None:
        if self.game.is_terminal() or self.ai_thinking:
            return

        # Apply any completed move from background worker
        with self.state_lock:
            if self.pending_ai_move is not None:
                sess_id, expected_move_cnt, move, stats = self.pending_ai_move
                self.pending_ai_move = None
                if sess_id == self.session_id and expected_move_cnt == self.game.move_count:
                    moving_symbol = self.game.current_player
                    if self.game.is_valid_move(move):
                        self.game.apply_move(move)
                        self.move_history.append(move)
                        self.last_ai_stats = stats
                        if moving_symbol == PLAYER_X:
                            self.stats_x = stats
                        else:
                            self.stats_o = stats
                        self.last_move_timestamp = time.perf_counter()
                return

        if self.mode == "HVAI":
            if self.game.current_player != self.hvai_human_symbol:
                self._spawn_ai_worker(self.hvai_ai_type, self.hvai_difficulty)

        elif self.mode == "AIVAI":
            if self.aivai_paused:
                return
            if time.perf_counter() - self.last_move_timestamp < self.aivai_delay:
                return
            if self.game.current_player == PLAYER_X:
                self._spawn_ai_worker(self.aivai_x_type, self.aivai_x_diff)
            else:
                self._spawn_ai_worker(self.aivai_o_type, self.aivai_o_diff)

    def _spawn_ai_worker(self, algo_type: str, difficulty: str) -> None:
        self.ai_thinking = True
        current_session = self.session_id
        current_move_count = self.game.move_count
        state_copy = self.game.copy()

        def ai_task() -> None:
            try:
                agent = self._create_agent(algo_type, difficulty)
                move = agent.get_move(state_copy)  # type: ignore[attr-defined]
                stats = dict(agent.last_stats)     # type: ignore[attr-defined]
                with self.state_lock:
                    if current_session == self.session_id and move is not None:
                        self.pending_ai_move = (current_session, current_move_count, move, stats)
            finally:
                with self.state_lock:
                    if current_session == self.session_id:
                        self.ai_thinking = False

        threading.Thread(target=ai_task, daemon=True).start()

    # ------------------------------------------------------------------
    # Geometry & input
    # ------------------------------------------------------------------
    def _sub_rect(self, b_idx: int) -> pygame.Rect:
        macro = self.board_rect.width / 3.0
        r, c = divmod(b_idx, 3)
        return pygame.Rect(
            int(self.board_rect.x + c * macro + 5), int(self.board_rect.y + r * macro + 5),
            int(macro - 10), int(macro - 10),
        )

    def _cell_rect(self, b_idx: int, c_idx: int) -> pygame.Rect:
        macro = self.board_rect.width / 3.0
        mr, mc = divmod(b_idx, 3)
        cr, cc = divmod(c_idx, 3)
        pad = macro * BOARD_PAD
        cell = (macro - 2 * pad) / 3.0
        x = self.board_rect.x + mc * macro + pad + cc * cell
        y = self.board_rect.y + mr * macro + pad + cr * cell
        return pygame.Rect(round(x), round(y), round(cell), round(cell))

    def _get_cell_from_mouse(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        br = self.board_rect
        if not br.collidepoint(pos):
            return None
        macro = br.width / 3.0
        rel_x, rel_y = pos[0] - br.x, pos[1] - br.y
        mcol, mrow = int(rel_x // macro), int(rel_y // macro)
        if not (0 <= mcol < 3 and 0 <= mrow < 3):
            return None
        local_x = rel_x - mcol * macro
        local_y = rel_y - mrow * macro
        pad = macro * BOARD_PAD
        if not (pad <= local_x <= macro - pad and pad <= local_y <= macro - pad):
            return None
        cell = (macro - 2 * pad) / 3.0
        ccol = max(0, min(2, int((local_x - pad) // cell)))
        crow = max(0, min(2, int((local_y - pad) // cell)))
        return (mrow * 3 + mcol, crow * 3 + ccol)

    def _human_can_move(self) -> bool:
        return (
            self.mode == "HVAI"
            and not self.ai_thinking
            and not self.game.is_terminal()
            and self.game.current_player == self.hvai_human_symbol
        )

    def _handle_board_click(self, pos: Tuple[int, int]) -> None:
        if not self._human_can_move():
            return
        move = self._get_cell_from_mouse(pos)
        if move is not None and self.game.is_valid_move(move):
            self.game.apply_move(move)
            self.move_history.append(move)
            self.last_move_timestamp = time.perf_counter()

    def _handle_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            if self.mode == "MENU":
                self._quit_app()
            elif self.mode == "TOURNAMENT":
                self._return_menu_from_tournament()
            else:
                self._switch_mode("MENU")
        elif key == pygame.K_t:
            self._toggle_theme()
        elif key == pygame.K_r and self.mode in ("HVAI", "AIVAI"):
            self._restart_game()
        elif key == pygame.K_u and self.mode == "HVAI":
            self._undo_move()
        elif key == pygame.K_SPACE and self.mode == "AIVAI":
            self._toggle_aivai_pause()

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fit(text: str, font: pygame.font.Font, max_w: int) -> str:
        if font.size(text)[0] <= max_w:
            return text
        while text and font.size(text + "...")[0] > max_w:
            text = text[:-1]
        return text + "..."

    def _text(
        self,
        text: str,
        font: pygame.font.Font,
        color: Color,
        pos: Tuple[int, int],
        anchor: str = "topleft",
        max_w: Optional[int] = None,
    ) -> pygame.Rect:
        if max_w is not None:
            text = self._fit(text, font, max_w)
        surf = font.render(text, True, color)
        rect = surf.get_rect(**{anchor: pos})
        self.screen.blit(surf, rect)
        return rect

    def _spaced_text(self, text: str, font: pygame.font.Font, color: Color, center: Tuple[int, int], spacing: int) -> None:
        glyphs = [font.render(ch, True, color) for ch in text]
        total = sum(g.get_width() for g in glyphs) + spacing * (len(glyphs) - 1)
        x = center[0] - total // 2
        for g in glyphs:
            self.screen.blit(g, g.get_rect(midleft=(x, center[1])))
            x += g.get_width() + spacing

    def _get_background(self) -> pygame.Surface:
        key = (self.width, self.height)
        if self._bg_cache is None or self._bg_cache[0] != key:
            surf = pygame.Surface(key)
            for y in range(self.height):
                t = y / max(1, self.height - 1)
                pygame.draw.line(surf, lerp_color(T.bg_top, T.bg_bottom, t), (0, y), (self.width, y))
            vr, vg, vb, va = T.vignette  # type: ignore[misc]
            ov = pygame.Surface(key, pygame.SRCALPHA)
            steps = 70
            full = pygame.Rect(0, 0, *key)
            for i in range(steps):
                a = int(va * (1 - i / steps) ** 2.2)  # type: ignore[operator]
                ring = full.inflate(-2 * i * 3, -2 * i * 3)
                if ring.w < 40 or ring.h < 40:
                    break
                if a > 0:
                    pygame.draw.rect(ov, (vr, vg, vb, a), ring, 3)
            surf.blit(ov, (0, 0))
            self._bg_cache = (key, surf)
        return self._bg_cache[1]

    def _player_label(self, sym: str) -> str:
        if self.mode == "HVAI":
            if sym == self.hvai_human_symbol:
                return "You"
            return f"{self.hvai_ai_type} · {self.hvai_difficulty}"
        if sym == PLAYER_X:
            return f"{self.aivai_x_type} · {self.aivai_x_diff}"
        return f"{self.aivai_o_type} · {self.aivai_o_diff}"

    # ------------------------------------------------------------------
    # Animation bookkeeping
    # ------------------------------------------------------------------
    def _sync_animations(self, now: float) -> None:
        g = self.game
        if g.move_count != self._prev_move_count:
            if g.move_count == self._prev_move_count + 1 and g.last_move is not None:
                self.cell_anim[g.last_move] = now
                for b in range(9):
                    if self._prev_macro[b] == EMPTY and g.macro_board[b] != EMPTY:
                        self.macro_anim[b] = now
            else:
                self.cell_anim.clear()
                self.macro_anim.clear()
            self._prev_move_count = g.move_count
            self._prev_macro = g.macro_board[:]
        if g.winner is not None and self.win_anim_start is None:
            self.win_anim_start = now
        elif g.winner is None:
            self.win_anim_start = None
            self.overlay_dismissed = False

    # ------------------------------------------------------------------
    # Rendering: Menu
    # ------------------------------------------------------------------
    def _draw_menu(self, now: float) -> None:
        # Floating faint symbols for ambiance
        for fx, fy, sym, size, phase in self._decor:
            x = int(fx * self.width + math.sin(now * 0.4 + phase) * 14)
            y = int(fy * self.height + math.cos(now * 0.5 + phase) * 14)
            col = T.color_x if sym == PLAYER_X else T.color_o
            glyph = symbol_surface(sym, int(size), col, 0.09).copy()
            glyph.set_alpha(int(T.decor_alpha))  # type: ignore[arg-type]
            self.screen.blit(glyph, glyph.get_rect(center=(x, y)))

        cx = self.width // 2
        ty = self.menu_title_y
        self._draw_logo((cx, ty - 66), 78, now)
        self._spaced_text("ULTIMATE TIC-TAC-TOE", self.font_title, T.text_primary, (cx, ty), 5)
        # Coloured versus line
        mm = self.font_subtitle.render("Minimax", True, T.color_x)
        vs = self.font_subtitle.render("  vs.  ", True, T.text_muted)
        mc = self.font_subtitle.render("MCTS", True, T.color_o)
        total = mm.get_width() + vs.get_width() + mc.get_width()
        x = cx - total // 2
        for s in (mm, vs, mc):
            self.screen.blit(s, (x, ty + 34))
            x += s.get_width()
        ry = ty + 64
        pygame.draw.line(self.screen, T.accent_gold, (cx - 150, ry), (cx - 14, ry), 2)
        pygame.draw.line(self.screen, T.accent_gold, (cx + 14, ry), (cx + 150, ry), 2)
        pygame.draw.polygon(self.screen, T.accent_gold, [(cx, ry - 6), (cx + 6, ry), (cx, ry + 6), (cx - 6, ry)])
        self._text("Adversarial search & Monte Carlo Tree Search benchmark suite",
                   self.font_small, T.text_muted, (cx, ty + 88), "center")

        for btn in self.buttons:
            btn.draw(self.screen, self.font_subtitle, self.font_small)
        self._text("T toggles light / dark  ·  Esc to quit  ·  the window is resizable", self.font_tiny, T.text_dim,
                   (cx, self.height - 28), "center")

    def _draw_logo(self, center: Tuple[int, int], size: int, now: float) -> None:
        """Tiny 3x3 emblem used on the menu."""
        bob = int(math.sin(now * 1.6) * 3)
        r = pygame.Rect(0, 0, size, size)
        r.center = (center[0], center[1] + bob)
        draw_shadow(self.screen, r, 16, 10)
        pygame.draw.rect(self.screen, T.panel_bg, r, border_radius=18)
        pygame.draw.rect(self.screen, T.btn_active, r, width=3, border_radius=18)
        cell = (size - 20) / 3.0
        gx, gy = r.x + 10, r.y + 10
        for i in (1, 2):
            pygame.draw.line(self.screen, T.grid_thick, (gx + i * cell, gy + 4), (gx + i * cell, gy + 3 * cell - 4), 2)
            pygame.draw.line(self.screen, T.grid_thick, (gx + 4, gy + i * cell), (gx + 3 * cell - 4, gy + i * cell), 2)
        layout = {0: PLAYER_X, 4: PLAYER_O, 8: PLAYER_X, 2: PLAYER_O, 6: PLAYER_X}
        for idx, sym in layout.items():
            rr, cc = divmod(idx, 3)
            g = symbol_surface(sym, int(cell * 0.78), T.color_x if sym == PLAYER_X else T.color_o, 0.16)  # type: ignore[arg-type]
            self.screen.blit(g, g.get_rect(center=(int(gx + (cc + 0.5) * cell), int(gy + (rr + 0.5) * cell))))

    # ------------------------------------------------------------------
    # Rendering: Board
    # ------------------------------------------------------------------
    def _draw_status_header(self, now: float) -> None:
        area = self.board_area
        g = self.game
        chip_gap = 14
        chip_w = (area.w - chip_gap) // 2
        wins = {PLAYER_X: 0, PLAYER_O: 0}
        wins[PLAYER_X], wins[PLAYER_O], _ = g.count_small_boards_won()

        for i, sym in enumerate((PLAYER_X, PLAYER_O)):
            rect = pygame.Rect(area.x + i * (chip_w + chip_gap), area.y, chip_w, 58)
            color = T.color_x if sym == PLAYER_X else T.color_o
            active = (not g.is_terminal()) and g.current_player == sym
            won = g.is_terminal() and g.winner == sym

            draw_shadow(self.screen, rect, 12, 6)
            pygame.draw.rect(self.screen, T.panel_bg, rect, border_radius=12)
            if active:
                pulse = 0.5 + 0.5 * math.sin(now * 4)
                draw_alpha_rect(self.screen, (*color, int(120 + 100 * pulse)), rect, 12, 3)
            elif won:
                pygame.draw.rect(self.screen, T.accent_gold, rect, width=3, border_radius=12)
            else:
                pygame.draw.rect(self.screen, T.panel_border, rect, width=2, border_radius=12)

            icon = symbol_surface(sym, 36, color, 0.13)
            self.screen.blit(icon, icon.get_rect(midleft=(rect.x + 12, rect.centery)))
            self._text(self._player_label(sym), self.font_subtitle, T.text_primary,
                       (rect.x + 58, rect.y + 8), max_w=rect.w - 58 - 70)

            ai_side = not (self.mode == "HVAI" and sym == self.hvai_human_symbol)
            if won:
                sub, sub_col = "Winner!", T.accent_gold
            elif active and ai_side and self.ai_thinking:
                sub, sub_col = "Thinking" + "." * (int(now * 3) % 4), color
            elif active and not ai_side:
                sub, sub_col = "Your turn", color
            elif active:
                sub, sub_col = "To move", color
            else:
                sub, sub_col = "Waiting", T.text_dim
            self._text(sub, self.font_small, sub_col, (rect.x + 58, rect.y + 34))

            self._text(str(wins[sym]), self.font_subtitle, color, (rect.right - 16, rect.y + 8), "topright")
            self._text("boards", self.font_tiny, T.text_dim, (rect.right - 16, rect.y + 34), "topright")

        # Target line
        if g.is_terminal():
            msg = "Game over"
        elif g.active_board is not None and g.macro_board[g.active_board] == EMPTY:
            msg = f"Play in board {g.active_board + 1} (highlighted in gold)"
        else:
            msg = "Free choice: play in any highlighted board"
        self._text(msg, self.font_small, T.text_muted, (area.centerx, area.y + 70), "midtop")

    def _draw_board(self, now: float) -> None:
        g = self.game
        br = self.board_rect
        draw_shadow(self.screen, br, 16, 12)
        pygame.draw.rect(self.screen, T.board_bg, br, border_radius=16)
        pygame.draw.rect(self.screen, T.grid_thick, br, width=int(T.border_w), border_radius=16)  # type: ignore[arg-type]

        valid = set(g.get_Valid_boards()) if not g.is_terminal() else set()
        pulse = 0.5 + 0.5 * math.sin(now * 3.2)
        hovered = self._get_cell_from_mouse(self.mouse_pos) if self._human_can_move() else None
        hover_ok = hovered is not None and g.is_valid_move(hovered)
        win_cells = set(g.winning_line) if (g.winning_line and self.win_anim_start is not None) else set()

        for b in range(9):
            sub = self._sub_rect(b)
            is_valid = b in valid
            pygame.draw.rect(self.screen, T.active_board_bg if is_valid else T.sub_bg, sub, border_radius=12)

            if is_valid:
                draw_alpha_rect(self.screen, (*T.accent_gold, int(25 + 45 * pulse)), sub.inflate(10, 10), 16, 5)
                draw_alpha_rect(self.screen, (*T.accent_gold, int(150 + 90 * pulse)), sub, 12, 2)
            else:
                draw_alpha_rect(self.screen, (*T.grid_thin, 255), sub, 12, 2)

            # Inner grid lines
            c0, c8 = self._cell_rect(b, 0), self._cell_rect(b, 8)
            for i in (1, 2):
                gx = self._cell_rect(b, i).x
                gy = self._cell_rect(b, i * 3).y
                pygame.draw.line(self.screen, T.grid_thin, (gx, c0.y + 4), (gx, c8.bottom - 4), int(T.grid_w))  # type: ignore[arg-type]
                pygame.draw.line(self.screen, T.grid_thin, (c0.x + 4, gy), (c8.right - 4, gy), int(T.grid_w))  # type: ignore[arg-type]

            # Cells
            for c in range(9):
                cell = self._cell_rect(b, c)
                sym = g.small_boards[b][c]
                if sym:
                    color = T.color_x if sym == PLAYER_X else T.color_o
                    if g.last_move == (b, c):
                        draw_alpha_rect(self.screen, (*color, 48), cell.inflate(-5, -5), 8)
                    start = self.cell_anim.get((b, c))
                    p = 1.0 if start is None else clamp01((now - start) / 0.30)
                    size = int(cell.width * 0.80 * (ease_out_back(p) if p < 1 else 1.0))
                    if size > 3:
                        glyph = symbol_surface(sym, size, color)
                        self.screen.blit(glyph, glyph.get_rect(center=cell.center))
                    # ripple
                    if start is not None and now - start < 0.55:
                        rp = (now - start) / 0.55
                        radius = int(cell.width * (0.25 + 0.55 * ease_out_cubic(rp)))
                        layer = pygame.Surface(cell.size, pygame.SRCALPHA)
                        pygame.draw.circle(layer, (*color, int(150 * (1 - rp))),
                                           (cell.width // 2, cell.height // 2), radius, 2)
                        self.screen.blit(layer, cell.topleft)
                elif hover_ok and hovered == (b, c):
                    draw_alpha_rect(self.screen, (*T.hover_cell_bg, 255), cell.inflate(-5, -5), 8)
                    hc = T.color_x if self.hvai_human_symbol == PLAYER_X else T.color_o
                    ghost = symbol_surface(self.hvai_human_symbol, int(cell.width * 0.80), hc).copy()
                    ghost.set_alpha(110)
                    self.screen.blit(ghost, ghost.get_rect(center=cell.center))

            # Captured board overlay
            status = g.macro_board[b]
            if status != EMPTY:
                start = self.macro_anim.get(b)
                p = 1.0 if start is None else clamp01((now - start) / 0.45)
                alpha = int(218 * ease_out_cubic(p))
                if status == PLAYER_X:
                    col = T.color_x
                elif status == PLAYER_O:
                    col = T.color_o
                else:
                    col = T.color_draw
                fill = lerp_color(T.board_bg, col, float(T.capture_mix) * (0.6 if status == DRAW else 1.0))  # type: ignore[arg-type]
                layer = pygame.Surface(sub.size, pygame.SRCALPHA)
                pygame.draw.rect(layer, (*fill, alpha), layer.get_rect(), border_radius=12)
                pygame.draw.rect(layer, (*col, int(alpha * 0.55)), layer.get_rect(), width=2, border_radius=12)
                self.screen.blit(layer, sub.topleft)
                gsize = int(sub.width * 0.70 * (ease_out_back(p) if p < 1 else 1.0))
                if gsize > 4:
                    if status == DRAW:
                        draw_equals(self.screen, sub.center, gsize, col)
                    else:
                        glyph = symbol_surface(status, gsize, col, 0.14)
                        self.screen.blit(glyph, glyph.get_rect(center=sub.center))
                if b in win_cells:
                    wp = 0.5 + 0.5 * math.sin(now * 5)
                    draw_alpha_rect(self.screen, (*T.accent_gold, int(130 + 120 * wp)), sub, 12, 4)

        # Macro dividers
        macro = br.width / 3.0
        for i in (1, 2):
            mx = int(br.x + i * macro)
            my = int(br.y + i * macro)
            pygame.draw.line(self.screen, T.grid_thick, (mx, br.top + 14), (mx, br.bottom - 14), int(T.macro_w))  # type: ignore[arg-type]
            pygame.draw.line(self.screen, T.grid_thick, (br.left + 14, my), (br.right - 14, my), int(T.macro_w))  # type: ignore[arg-type]

        # Animated winning line
        if g.winning_line is not None and self.win_anim_start is not None:
            a, _, c = g.winning_line
            ar, ac = divmod(a, 3)
            cr, cc = divmod(c, 3)
            p = ease_out_cubic((now - self.win_anim_start) / 0.6)
            sx, sy = br.x + (ac + 0.5) * macro, br.y + (ar + 0.5) * macro
            ex, ey = br.x + (cc + 0.5) * macro, br.y + (cr + 0.5) * macro
            tx, ty = sx + (ex - sx) * p, sy + (ey - sy) * p
            pygame.draw.line(self.screen, lerp_color(T.accent_gold, T.bg_top, 0.55), (sx, sy), (tx, ty), 14)
            pygame.draw.line(self.screen, T.accent_gold, (sx, sy), (tx, ty), 8)
            pygame.draw.line(self.screen, lerp_color(T.accent_gold, (255, 255, 255), 0.75), (sx, sy), (tx, ty), 3)
            pygame.draw.circle(self.screen, T.accent_gold, (int(sx), int(sy)), 8)
            pygame.draw.circle(self.screen, T.accent_gold, (int(tx), int(ty)), 8)

    def _draw_game_over_overlay(self, now: float) -> None:
        self.overlay_buttons = []
        g = self.game
        if not g.is_terminal() or self.overlay_dismissed or self.win_anim_start is None:
            return
        delay = 0.9 if g.winning_line else 0.4
        p = clamp01((now - self.win_anim_start - delay) / 0.4)
        if p <= 0:
            return
        br = self.board_rect
        dim = pygame.Surface(br.size, pygame.SRCALPHA)
        pygame.draw.rect(dim, (*T.scrim, int(T.scrim_alpha * p)), dim.get_rect(), border_radius=16)  # type: ignore[misc,operator]
        self.screen.blit(dim, br.topleft)

        w = g.winner
        if w == DRAW:
            title, tcol = "IT'S A DRAW", T.accent_gold
        else:
            tcol = T.color_x if w == PLAYER_X else T.color_o
            if self.mode == "HVAI":
                title = "YOU WIN!" if w == self.hvai_human_symbol else "AI WINS"
            else:
                title = f"PLAYER {w} WINS!"
        x_w, o_w, d_w = g.count_small_boards_won()
        subtitle = f"{g.move_count} moves   ·   boards  X {x_w}  /  O {o_w}  /  draw {d_w}"

        cw = min(440, br.w - 40)
        card = pygame.Rect(0, 0, cw, 230)
        card.center = (br.centerx, br.centery + int((1 - ease_out_cubic(p)) * 24))
        draw_shadow(self.screen, card, 16, 14)
        pygame.draw.rect(self.screen, T.panel_bg, card, border_radius=16)
        pygame.draw.rect(self.screen, tcol, card, width=3, border_radius=16)

        self._text(title, self.font_big, tcol, (card.centerx, card.y + 22), "midtop")
        if w != DRAW:
            self._text(self._player_label(w), self.font_body, T.text_primary, (card.centerx, card.y + 78), "midtop")
        self._text(subtitle, self.font_small, T.text_muted, (card.centerx, card.y + 108), "midtop", card.w - 20)

        bw, bh = (card.w - 40 - 16) // 3, 42
        by = card.bottom - 30 - bh
        specs = [
            ("Play Again", self._restart_game, True),
            ("View Board", self._dismiss_overlay, False),
            ("Menu", lambda: self._switch_mode("MENU"), False),
        ]
        for i, (txt, cb, act) in enumerate(specs):
            btn = Button(pygame.Rect(card.x + 20 + i * (bw + 8), by, bw, bh), txt, cb, active=act)
            btn.hovered = btn.rect.collidepoint(self.mouse_pos)
            self.overlay_buttons.append(btn)
            btn.draw(self.screen, self.font_body)

    def _dismiss_overlay(self) -> None:
        self.overlay_dismissed = True
        self.overlay_buttons = []

    # ------------------------------------------------------------------
    # Rendering: Sidebar
    # ------------------------------------------------------------------
    def _section_title(self, text: str, x: int, y: int, w: int) -> int:
        self._text(text, self.font_tiny, T.accent_gold, (x, y))
        pygame.draw.line(self.screen, T.panel_border, (x, y + 18), (x + w, y + 18), 1)
        return y + 26

    def _draw_sidebar(self) -> None:
        sb = self.sidebar_rect
        draw_shadow(self.screen, sb, 14, 12)
        pygame.draw.rect(self.screen, T.panel_bg, sb, border_radius=14)
        pygame.draw.rect(self.screen, T.panel_border, sb, width=2, border_radius=14)

        px = sb.x + 20
        inner = sb.w - 40
        title = "Human vs. AI" if self.mode == "HVAI" else "AI vs. AI Match"
        self._text(title, self.font_subtitle, T.text_primary, (px, sb.y + 20))

        for text, x, y, col in self.sidebar_labels:
            self._text(text, self.font_tiny, col, (x, y))

        for btn in self.buttons:
            btn.draw(self.screen, self.font_body if btn.rect.height >= 34 else self.font_small)

        # Stats area
        y = self.sidebar_stats_y
        bottom = self.sidebar_buttons_top - (52 if self.mode == "AIVAI" else 14)
        g = self.game
        xw, ow, dr = g.count_small_boards_won()

        y = self._section_title("MATCH", px, y, inner)
        self._text(f"Move {g.move_count}", self.font_small, T.text_primary, (px, y))
        self._text(f"X {xw}  ·  Draw {dr}  ·  O {ow}", self.font_small, T.text_muted, (px + inner, y), "topright")
        y += 22
        bar = pygame.Rect(px, y, inner, 12)
        pygame.draw.rect(self.screen, T.board_bg, bar, border_radius=6)
        seg_x, seg_o, seg_d = int(inner * xw / 9), int(inner * ow / 9), int(inner * dr / 9)
        if seg_x:
            pygame.draw.rect(self.screen, T.color_x, (bar.x, bar.y, seg_x, 12), border_radius=6)
        if seg_d:
            pygame.draw.rect(self.screen, T.color_draw, (bar.x + seg_x, bar.y, seg_d, 12), border_radius=6)
        if seg_o:
            pygame.draw.rect(self.screen, T.color_o, (bar.right - seg_o, bar.y, seg_o, 12), border_radius=6)
        pygame.draw.rect(self.screen, T.panel_border, bar, width=1, border_radius=6)
        y += 28

        header = "AI TELEMETRY" if self.mode == "HVAI" else "LATEST MOVE TELEMETRY"
        y = self._section_title(header, px, y, inner)
        stats = self.last_ai_stats
        rows: List[Tuple[str, str]] = []
        if stats:
            algo = str(stats.get("algorithm", "-"))
            move = stats.get("chosen_move")
            move_str = f"Board {move[0] + 1} · Cell {move[1] + 1}" if isinstance(move, tuple) else "-"
            rows += [
                ("Agent", f"{algo} ({stats.get('difficulty', '-')})"),
                ("Move", move_str),
                ("Think time", f"{float(stats.get('thinking_time', 0.0)) * 1000:.1f} ms"),
            ]
            if algo == "Minimax":
                rows += [
                    ("Search depth", str(stats.get("search_depth", "-"))),
                    ("Nodes evaluated", f"{int(stats.get('nodes_evaluated', 0)):,}"),
                    ("Alpha-beta cutoffs", f"{int(stats.get('cutoffs', 0)):,}"),
                    ("Heuristic score", str(stats.get("eval_score", 0.0))),
                ]
            elif algo == "MCTS":
                rows += [
                    ("Simulations", f"{int(stats.get('simulations', 0)):,}"),
                    ("Tree nodes", f"{int(stats.get('nodes_created', 0)):,}"),
                    ("Win estimate", f"{stats.get('root_move_win_rate', 0.0)}%"),
                ]
            for label, value in rows:
                self._text(label, self.font_small, T.text_muted, (px, y))
                self._text(value, self.font_small, T.text_primary, (px + inner, y), "topright")
                y += 21
        else:
            msg = "Telemetry appears after the AI's first move." if self.mode == "HVAI" else "Waiting for the first move..."
            self._text(msg, self.font_small, T.text_dim, (px, y))
            y += 21
        y += 8

        # Recent moves (only if space remains)
        space = bottom - y
        if space >= 76 and self.move_history:
            y = self._section_title("RECENT MOVES", px, y, inner)
            count = min(5, max(1, (bottom - y) // 20))
            total = len(self.move_history)
            for k in range(count):
                idx = total - 1 - k
                if idx < 0:
                    break
                bb, cc = self.move_history[idx]
                sym = PLAYER_X if idx % 2 == 0 else PLAYER_O
                col = T.color_x if sym == PLAYER_X else T.color_o
                self._text(f"#{idx + 1}", self.font_small, T.text_dim, (px, y))
                self._text(sym, self.font_small, col, (px + 40, y))
                self._text(f"Board {bb + 1} · Cell {cc + 1}", self.font_small, T.text_primary, (px + 62, y))
                y += 20

        hint = "R restart · U undo · T theme · Esc menu" if self.mode == "HVAI" else "R restart · Space pause · T theme · Esc menu"
        self._text(hint, self.font_tiny, T.text_dim, (sb.centerx, sb.bottom - 22), "center")

    # ------------------------------------------------------------------
    # Rendering: Tournament
    # ------------------------------------------------------------------
    def _get_scaled_chart(self, max_w: int, max_h: int) -> Optional[pygame.Surface]:
        with self.state_lock:
            surf = self.tournament_chart_surface
        if surf is None:
            return None
        key = (id(surf), max_w, max_h)
        if self._chart_cache is not None and self._chart_cache[0] == key:
            return self._chart_cache[1]
        sw, sh = surf.get_size()
        if abs(sw - max_w) <= 4 and sh <= max_h + 4:
            self._chart_cache = (key, surf)
            return surf
        s = min(max_w / sw, max_h / sh)
        scaled = pygame.transform.smoothscale(surf, (max(1, int(sw * s)), max(1, int(sh * s))))
        self._chart_cache = (key, scaled)
        return scaled

    def _draw_tournament_screen(self, now: float) -> None:
        self._text("AUTOMATED 100-GAME AI TOURNAMENT", self.font_subtitle, T.text_primary, (60, 30))
        self._text("Games 1-50: Minimax (X) vs. MCTS (O)   ·   Games 51-100: MCTS (X) vs. Minimax (O)",
                   self.font_small, T.text_muted, (60, 62))

        x0, x1 = getattr(self, "_tourn_group_x", (60, 400))
        self._text("MINIMAX DIFFICULTY", self.font_tiny, T.color_x, (x0, 92))
        self._text("MCTS DIFFICULTY", self.font_tiny, T.color_o, (x1, 92))
        for btn in self.buttons:
            btn.draw(self.screen, self.font_body)

        runner = self.tournament_runner
        running = self._tournament_running()
        completed = runner.completed_games if runner else 0
        pct = completed / 100.0

        # Progress bar
        bar = pygame.Rect(60, 168, self.width - 120, 28)
        pygame.draw.rect(self.screen, T.panel_bg, bar, border_radius=10)
        if pct > 0:
            fill = pygame.Rect(60, 168, max(14, int(bar.w * pct)), 28)
            pygame.draw.rect(self.screen, T.btn_active, fill, border_radius=10)
        if running:
            sweep = int(((now * 0.6) % 1.0) * bar.w)
            draw_alpha_rect(self.screen, (255, 255, 255, 40 if T.is_dark else 90), pygame.Rect(60 + sweep - 40, 168, 80, 28), 10)
        pygame.draw.rect(self.screen, T.panel_border, bar, width=2, border_radius=10)
        if runner is None:
            status = "Ready to start"
        else:
            status = f"{completed} / 100 games  ({pct * 100:.0f}%)"
            if running and completed > 0:
                elapsed = time.perf_counter() - runner.start_time
                eta = elapsed / completed * (100 - completed)
                status += f"   ·   ETA ~ {int(eta // 60)}m {int(eta % 60):02d}s"
            elif running:
                status += "   ·   running first game..."
        self._text(status, self.font_small, T.text_primary, bar.center, "center")

        res = self.tournament_results
        if res is None and runner is not None:
            res = runner.get_live_snapshot()

        # Stacked outcome bar
        obar = pygame.Rect(60, 208, self.width - 120, 22)
        pygame.draw.rect(self.screen, T.panel_bg, obar, border_radius=8)
        if res and int(res.get("completed_games", 0)) > 0:
            n = int(res["completed_games"])
            mm, dr, mc = int(res["minimax_wins"]), int(res["draws"]), int(res["mcts_wins"])
            wm, wd = int(obar.w * mm / n), int(obar.w * dr / n)
            wc = obar.w - wm - wd
            segs = [(T.color_x, mm, wm, obar.x), (T.color_draw, dr, wd, obar.x + wm), (T.color_o, mc, wc, obar.x + wm + wd)]
            for col, cnt, w, sx in segs:
                if cnt > 0 and w > 0:
                    pygame.draw.rect(self.screen, col, (sx, obar.y, w, obar.h))
                    if w > 70:
                        name = {T.color_x: "Minimax", T.color_draw: "Draws", T.color_o: "MCTS"}[col]
                        self._text(f"{name} {cnt}", self.font_tiny, T.text_on_color, (sx + w // 2, obar.centery), "center")  # type: ignore[arg-type]
            # re-round the outer corners
            pygame.draw.rect(self.screen, T.bg_top, obar, width=3, border_radius=8)
        pygame.draw.rect(self.screen, T.panel_border, obar, width=1, border_radius=8)

        top = 246
        card_h = self.height - top - 30
        # Left stats card
        card = pygame.Rect(60, top, 405, card_h)
        draw_shadow(self.screen, card, 12, 8)
        pygame.draw.rect(self.screen, T.panel_bg, card, border_radius=12)
        pygame.draw.rect(self.screen, T.panel_border, card, width=2, border_radius=12)

        if res:
            c_games = int(res.get("completed_games", 0))
            final = c_games == 100 and not res.get("aborted", False)
            banner = "FINAL RESULTS (100/100)" if final else f"PARTIAL / IN PROGRESS ({c_games}/100)"
            if res.get("aborted"):
                banner = f"STOPPED EARLY ({c_games}/100)"
            lx, ly, lw = card.x + 18, card.y + 14, card.w - 36
            self._text(banner, self.font_small, T.accent_gold, (lx, ly))
            ly += 24
            self._text(str(res.get("minimax_config", "-")), self.font_tiny, T.color_x, (lx, ly), max_w=lw)
            ly += 17
            self._text(str(res.get("mcts_config", "-")), self.font_tiny, T.color_o, (lx, ly), max_w=lw)
            ly += 26

            def wld(d: Dict[str, int]) -> str:
                return f"{d['wins']}W / {d['losses']}L / {d['draws']}D"

            rows: List[Optional[Tuple[str, str]]] = [
                ("Completed (failed)", f"{c_games}  ({res.get('failed_games', 0)})"),
                ("Minimax wins", f"{res.get('minimax_wins', 0)}  ({res.get('minimax_win_rate', 0.0):.1f}%)"),
                ("MCTS wins", f"{res.get('mcts_wins', 0)}  ({res.get('mcts_win_rate', 0.0):.1f}%)"),
                ("Draws", f"{res.get('draws', 0)}  ({res.get('draw_rate', 0.0):.1f}%)"),
                ("Minimax score rate", f"{res.get('minimax_score_rate', 0.0):.1f}%"),
                ("MCTS score rate", f"{res.get('mcts_score_rate', 0.0):.1f}%"),
                None,
                ("Minimax as X (G1-50)", wld(res["minimax_as_x"])),
                ("Minimax as O (G51-100)", wld(res["minimax_as_o"])),
                ("MCTS as X (G51-100)", wld(res["mcts_as_x"])),
                ("MCTS as O (G1-50)", wld(res["mcts_as_o"])),
                None,
                ("Avg move: Minimax", f"{res.get('minimax_avg_move_ms', 0.0):.1f} ms"),
                ("Avg move: MCTS", f"{res.get('mcts_avg_move_ms', 0.0):.1f} ms"),
                ("Elapsed", f"{res.get('duration_sec', 0.0):.1f} s"),
            ]
            for row in rows:
                if row is None:
                    ly += 10
                    continue
                self._text(row[0], self.font_small, T.text_muted, (lx, ly))
                self._text(row[1], self.font_small, T.text_primary, (lx + lw, ly), "topright")
                ly += 22

        # Right chart card
        chart_rect = self._chart_card_rect()
        draw_shadow(self.screen, chart_rect, 12, 8)
        pygame.draw.rect(self.screen, T.panel_bg, chart_rect, border_radius=12)
        pygame.draw.rect(self.screen, T.panel_border, chart_rect, width=2, border_radius=12)
        want = (self._chart_size(), str(T.name))
        if (self.tournament_results is not None and self._chart_rendered_for != want
                and not self._chart_busy and now - self._last_resize > 0.4):
            self._rerender_chart()
        chart = self._get_scaled_chart(chart_rect.w - 24, chart_rect.h - 24)
        if chart is not None:
            self.screen.blit(chart, chart.get_rect(center=chart_rect.center))
        else:
            msg = "Results chart will appear when the tournament finishes"
            if running:
                msg += "." * (int(now * 2) % 4)
            self._text(msg, self.font_body, T.text_muted, chart_rect.center, "center")

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------
    def _update_cursor(self) -> None:
        want_hand = any(b.rect.collidepoint(self.mouse_pos) and b.is_enabled()
                        for b in self.buttons + self.overlay_buttons)
        if not want_hand and self._human_can_move() and not self.overlay_buttons:
            cell = self._get_cell_from_mouse(self.mouse_pos)
            want_hand = cell is not None and self.game.is_valid_move(cell)
        if want_hand != self._cursor_hand:
            try:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND if want_hand else pygame.SYSTEM_CURSOR_ARROW)
            except (AttributeError, pygame.error):
                pass
            self._cursor_hand = want_hand

    def _handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self._quit_app()
        elif event.type == pygame.VIDEORESIZE:
            self._last_resize = time.perf_counter()
            if event.w < MIN_W or event.h < MIN_H:
                self.width, self.height = max(MIN_W, event.w), max(MIN_H, event.h)
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
            else:
                self.width, self.height = event.w, event.h
                self.screen = pygame.display.get_surface()
            self._build_ui_buttons()
        elif event.type == pygame.KEYDOWN:
            self._handle_key(event.key)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for btn in list(self.overlay_buttons) + list(self.buttons):
                if btn.rect.collidepoint(event.pos) and btn.is_enabled():
                    btn.callback()
                    return
            if not self.overlay_buttons:
                self._handle_board_click(event.pos)

    def _frame(self) -> None:
        """Process input, update state and draw one frame."""
        now = time.perf_counter()
        self.mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            self._handle_event(event)

        for btn in self.buttons + self.overlay_buttons:
            btn.hovered = btn.rect.collidepoint(self.mouse_pos)

        if self.mode in ("HVAI", "AIVAI"):
            self._trigger_ai_turn_if_needed()
            self._sync_animations(now)

        self.screen.blit(self._get_background(), (0, 0))
        if self.mode == "MENU":
            self._draw_menu(now)
        elif self.mode in ("HVAI", "AIVAI"):
            self._draw_status_header(now)
            self._draw_board(now)
            self._draw_game_over_overlay(now)
            self._draw_sidebar()
        elif self.mode == "TOURNAMENT":
            self._draw_tournament_screen(now)

        self._update_cursor()
        pygame.display.flip()

    def run(self) -> None:
        while self.running:
            self._frame()
            # While the tournament / an AI is computing on a background thread, draw fewer frames
            # so the UI does not compete with it for the Python interpreter lock.
            if self.mode == "TOURNAMENT" and self._tournament_running():
                fps = 12
            elif self.ai_thinking:
                fps = 30
            else:
                fps = 60
            self.clock.tick(fps)
        pygame.quit()