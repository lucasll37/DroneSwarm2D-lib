"""Drawing helpers and UI widgets used by the pygame front-end.

Everything here is purely visual: nothing in this module changes the state of
the simulation. The helpers are written to be cheap enough to run every frame
(cached fonts, sparse heat-maps, cropped trajectory surfaces).

PT-BR: Funções de desenho e widgets da interface pygame. Nada aqui altera o
estado da simulação; as funções usam caches para rodar a cada quadro.

The names defined here used to live in ``AirTrafficEnv`` and ``utils``; they are
still re-exported from those modules for backward compatibility.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame

from .settings import CELL_SIZE, FONT_FAMILY, SIM_HEIGHT, SIM_WIDTH

Color = Tuple[int, ...]

# -----------------------------------------------------------------------------
# Fonts
# -----------------------------------------------------------------------------

_font_cache: Dict[Tuple[int, bool], pygame.font.Font] = {}


def get_font(size: int, bold: bool = False) -> pygame.font.Font:
    """Return a cached ``SysFont(FONT_FAMILY, size, bold)``.

    ``pygame.font.SysFont`` opens the font file on every call, which used to be
    one of the most expensive operations per frame. Fonts are cached and the
    cache is dropped automatically if pygame's font module is re-initialised.

    PT-BR: Retorna uma fonte em cache (criar ``SysFont`` a cada quadro é caro).
    """
    if not pygame.font.get_init():
        _font_cache.clear()
        pygame.font.init()
    key = (size, bold)
    font = _font_cache.get(key)
    if font is None:
        font = pygame.font.SysFont(FONT_FAMILY, size, bold=bold)
        _font_cache[key] = font
    return font


def clear_caches() -> None:
    """Drop every cached pygame object (call after ``pygame.quit()``)."""
    _font_cache.clear()
    _dashed_circle_offsets.cache_clear()


# -----------------------------------------------------------------------------
# Primitive shapes
# -----------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _dashed_circle_offsets(radius: int, dash_length: int, space_length: int) -> Tuple[Tuple[float, ...], ...]:
    """Pre-compute the dash end-points of a dashed circle centred at the origin."""
    circumference = 2 * math.pi * radius
    num_dashes = int(circumference / (dash_length + space_length))
    if num_dashes <= 0:
        return ()
    angle_between = 2 * math.pi / num_dashes
    dash_angle = dash_length / radius
    offsets = []
    for i in range(num_dashes):
        start_angle = i * angle_between
        end_angle = start_angle + dash_angle
        offsets.append((
            radius * math.cos(start_angle), radius * math.sin(start_angle),
            radius * math.cos(end_angle), radius * math.sin(end_angle),
        ))
    return tuple(offsets)


def draw_dashed_circle(
    surface: pygame.Surface,
    color: Color,
    center: Tuple[float, float],
    radius: int,
    dash_length: int = 5,
    space_length: int = 5,
    width: int = 1,
) -> None:
    """Draw a dashed circle.

    PT-BR: Desenha um círculo tracejado.

    Args:
        surface: Target surface.
        color: RGB or RGBA colour.
        center: Circle centre ``(x, y)``.
        radius: Radius in pixels; nothing is drawn if ``radius <= 0``.
        dash_length: Length of each dash in pixels.
        space_length: Gap between dashes in pixels.
        width: Line width in pixels.
    """
    if radius <= 0:
        return
    cx, cy = center[0], center[1]
    draw_line = pygame.draw.line
    for x0, y0, x1, y1 in _dashed_circle_offsets(radius, dash_length, space_length):
        draw_line(surface, color, (cx + x0, cy + y0), (cx + x1, cy + y1), width)


def draw_dashed_line(
    surface: pygame.Surface,
    color: Color,
    start_pos: Any,
    end_pos: Any,
    width: int = 1,
    dash_length: int = 5,
    space_length: int = 5,
) -> None:
    """Draw a dashed line between two points (nothing if they coincide).

    PT-BR: Desenha uma linha tracejada entre dois pontos.
    """
    start = pygame.math.Vector2(start_pos)
    displacement = pygame.math.Vector2(end_pos) - start
    length = displacement.length()
    if length == 0:
        return

    unit = displacement.normalize()
    dash_vector = unit * dash_length
    period = dash_length + space_length
    for i in range(int(length / period)):
        dash_start = start + period * i * unit
        pygame.draw.line(surface, color, dash_start, dash_start + dash_vector, width)


def draw_fading_trajectory(
    surface: pygame.Surface,
    points: Sequence[pygame.math.Vector2],
    color: Tuple[int, int, int],
    decay_rate: float,
    min_alpha: int = 30,
    width: int = 2,
) -> None:
    """Draw a poly-line whose older segments fade out exponentially.

    The lines are drawn on a transparent surface cropped to the trajectory's
    bounding box (instead of a full-screen one), which makes this function
    several times faster without changing a single pixel of the result.

    PT-BR: Desenha a trajetória com segmentos antigos esmaecendo
    exponencialmente, usando uma superfície recortada para ganhar desempenho.
    """
    n = len(points)
    if n < 2:
        return

    xs = [int(p.x) for p in points]
    ys = [int(p.y) for p in points]
    margin = width + 1
    # Bounding box of the trajectory, cropped to the target surface so that lines
    # leaving the surface are clipped exactly where they would be on a full-size layer.
    box = pygame.Rect(min(xs) - margin, min(ys) - margin,
                      max(xs) - min(xs) + 2 * margin + 1, max(ys) - min(ys) + 2 * margin + 1)
    box = box.clip(surface.get_rect())
    if box.width == 0 or box.height == 0:
        return
    left, top = box.x, box.y

    traj_surf = pygame.Surface(box.size, pygame.SRCALPHA)
    r, g, b = color[0], color[1], color[2]
    for i in range(n - 1):
        alpha = max(int(255 * math.exp(-decay_rate * (n - 1 - i))), min_alpha)
        pygame.draw.line(
            traj_surf, (r, g, b, alpha),
            (xs[i] - left, ys[i] - top), (xs[i + 1] - left, ys[i + 1] - top), width,
        )
    surface.blit(traj_surf, (left, top))


# -----------------------------------------------------------------------------
# Simulation layers
# -----------------------------------------------------------------------------

def draw_grid(surface: pygame.Surface) -> None:
    """Draw the simulation grid lines.

    PT-BR: Desenha as linhas da grade.
    """
    grid_color = (15, 15, 15)
    for x in range(0, SIM_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, grid_color, (x, 0), (x, SIM_HEIGHT), 1)
    for y in range(0, SIM_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, grid_color, (0, y), (SIM_WIDTH, y), 1)


def draw_heatmap(surface: pygame.Surface, global_intensity: np.ndarray, base_color: str) -> None:
    """Paint every grid cell whose intensity is above 0.01.

    PT-BR: Pinta as células com intensidade acima de 0,01 ("red", "orange" ou azul).

    Args:
        surface: Target surface.
        global_intensity: ``(GRID_WIDTH, GRID_HEIGHT)`` matrix with values in [0, 1].
        base_color: ``"red"``, ``"orange"`` or anything else for blue.
    """
    for i, j in np.argwhere(global_intensity > 0.01):
        intensity = float(global_intensity[i, j])
        if base_color == "red":
            color: Tuple[int, int, int] = (int(intensity * 255), 0, 0)
        elif base_color == "orange":
            color = (int(intensity * 255), int(intensity * 165), 0)
        else:
            color = (0, 0, int(intensity * 255))
        pygame.draw.rect(surface, color, pygame.Rect(int(i) * CELL_SIZE, int(j) * CELL_SIZE, CELL_SIZE, CELL_SIZE))


def draw_direction(
    surface: pygame.Surface,
    global_intensity: np.ndarray,
    global_direction: np.ndarray,
    threshold: float,
) -> None:
    """Draw a short white tick showing the estimated heading of each active cell.

    PT-BR: Desenha um pequeno traço com a direção estimada de cada célula ativa.
    """
    for i, j in np.argwhere(global_intensity > threshold):
        dir_vec = pygame.math.Vector2(float(global_direction[i, j, 0]), float(global_direction[i, j, 1]))
        if dir_vec.length() > 0:
            dir_vec = dir_vec.normalize() * 5
            center_x = int(i) * CELL_SIZE + CELL_SIZE / 2
            center_y = int(j) * CELL_SIZE + CELL_SIZE / 2
            pygame.draw.line(surface, (255, 255, 255), (center_x, center_y),
                             (center_x + dir_vec.x, center_y + dir_vec.y), 1)


def draw_friend_communication(
    surface: pygame.Surface,
    friend_drones: List[Any],
    show_dashed: bool = True,
) -> None:
    """Draw a dashed line for every active communication link (once per pair).

    PT-BR: Desenha uma linha tracejada para cada enlace de comunicação ativo.
    """
    if not show_dashed:
        return
    for drone in friend_drones:
        for nbr in getattr(drone, "neighbors", []):
            if drone.drone_id < nbr.drone_id:
                draw_dashed_line(surface, (255, 255, 255, 128), drone.pos, nbr.pos,
                                 width=2, dash_length=5, space_length=5)


# -----------------------------------------------------------------------------
# Widgets
# -----------------------------------------------------------------------------

class Button:
    """Clickable button of the control panel; optionally a toggle.

    PT-BR: Botão clicável do painel de controle; opcionalmente um alternador.

    Args:
        rect: ``(x, y, width, height)``.
        text: Label.
        callback: Function called on click.
        toggled: ``None`` for a push button, ``True``/``False`` for a toggle.
        color: Background colour (RGB).
    """

    def __init__(
        self,
        rect: Tuple[int, int, int, int],
        text: str,
        callback: Callable[[], Any],
        toggled: Optional[bool] = None,
        color: Optional[Tuple[int, int, int]] = (70, 70, 70),
    ) -> None:
        self.rect: pygame.Rect = pygame.Rect(rect)
        self.text: str = text
        self.callback = callback
        self.original_color = color
        self.toggled_color: Tuple[int, int, int] = (0, 100, 0)
        self.toggled: Optional[bool] = toggled
        self.font: pygame.font.Font = get_font(11)

    @staticmethod
    def darken_color(color: Tuple[int, int, int], amount: int = 30) -> Tuple[int, ...]:
        """Return ``color`` darkened by ``amount`` on every channel."""
        return tuple(max(c - amount, 0) for c in color)

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the button (darker while hovered if it is a toggle)."""
        if self.toggled is not None:
            base_color = self.toggled_color if self.toggled else self.original_color
            hovered = self.rect.collidepoint(pygame.mouse.get_pos())
            color = self.darken_color(base_color) if hovered else base_color
        else:
            color = self.original_color

        pygame.draw.rect(surface, color, self.rect)
        label = self.font.render(self.text, True, (255, 255, 255))
        surface.blit(label, (self.rect.x + 5, self.rect.y + 10))

    def handle_event(self, event: pygame.event.Event) -> None:
        """Flip the toggle state and fire the callback when clicked."""
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            if self.toggled is not None:
                self.toggled = not self.toggled
            self.callback()
