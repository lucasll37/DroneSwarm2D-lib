"""Utility functions: grid geometry, interception maths, matrix helpers and I/O.

PT-BR: Funções utilitárias: geometria da grade, cálculo de interceptação,
utilidades de matrizes, conversão geográfica e carregamento de SVG.

Contents:
    * Grid helpers: :func:`pos_to_cell`, :func:`cell_center`, :func:`active_cells`.
    * Interception: :func:`intercept_direction`, :func:`can_intercept`.
    * Strategy helper: :func:`get_friends_hold`.
    * Matrices: :func:`generate_sparse_matrix`, :func:`smooth_matrix_with_kernel_10x10`.
    * Conversions / I/O: :func:`sim_to_geo`, :func:`load_svg_as_surface`.
    * Debug plotting: :func:`plot_individual_states_matplotlib`.

The drawing helpers ``draw_dashed_circle`` and ``draw_dashed_line`` now live in
:mod:`DroneSwarm2D.core.rendering` and are re-exported here.
"""

from __future__ import annotations

import io
import math
import random
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import cairosvg
import numpy as np
import pygame

from .rendering import draw_dashed_circle, draw_dashed_line  # noqa: F401  (backward compatibility)
from .settings import (
    CELL_SIZE,
    ENEMY_SPEED,
    FRIEND_SPEED,
    GRID_HEIGHT,
    GRID_WIDTH,
    INTEREST_POINT_CENTER,
    PLOT_THRESHOLD,
    SIM_HEIGHT,
    SIM_WIDTH,
)

Cell = Tuple[int, int]

# -----------------------------------------------------------------------------
# Grid helpers
# -----------------------------------------------------------------------------

def pos_to_cell(
    pos: pygame.math.Vector2,
    cell_size: float = CELL_SIZE,
    grid_width: int = GRID_WIDTH,
    grid_height: int = GRID_HEIGHT,
) -> Cell:
    """Convert a position in pixels to the ``(x, y)`` index of its grid cell.

    PT-BR: Converte uma posição em pixels para o índice ``(x, y)`` da célula.

    Indices are clamped to ``[0, grid_width - 1]`` x ``[0, grid_height - 1]``.

    Args:
        pos: Position (anything with ``.x`` and ``.y``).
        cell_size: Cell size in pixels.
        grid_width: Grid width in cells.
        grid_height: Grid height in cells.

    Returns:
        ``(cell_x, cell_y)``.
    """
    x = int(max(0, min(pos.x // cell_size, grid_width - 1)))
    y = int(max(0, min(pos.y // cell_size, grid_height - 1)))
    return (x, y)


def cell_center(cell: Tuple[int, int], cell_size: float = CELL_SIZE) -> pygame.math.Vector2:
    """Return the centre (in pixels) of a grid cell.

    PT-BR: Retorna o centro (em pixels) de uma célula da grade.
    """
    return pygame.math.Vector2((cell[0] + 0.5) * cell_size, (cell[1] + 0.5) * cell_size)


def active_cells(matrix: np.ndarray, threshold: float) -> List[Cell]:
    """Return the cells whose value is ``>= threshold``, in row-major order.

    This is the vectorised equivalent of iterating with ``np.ndenumerate`` and
    keeping the cells above the threshold, and is orders of magnitude faster.

    PT-BR: Retorna as células com valor ``>= threshold`` (em ordem de linhas),
    equivalente vetorizado de percorrer a matriz com ``np.ndenumerate``.
    """
    rows, cols = np.nonzero(matrix >= threshold)
    return list(zip(rows.tolist(), cols.tolist(), strict=True))


def sim_to_geo(pos_x: float, pos_y: float) -> Tuple[float, float]:
    """Convert simulation pixels to ``(longitude, latitude)``.

    The simulation rectangle is mapped linearly onto the box defined by the
    ``GEO_TOP_LEFT`` and ``GEO_BOTTOM_RIGHT`` settings.

    PT-BR: Converte coordenadas da simulação para ``(longitude, latitude)``.
    """
    from .settings import GEO_BOTTOM_RIGHT, GEO_TOP_LEFT

    lon_left, lat_top = GEO_TOP_LEFT
    lon_right, lat_bottom = GEO_BOTTOM_RIGHT
    lon = lon_left + (pos_x / SIM_WIDTH) * (lon_right - lon_left)
    lat = lat_top + (pos_y / SIM_HEIGHT) * (lat_bottom - lat_top)
    return lon, lat


def load_svg_as_surface(svg_path: str) -> pygame.Surface:
    """Rasterise an SVG file into a pygame surface with per-pixel alpha.

    PT-BR: Converte um arquivo SVG em uma superfície pygame com transparência.

    Note:
        Requires an initialised display (``convert_alpha``).
    """
    png_data: bytes = cairosvg.svg2png(url=svg_path)
    return pygame.image.load(io.BytesIO(png_data)).convert_alpha()


# -----------------------------------------------------------------------------
# Interception
# -----------------------------------------------------------------------------

def _interception_time(
    chaser_pos: pygame.math.Vector2,
    chaser_speed: float,
    target_pos: pygame.math.Vector2,
    target_vel: pygame.math.Vector2,
) -> Optional[float]:
    """Smallest ``t > 0`` solving ``|r + v t| = s t`` (``None`` if there is none)."""
    r = target_pos - chaser_pos
    a = target_vel.dot(target_vel) - chaser_speed ** 2
    b = 2 * r.dot(target_vel)
    c = r.dot(r)

    if abs(a) < 1e-6:  # Same speed: the equation degenerates to a linear one.
        if abs(b) > 1e-6:
            t = -c / b
            return t if t > 0 else None
        return None

    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return None
    sqrt_disc = math.sqrt(discriminant)
    candidates = [t for t in ((-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)) if t > 0]
    return min(candidates) if candidates else None


def intercept_direction(
    chaser_pos: pygame.math.Vector2,
    chaser_speed: float,
    target_pos: pygame.math.Vector2,
    target_vel: pygame.math.Vector2,
) -> pygame.math.Vector2:
    """Velocity a chaser should fly to intercept a target moving at constant velocity.

    Solves ``|r + target_vel * t| = chaser_speed * t`` (``r = target_pos - chaser_pos``)
    for the earliest ``t > 0``. If there is no solution, the chaser simply heads
    straight to the target's current position.

    PT-BR: Velocidade que o perseguidor deve adotar para interceptar um alvo com
    velocidade constante; sem solução, aponta direto para o alvo.

    Args:
        chaser_pos: Chaser position.
        chaser_speed: Chaser speed (pixels per step).
        target_pos: Target position.
        target_vel: Target velocity (pixels per step).

    Returns:
        Velocity vector of length ``chaser_speed`` (or zero if both positions coincide).
    """
    t = _interception_time(chaser_pos, chaser_speed, target_pos, target_vel)
    if t is None:
        direction = target_pos - chaser_pos
    else:
        direction = (target_pos + target_vel * t) - chaser_pos

    if direction.length() > 0:
        return direction.normalize() * chaser_speed
    return pygame.math.Vector2(0, 0)


def can_intercept(
    chaser_pos: pygame.math.Vector2,
    chaser_speed: float,
    target_pos: pygame.math.Vector2,
    target_vel: pygame.math.Vector2,
    point_of_interest: Optional[pygame.math.Vector2] = None,
) -> bool:
    """Whether the chaser can intercept the target before it reaches ``point_of_interest``.

    PT-BR: Indica se o perseguidor consegue interceptar o alvo antes que ele
    alcance o ponto de interesse.

    Args:
        chaser_pos: Chaser position.
        chaser_speed: Chaser speed.
        target_pos: Target position.
        target_vel: Target velocity. A stationary target returns ``False``.
        point_of_interest: Point being defended (defaults to ``INTEREST_POINT_CENTER``).
    """
    if point_of_interest is None:
        point_of_interest = INTEREST_POINT_CENTER

    r = target_pos - chaser_pos
    a = target_vel.dot(target_vel) - chaser_speed ** 2
    if abs(a) < 1e-6 and abs(2 * r.dot(target_vel)) <= 1e-6:
        # Degenerate case: only "already there" counts as an interception.
        return r.length() == 0

    t = _interception_time(chaser_pos, chaser_speed, target_pos, target_vel)
    if t is None or target_vel.length() == 0:
        return False

    time_to_poi = (point_of_interest - target_pos).length() / target_vel.length()
    return t < time_to_poi


# -----------------------------------------------------------------------------
# Strategy helper
# -----------------------------------------------------------------------------

def get_friends_hold(
    state: Dict[str, Any],
    friend_activation_threshold_position: float = 0.7,
    enemy_activation_threshold_position: float = 0.4,
) -> list:
    """Return the friendly positions that are *not* assigned to any interception.

    Greedy assignment: enemies are processed from the closest to the interest
    point outwards and each one is given the closest free friend that
    :func:`can_intercept` it. Friends left over must hold position.

    PT-BR: Retorna as posições amigas que não foram designadas a nenhuma
    interceptação (devem manter posição), usando uma atribuição gulosa.

    Args:
        state: Behaviour state dictionary (``pos``, ``friend_intensity``,
            ``enemy_intensity``, ``enemy_direction``).
        friend_activation_threshold_position: Minimum intensity for a friend cell.
        enemy_activation_threshold_position: Minimum intensity for an enemy cell.

    Returns:
        List of ``(cell, position)`` tuples.
    """
    pos_arr = np.squeeze(state["pos"])
    pos = pygame.math.Vector2(float(pos_arr[0]), float(pos_arr[1]))
    friend_intensity = np.squeeze(state["friend_intensity"])
    enemy_intensity = np.squeeze(state["enemy_intensity"])
    enemy_direction = np.squeeze(state["enemy_direction"])

    enemy_targets = []
    for cell in active_cells(enemy_intensity, enemy_activation_threshold_position):
        target_pos = cell_center(cell)
        enemy_targets.append((cell, target_pos, target_pos.distance_to(INTEREST_POINT_CENTER)))

    my_cell = pos_to_cell(pos)
    friend_candidates = [(cell, cell_center(cell))
                         for cell in active_cells(friend_intensity, friend_activation_threshold_position)]
    if not any(cell == my_cell for cell, _ in friend_candidates):
        friend_candidates.append((my_cell, cell_center(my_cell)))

    if not enemy_targets:
        return list(friend_candidates)

    enemy_targets.sort(key=lambda t: t[2])
    assigned_friend_cells = set()
    for cell, target_pos, _ in enemy_targets:
        enemy_vel = pygame.math.Vector2(float(enemy_direction[cell][0]), float(enemy_direction[cell][1])) * ENEMY_SPEED
        for candidate_cell, candidate_pos in sorted(friend_candidates, key=lambda x: x[1].distance_to(target_pos)):
            if candidate_cell not in assigned_friend_cells and can_intercept(
                candidate_pos, FRIEND_SPEED, target_pos, enemy_vel, INTEREST_POINT_CENTER
            ):
                assigned_friend_cells.add(candidate_cell)
                break

    return [(cell, candidate_pos) for cell, candidate_pos in friend_candidates
            if cell not in assigned_friend_cells]


# -----------------------------------------------------------------------------
# Matrices
# -----------------------------------------------------------------------------

def generate_sparse_matrix(
    shape: Tuple[int, int] = (GRID_WIDTH, GRID_HEIGHT),
    max_nonzero: int = 20,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random sparse intensity matrix plus a matching matrix of unit directions.

    Used to simulate faulty ("broken") drones that broadcast garbage.
    Up to ``max_nonzero`` cells get an intensity drawn from ``N(1, 1)`` clipped
    to ``[0, 1]`` and a random unit direction.

    PT-BR: Gera uma matriz esparsa aleatória de intensidades e uma matriz de
    direções unitárias (usada para simular drones defeituosos).

    Args:
        shape: Shape of the intensity matrix.
        max_nonzero: Maximum number of non-zero cells.
        seed: Seed for reproducibility. With a seed, private RNGs are used, so the
            global ``random`` / ``numpy.random`` state is left untouched.

    Returns:
        ``(matrix, direction)`` with shapes ``shape`` and ``shape + (2,)``.

    Example:
        >>> matrix, direction = generate_sparse_matrix((50, 50), max_nonzero=10, seed=1)
        >>> matrix.shape, direction.shape
        ((50, 50), (50, 50, 2))
    """
    if seed is not None:
        py_rng: Any = random.Random(seed)
        np_rng: Any = np.random.RandomState(seed)
    else:
        py_rng, np_rng = random, np.random

    matrix = np.zeros(shape)
    direction = np.zeros((shape[0], shape[1], 2), dtype=float)
    total_cells = shape[0] * shape[1]
    nonzero_count = min(py_rng.randint(0, max_nonzero), total_cells)

    for idx in py_rng.sample(range(total_cells), nonzero_count):
        i, j = divmod(idx, shape[1])
        matrix[i, j] = np.clip(np_rng.normal(1, 1), 0, 1)
        vec = pygame.math.Vector2(py_rng.uniform(-1, 1), py_rng.uniform(-1, 1))
        if vec.length() > 0:
            vec = vec.normalize()
        direction[i, j] = (vec.x, vec.y)

    return matrix, direction


@lru_cache(maxsize=32)
def _unit_flat_topped_gaussian_10x10(sigma: float, flat_radius: float) -> np.ndarray:
    x = np.linspace(-4.5, 4.5, 10)
    xx, yy = np.meshgrid(x, x)  # yy varies along rows, xx along columns
    r = np.sqrt(xx ** 2 + yy ** 2)
    bump = np.where(r < flat_radius, 1.0, np.exp(-0.5 * (r / sigma) ** 2))
    bump.setflags(write=False)
    return bump


def symmetrical_flat_topped_gaussian_10x10(value: float, sigma: float, flat_radius: float) -> np.ndarray:
    """10x10 Gaussian kernel with a flat top, scaled by ``value``.

    Coordinates span ``[-4.5, 4.5]`` so the kernel is symmetric. Inside
    ``flat_radius`` the kernel is 1; outside it decays as ``exp(-r²/2σ²)``.

    PT-BR: Kernel gaussiano 10x10 com topo plano, multiplicado por ``value``.
    """
    return value * _unit_flat_topped_gaussian_10x10(float(sigma), float(flat_radius))


def smooth_matrix_with_kernel_10x10(
    matrix: np.ndarray,
    direction: np.ndarray,
    sigma: float = 1.0,
    flat_radius: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Spread every positive cell with a flat-topped Gaussian bump (for plotting).

    For each peak, cells where the bump is larger than the current value take
    the bump value *and* the peak's direction. The input arrays are not modified.

    PT-BR: Espalha cada célula positiva com um "morro" gaussiano (usado nos
    gráficos 3D); as células elevadas herdam a direção do pico.

    Args:
        matrix: 2-D array with values in [0, 1].
        direction: Array with shape ``matrix.shape + (2,)``.
        sigma: Standard deviation of the Gaussian part.
        flat_radius: Radius of the flat top.

    Returns:
        ``(smoothed_matrix, smoothed_direction)``.
    """
    result = np.copy(matrix)
    result_direction = np.copy(direction)
    unit_bump = _unit_flat_topped_gaussian_10x10(float(sigma), float(flat_radius))
    anchor = 5  # kernel_size // 2

    for i, j in np.argwhere(matrix > 0):
        bump = matrix[i, j] * unit_bump

        i_start, j_start = max(0, i - anchor), max(0, j - anchor)
        i_end, j_end = min(matrix.shape[0], i + anchor), min(matrix.shape[1], j + anchor)
        bump_i_start = anchor - (i - i_start)
        bump_j_start = anchor - (j - j_start)

        region = result[i_start:i_end, j_start:j_end]
        bump_region = bump[bump_i_start:bump_i_start + (i_end - i_start),
                           bump_j_start:bump_j_start + (j_end - j_start)]

        mask = bump_region > region
        np.maximum(region, bump_region, out=region)
        result_direction[i_start:i_end, j_start:j_end][mask] = direction[i, j]

    return result, result_direction


# -----------------------------------------------------------------------------
# Debug plotting
# -----------------------------------------------------------------------------

def plot_individual_states_matplotlib(state: Dict[str, Any]) -> None:
    """Open a matplotlib window with a 3-D plot of a drone's enemy detections.

    The surface height is the detection recency and the colour encodes the
    estimated heading (angle in radians). Intended for interactive debugging.

    PT-BR: Abre uma janela matplotlib com o gráfico 3D das detecções de
    inimigos de um drone (altura = recência, cor = direção).

    Args:
        state: Behaviour state dictionary with ``pos``, ``enemy_intensity`` and
            ``enemy_direction`` (``friend_*`` keys are accepted but not plotted).
    """
    import matplotlib.pyplot as plt  # Imported lazily: pyplot is slow to import.

    fig = plt.figure(figsize=(8, 6))
    cmap = plt.cm.hsv
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 0.4))

    x = np.linspace(CELL_SIZE / 2, SIM_WIDTH - CELL_SIZE / 2, GRID_WIDTH)
    y = np.linspace(CELL_SIZE / 2, SIM_HEIGHT - CELL_SIZE / 2, GRID_HEIGHT)
    X, Y = np.meshgrid(x, y)

    Z_smoothed, result_direction = smooth_matrix_with_kernel_10x10(
        state["enemy_intensity"], state["enemy_direction"], sigma=2, flat_radius=1
    )
    if Z_smoothed.shape != X.shape:
        Z_smoothed = Z_smoothed.T
        result_direction = result_direction.transpose(1, 0, 2)

    angle = np.arctan2(result_direction[..., 1], result_direction[..., 0])
    facecolors = cmap((angle + math.pi) / (2 * math.pi))
    facecolors[Z_smoothed < PLOT_THRESHOLD] = [1, 1, 1, 1]

    ax.plot_surface(X, Y, Z_smoothed, facecolors=facecolors, linewidth=0, antialiased=True, shade=False)
    ax.contourf(X, Y, Z_smoothed, zdir="x", offset=ax.get_xlim()[0], cmap="Greys")
    ax.contourf(X, Y, Z_smoothed, zdir="y", offset=ax.get_ylim()[0], cmap="Greys")
    ax.plot_wireframe(X, Y, Z_smoothed, color="black", linewidth=0.2, rstride=1, cstride=1)

    px, py = state["pos"][0], state["pos"][1]
    ax.plot([px, px], [py, py], [0, 1], color="black", linewidth=4, zorder=10)
    ax.plot([ax.get_xlim()[0]] * 2, [py, py], [0, 1], color="black", linewidth=4, zorder=10)
    ax.plot([px, px], [ax.get_ylim()[0]] * 2, [0, 1], color="black", linewidth=4, zorder=10)

    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.set_zlabel("Recency", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, SIM_WIDTH)
    ax.set_ylim(SIM_HEIGHT, 0)
    ax.set_zlim(0, 1.5)

    fig.suptitle("Detection and Position Plot", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.75, 1])

    mappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-math.pi, vmax=math.pi), cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, cax=fig.add_axes([0.85, 0.2, 0.02, 0.7]), label="Angle (rad)")
    cbar.set_ticks([-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi])
    cbar.set_ticklabels(["-π", "-π/2", "0", "+π/2", "+π"])
    plt.show()
