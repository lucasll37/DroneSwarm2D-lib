"""Defensive ("friendly") drone: local perception, ad-hoc communication and motion.

PT-BR: Drone defensivo ("amigo"): percepção local, comunicação ad-hoc e movimento.

Each friendly drone keeps its own picture of the world as grid matrices
(``GRID_WIDTH x GRID_HEIGHT``):

* ``enemy_intensity`` / ``friend_intensity`` - recency of a detection in a cell
  (1 when fresh, multiplied by ``DECAY_FACTOR`` every step);
* ``enemy_direction`` / ``friend_direction`` - unit heading observed in the cell;
* ``enemy_timestamp`` / ``friend_timestamp`` - logical time of the last update of
  the cell, used to decide which information is newer when two drones merge.

Every simulation step (:meth:`FriendDrone.update`) a drone:

1. decays its matrices;
2. chooses its communication neighbours (``N_CONNECTIONS`` nearest, symmetric);
3. senses enemies (directly, or only when ``N_LINE_SIGHT_CROSSING`` friends
   cover them - "passive detection") and friends in range;
4. merges its matrices with its neighbours, ``CICLE_COMM_BY_STEP`` times, each
   message being lost with probability ``MESSAGE_LOSS_PROBABILITY``;
5. asks its behaviour for a velocity and moves.

Timestamps come from a deterministic logical clock (incremented on every drone
update) instead of the wall clock, so a run is fully reproducible from its seed.
"""

from __future__ import annotations

import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame

from .rendering import draw_dashed_circle, draw_dashed_line, draw_fading_trajectory, get_font
from .settings import (
    AEW_DETECTION_RANGE,
    CELL_SIZE,
    CICLE_COMM_BY_STEP,
    COMMUNICATION_RANGE,
    DECAY_FACTOR,
    EXTERNAL_RADIUS,
    FRIEND_DETECTION_RANGE,
    FRIEND_SPEED,
    GRID_HEIGHT,
    GRID_WIDTH,
    MESSAGE_LOSS_PROBABILITY,
    N_CONNECTIONS,
    N_LINE_SIGHT_CROSSING,
    RADAR_DETECTION_RANGE,
    SIM_HEIGHT,
    SIM_WIDTH,
    TRIANGULATION_GRANULARITY,
    UPDATE_STATE_BROKEN,
)
from .utils import generate_sparse_matrix, intercept_direction, load_svg_as_surface, pos_to_cell  # noqa: F401

_ASSETS_DIR = Path(__file__).parent.parent / "assets"


@lru_cache(maxsize=64)
def _disk_mask(radius_cells: int) -> np.ndarray:
    """Boolean ``(2r+1, 2r+1)`` mask of the cells within ``r`` cells of the centre."""
    offsets = np.arange(-radius_cells, radius_cells + 1)
    xv, yv = np.meshgrid(offsets, offsets, indexing="ij")
    mask = np.sqrt(xv ** 2 + yv ** 2) <= radius_cells
    mask.setflags(write=False)
    return mask


def _positions(drones: Sequence[Any]) -> np.ndarray:
    """``(n, 2)`` float array with the drones' positions."""
    return np.array([(d.pos.x, d.pos.y) for d in drones], dtype=float).reshape(len(drones), 2)


class FriendDrone:
    """A defensive drone of the swarm.

    PT-BR: Um drone defensivo do enxame.

    Args:
        interest_point_center: Centre of the point being defended.
        position: Initial ``(x, y)`` position in pixels.
        behavior: Decision policy (a :class:`~DroneSwarm2D.behaviorsDefault.BaseBehavior`).
            Defaults to ``FriendCommonBehaviorDefault``.
        fixed: If true the drone never moves (ground radar).
        broken: If true the drone's sensors report random garbage, which it then
            spreads through the network (fault-injection experiments).

    Attributes:
        pos, vel: Position and velocity (``pygame.math.Vector2``, pixels/step).
        drone_id: Unique id within the episode.
        neighbors: Drones it exchanges information with in the current step.
        enemy_intensity, enemy_direction, enemy_timestamp: Enemy perception.
        friend_intensity, friend_direction, friend_timestamp: Friend perception.
        info: Last ``(label, target, reference, extra)`` returned by the behaviour.
        state_history: Number of steps spent in each behaviour label.
        distance_traveled: Total distance flown in pixels.
    """

    # Class-level state (reset by the environment at every episode).
    friend_id_counter: int = 0
    _logical_clock: int = 0

    original_drone_image: pygame.Surface = load_svg_as_surface(f"{_ASSETS_DIR}/drone_0.svg")
    original_broken_drone_image: pygame.Surface = load_svg_as_surface(f"{_ASSETS_DIR}/drone_broken.svg")
    original_aew_image: pygame.Surface = load_svg_as_surface(f"{_ASSETS_DIR}/radar_0.svg")
    original_radar_image: pygame.Surface = load_svg_as_surface(f"{_ASSETS_DIR}/radar_0.svg")

    class_seed: Optional[int] = None
    class_rng: Optional[random.Random] = None
    class_np_rng: Optional[np.random.RandomState] = None

    def __init__(
        self,
        interest_point_center: pygame.math.Vector2,
        position: Tuple[float, float],
        behavior: Optional[Any] = None,
        fixed: bool = False,
        broken: bool = False,
    ) -> None:
        if FriendDrone.class_seed is None:
            FriendDrone.set_class_seed()
        if behavior is None:
            from ..behaviorsDefault import FriendCommonBehaviorDefault
            behavior = FriendCommonBehaviorDefault()

        # Kinematics and role
        self.pos: pygame.math.Vector2 = pygame.math.Vector2(position[0], position[1])
        self.interest_point_center: pygame.math.Vector2 = interest_point_center
        self.behavior: Any = behavior
        self.fixed: bool = fixed
        self.selected: bool = False
        self.joystick_controlled: bool = False
        self.vel: pygame.math.Vector2 = pygame.math.Vector2(0, 0)
        self.orbit_radius: Optional[float] = None
        self.trajectory: List[pygame.math.Vector2] = []
        self.return_to_base: bool = False
        self.info: Tuple[str, Any, Any, Any] = ("", None, None, None)
        self.detection_mode: Optional[str] = None
        self.neighbors: List[FriendDrone] = []

        # Identity
        self.color: Tuple[int, int, int] = (255, 255, 255)
        self.drone_id: int = self._assign_id()
        self.in_election: bool = False
        self.is_leader: bool = False
        self.leader_id: int = self.drone_id
        self.broken: bool = broken

        # Fault injection (broken drones)
        self.timer_state_broken: int = 0
        self.update_state_broken: int = UPDATE_STATE_BROKEN
        self.broken_friend_intensity: Optional[np.ndarray] = None
        self.broken_friend_direction: Optional[np.ndarray] = None
        self.broken_enemy_intensity: Optional[np.ndarray] = None
        self.broken_enemy_direction: Optional[np.ndarray] = None
        self._broken_rng = random.Random((FriendDrone.class_seed or 0) * 1_000_003 + self.drone_id)

        # Tracking of the targets currently in sensor range (keyed by id(object))
        self.aux_enemy_detections: Dict[int, Tuple[int, int]] = {}
        self.aux_friend_detections: Dict[int, Tuple[int, int]] = {}
        self.current_enemy_pos_detection: Dict[int, pygame.math.Vector2] = {}
        self.current_friend_pos_detection: Dict[int, pygame.math.Vector2] = {}

        # Statistics
        self.state_history: Dict[str, int] = {}
        self.current_state: str = ""
        self.total_steps: int = 0
        self.messages_sent: int = 0
        self.distance_traveled: float = 0.0
        self.last_position: pygame.math.Vector2 = self.pos.copy()
        self.active_connections: int = 0
        self.messages_sent_this_cycle: int = 0

        # Passive detection (lines of sight). The fine-grained matrix is huge
        # (GRID * TRIANGULATION_GRANULARITY² cells), so it is only materialised
        # on demand from the list of marked cells.
        self._passive_shape: Tuple[int, int] = (GRID_WIDTH * TRIANGULATION_GRANULARITY,
                                                GRID_HEIGHT * TRIANGULATION_GRANULARITY)
        self._passive_cells: np.ndarray = np.empty((0, 2), dtype=np.intp)
        self._passive_detection_matrix: Optional[np.ndarray] = None
        self.merged_passive_detection_matrix: Optional[np.ndarray] = None
        self.direction_vectors: Dict[str, pygame.math.Vector2] = {}

        # Perception matrices
        self.enemy_intensity: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        self.enemy_direction: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT, 2))
        self.enemy_timestamp: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        self.friend_intensity: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        self.friend_direction: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT, 2))
        self.friend_timestamp: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT))

        self._setup_drone_image()

    # ------------------------------------------------------------------ class API

    @classmethod
    def set_class_seed(cls, seed: Optional[int] = None) -> None:
        """Seed the RNGs shared by all friendly drones (``None`` = random seed).

        PT-BR: Define a semente dos geradores aleatórios compartilhados.
        """
        cls.class_seed = seed if seed is not None else random.randint(0, 10000000)
        cls.class_rng = random.Random(cls.class_seed)
        cls.class_np_rng = np.random.RandomState(cls.class_seed)

    @classmethod
    def reset_clock(cls) -> None:
        """Reset the logical clock used to timestamp detections (new episode)."""
        cls._logical_clock = 0

    @staticmethod
    def _now() -> int:
        return FriendDrone._logical_clock

    # ------------------------------------------------------------------ properties

    @property
    def behavior_type(self) -> Optional[str]:
        """Role of the drone (``"COMMON"``, ``"AEW"``, ``"RADAR"``...), from its behaviour."""
        return getattr(self.behavior, "type", None)

    @property
    def passive_detection_matrix(self) -> np.ndarray:
        """Fine grid (``GRID * TRIANGULATION_GRANULARITY``) marking the lines of sight.

        Built lazily from the cells marked by :meth:`update_passive_detection`.
        """
        if self._passive_detection_matrix is None:
            matrix = np.zeros(self._passive_shape)
            if len(self._passive_cells):
                matrix[self._passive_cells[:, 0], self._passive_cells[:, 1]] = 1
            self._passive_detection_matrix = matrix
        return self._passive_detection_matrix

    @passive_detection_matrix.setter
    def passive_detection_matrix(self, value: np.ndarray) -> None:
        self._passive_detection_matrix = value

    # ------------------------------------------------------------------ helpers

    def _setup_drone_image(self) -> None:
        """Pick and scale the sprite according to the drone's role."""
        if self.behavior_type == "RADAR":
            source, scale = self.original_radar_image, 0.03
        elif self.broken:
            source, scale = self.original_broken_drone_image, 0.02
        elif self.joystick_controlled:
            source, scale = self.original_radar_image, 0.03
        else:
            source, scale = self.original_drone_image, 0.02

        desired_width = int(SIM_WIDTH * scale)
        aspect_ratio = source.get_height() / source.get_width()
        self.image: pygame.Surface = pygame.transform.scale(source, (desired_width, int(desired_width * aspect_ratio)))

    def _assign_id(self) -> int:
        current_id = FriendDrone.friend_id_counter
        FriendDrone.friend_id_counter += 1
        return current_id

    def _get_detection_range(self) -> float:
        """Sensor range in pixels for this drone's role."""
        behavior_type = self.behavior_type
        if behavior_type == "AEW":
            return AEW_DETECTION_RANGE
        if behavior_type == "RADAR":
            return RADAR_DETECTION_RANGE
        return FRIEND_DETECTION_RANGE

    # ------------------------------------------------------------------ perception

    def decay_matrices(self) -> None:
        """Multiply both intensity matrices by ``DECAY_FACTOR`` (information ageing)."""
        self.enemy_intensity *= DECAY_FACTOR
        self.friend_intensity *= DECAY_FACTOR

    def _register_detection(
        self,
        target: Any,
        cell: Tuple[int, int],
        current_time: int,
        positions: Dict[int, pygame.math.Vector2],
        last_cells: Dict[int, Tuple[int, int]],
        intensity: np.ndarray,
        direction: np.ndarray,
        timestamp: np.ndarray,
    ) -> None:
        """Write a detection of ``target`` into the given matrices.

        The first time a target is seen only its position is stored (a heading
        needs two observations). Afterwards its cell is set to intensity 1 with
        the observed heading, and the cell it left (if any) is cleared.
        """
        key = id(target)
        if key not in positions:
            positions[key] = target.pos.copy()
            return

        prev_cell = last_cells.get(key)
        if prev_cell is not None and prev_cell != cell:
            intensity[prev_cell] = 0
            direction[prev_cell] = (0, 0)
            timestamp[prev_cell] = current_time

        last_cells[key] = cell
        intensity[cell] = 1.0
        delta = target.pos - positions[key]
        positions[key] = target.pos.copy()
        if delta.length() > 0:
            direction[cell] = tuple(delta.normalize())
        timestamp[cell] = current_time

    def _register_enemy(self, enemy: Any, current_time: int) -> None:
        self._register_detection(
            enemy, pos_to_cell(enemy.pos), current_time,
            self.current_enemy_pos_detection, self.aux_enemy_detections,
            self.enemy_intensity, self.enemy_direction, self.enemy_timestamp,
        )

    def _forget_enemy(self, enemy: Any) -> None:
        key = id(enemy)
        self.current_enemy_pos_detection.pop(key, None)
        self.aux_enemy_detections.pop(key, None)

    def _perform_direct_detection(self, enemy_drones: List[Any], detection_range: float, current_time: int) -> None:
        """Detect every enemy closer than ``detection_range``."""
        for enemy in enemy_drones:
            if self.pos.distance_to(enemy.pos) >= detection_range:
                self._forget_enemy(enemy)
                continue
            self._register_enemy(enemy, current_time)

    @staticmethod
    def _sensor_coverage(friend_drones: Sequence[FriendDrone], enemy_drones: Sequence[Any]) -> np.ndarray:
        """Number of friendly sensors covering each enemy (vectorised)."""
        if not friend_drones or not enemy_drones:
            return np.zeros(len(enemy_drones), dtype=int)
        friends = _positions(friend_drones)
        enemies = _positions(enemy_drones)
        ranges = np.array([f._get_detection_range() for f in friend_drones], dtype=float)
        dx = friends[None, :, 0] - enemies[:, None, 0]
        dy = friends[None, :, 1] - enemies[:, None, 1]
        return (np.sqrt(dx * dx + dy * dy) <= ranges[None, :]).sum(axis=1)

    def update_passive_detection(self, enemy_drones: List[Any]) -> None:
        """Trace a line of sight towards every enemy within sensor range.

        Fills :attr:`direction_vectors` (one unit vector per bearing, used for
        debug drawing) and marks the cells crossed by each line on the fine
        passive-detection grid.

        PT-BR: Traça uma linha de visada para cada inimigo dentro do alcance.
        """
        self.direction_vectors = {}
        self._passive_detection_matrix = None
        detection_range = self._get_detection_range()
        cell_size = CELL_SIZE / TRIANGULATION_GRANULARITY
        n_steps = int(detection_range / cell_size) + 2
        width, height = self._passive_shape
        marked: List[np.ndarray] = []

        for enemy in enemy_drones:
            delta = enemy.pos - self.pos
            distance = delta.length()
            if not (0 < distance <= detection_range):
                continue

            direction = delta.normalize()
            angle = math.atan2(direction.y, direction.x)
            self.direction_vectors[f"{round(angle / 0.01) * 0.01:.2f}"] = direction

            # March along the ray in steps of one fine cell (cumulative sum keeps
            # exactly the same floating-point path as the original loop).
            step = direction * cell_size
            path = np.empty((n_steps + 1, 2))
            path[0] = (self.pos.x, self.pos.y)
            path[1:] = (step.x, step.y)
            points = np.cumsum(path, axis=0)[1:]
            dist = np.sqrt((points[:, 0] - self.pos.x) ** 2 + (points[:, 1] - self.pos.y) ** 2)
            beyond = np.flatnonzero(dist > detection_range)
            if beyond.size:
                points = points[:beyond[0]]

            cells = np.floor_divide(points, cell_size).astype(np.intp)
            inside = (cells[:, 0] >= 0) & (cells[:, 0] < width) & (cells[:, 1] >= 0) & (cells[:, 1] < height)
            marked.append(cells[inside])

        self._passive_cells = np.concatenate(marked) if marked else np.empty((0, 2), dtype=np.intp)

    def update_local_enemy_detection(self, friend_drones: List[FriendDrone], enemy_drones: List[Any]) -> None:
        """Sense enemies according to :attr:`detection_mode`.

        * ``"direct"`` - every enemy within sensor range is detected.
        * ``"triangulation"`` - an enemy is only detected if at least
          ``N_LINE_SIGHT_CROSSING`` friendly sensors cover it.

        Cells in range without a fresh detection are then cleared.

        PT-BR: Detecta inimigos conforme o modo (direto ou por triangulação) e
        limpa as células vazias dentro do alcance.
        """
        current_time = self._now()
        detection_range = self._get_detection_range()

        if self.detection_mode == "direct":
            self._perform_direct_detection(enemy_drones, detection_range, current_time)
        else:
            coverage = self._sensor_coverage(friend_drones, enemy_drones)
            for enemy, count in zip(enemy_drones, coverage, strict=True):
                if count < N_LINE_SIGHT_CROSSING:
                    continue
                if self.pos.distance_to(enemy.pos) >= detection_range:
                    self._forget_enemy(enemy)
                    continue
                self._register_enemy(enemy, current_time)

        self._clean_empty_cells(detection_range, current_time, is_enemy=True)
        if self.broken:
            self._apply_broken_detection(detection_range, is_enemy=True)

    def _sensor_window(self, detection_range: float) -> Tuple[slice, slice, np.ndarray]:
        """Grid window around the drone and the disk mask of its sensor footprint.

        The footprint radius is 80% of the sensor range, in cells.
        """
        radius = int(np.floor(detection_range / CELL_SIZE) * 0.8)
        cx, cy = pos_to_cell(self.pos)
        x_min, x_max = max(cx - radius, 0), min(cx + radius, GRID_WIDTH - 1)
        y_min, y_max = max(cy - radius, 0), min(cy + radius, GRID_HEIGHT - 1)
        disk = _disk_mask(radius)[x_min - (cx - radius):x_max - (cx - radius) + 1,
                                  y_min - (cy - radius):y_max - (cy - radius) + 1]
        return slice(x_min, x_max + 1), slice(y_min, y_max + 1), disk

    def _clean_empty_cells(self, detection_range: float, current_time: int, is_enemy: bool = True) -> None:
        """Clear the cells inside the sensor footprint that hold no fresh detection.

        If a drone looks at a cell and sees nothing there, any older belief about
        that cell is reset (with a new timestamp, so the "nothing here" spreads).
        """
        xs, ys, disk = self._sensor_window(detection_range)
        if is_enemy:
            region_intensity, region_timestamp = self.enemy_intensity[xs, ys], self.enemy_timestamp[xs, ys]
        else:
            region_intensity, region_timestamp = self.friend_intensity[xs, ys], self.friend_timestamp[xs, ys]

        mask_empty = disk & (region_intensity < 1)
        region_intensity[mask_empty] = 0
        region_timestamp[mask_empty] = current_time

    def _new_fake_pattern(self, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Draw a new random (intensity, direction) pattern for a broken drone."""
        return generate_sparse_matrix(shape, max_nonzero=10, seed=self._broken_rng.randrange(2 ** 32))

    def _apply_broken_detection(self, detection_range: float, is_enemy: bool = True) -> None:
        """Overwrite the sensor footprint with random garbage (broken drones only).

        The same fake pattern is reported for ``UPDATE_STATE_BROKEN`` calls; then
        a new one is drawn and the footprint is re-timestamped so the change
        propagates. Each broken drone has its own (seeded) pattern generator.
        """
        xs, ys, disk = self._sensor_window(detection_range)
        if is_enemy:
            intensity, direction, timestamp = self.enemy_intensity, self.enemy_direction, self.enemy_timestamp
            fake_intensity, fake_direction = self.broken_enemy_intensity, self.broken_enemy_direction
        else:
            intensity, direction, timestamp = self.friend_intensity, self.friend_direction, self.friend_timestamp
            fake_intensity, fake_direction = self.broken_friend_intensity, self.broken_friend_direction

        # The footprint is clipped near the borders, so its shape can change.
        if fake_direction is None or fake_intensity.shape != disk.shape:
            fake_intensity, fake_direction = self._new_fake_pattern(disk.shape)

        if self.timer_state_broken < self.update_state_broken:
            np.copyto(intensity[xs, ys], fake_intensity, where=disk)
            np.copyto(direction[xs, ys], fake_direction, where=disk[..., None])
            self.timer_state_broken += 1
        else:
            fake_intensity, fake_direction = self._new_fake_pattern(disk.shape)
            timestamp[xs, ys] = self._now()
            self.timer_state_broken = 0

        if is_enemy:
            self.broken_enemy_intensity, self.broken_enemy_direction = fake_intensity, fake_direction
        else:
            self.broken_friend_intensity, self.broken_friend_direction = fake_intensity, fake_direction

    def update_local_friend_detection(self, friend_drones: List[FriendDrone]) -> None:
        """Sense the other friendly drones within communication range (AEWs excluded).

        PT-BR: Detecta os outros drones amigos dentro do alcance de comunicação.
        """
        current_time = self._now()
        for friend in friend_drones:
            if friend is self or friend.behavior_type == "AEW":
                continue
            if self.pos.distance_to(friend.pos) >= COMMUNICATION_RANGE:
                key = id(friend)
                self.current_friend_pos_detection.pop(key, None)
                self.aux_friend_detections.pop(key, None)
                continue
            self._register_detection(
                friend, pos_to_cell(friend.pos), current_time,
                self.current_friend_pos_detection, self.aux_friend_detections,
                self.friend_intensity, self.friend_direction, self.friend_timestamp,
            )

        self._clean_empty_cells(FRIEND_DETECTION_RANGE, current_time, is_enemy=False)
        if self.broken:
            self._apply_broken_detection(FRIEND_DETECTION_RANGE, is_enemy=False)

    # ------------------------------------------------------------------ communication

    def merge_enemy_matrix(self, neighbor: FriendDrone) -> None:
        """Copy from ``neighbor`` every enemy cell it knows more recently than we do.

        PT-BR: Copia do vizinho as células de inimigos com informação mais recente.
        """
        newer = neighbor.enemy_timestamp > self.enemy_timestamp
        if not newer.any():
            return
        np.copyto(self.enemy_intensity, neighbor.enemy_intensity, where=newer)
        np.copyto(self.enemy_direction, neighbor.enemy_direction, where=newer[..., None])
        np.copyto(self.enemy_timestamp, neighbor.enemy_timestamp, where=newer)

    def merge_friend_matrix(self, neighbor: FriendDrone) -> None:
        """Copy from ``neighbor`` every friend cell it knows more recently than we do."""
        newer = neighbor.friend_timestamp > self.friend_timestamp
        if not newer.any():
            return
        np.copyto(self.friend_intensity, neighbor.friend_intensity, where=newer)
        np.copyto(self.friend_direction, neighbor.friend_direction, where=newer[..., None])
        np.copyto(self.friend_timestamp, neighbor.friend_timestamp, where=newer)

    def update_neighbors(self, all_drones: List[FriendDrone]) -> None:
        """Choose the communication neighbours for this step.

        A drone links to its ``N_CONNECTIONS`` nearest drones within
        ``COMMUNICATION_RANGE`` *and* to every drone that has it among its own
        ``N_CONNECTIONS`` nearest, so links are always symmetric. The resulting
        list is ordered deterministically (own choices first, by distance).

        PT-BR: Escolhe os vizinhos de comunicação: os ``N_CONNECTIONS`` mais
        próximos e quem o escolheu (enlaces simétricos), em ordem determinística.
        """
        n = len(all_drones)
        index = next((k for k, d in enumerate(all_drones) if d is self), None)
        positions = _positions(all_drones)

        if index is None:  # self is not part of the list: no reverse links possible.
            dist = np.sqrt((self.pos.x - positions[:, 0]) ** 2 + (self.pos.y - positions[:, 1]) ** 2)
            candidates = np.flatnonzero(dist < COMMUNICATION_RANGE)
            nearest = candidates[np.argsort(dist[candidates], kind="stable")][:N_CONNECTIONS]
            self.neighbors = [all_drones[k] for k in nearest]
            return

        dx = positions[:, None, 0] - positions[None, :, 0]
        dy = positions[:, None, 1] - positions[None, :, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        in_range = dist < COMMUNICATION_RANGE
        np.fill_diagonal(in_range, False)

        candidates = np.flatnonzero(in_range[index])
        nearest = candidates[np.argsort(dist[index, candidates], kind="stable")][:N_CONNECTIONS]

        # Rank of `self` in every other drone's (stable) distance ordering.
        to_self = dist[:, index][:, None]
        order = np.arange(n)[None, :]
        ahead = in_range & ((dist < to_self) | ((dist == to_self) & (order < index)))
        reverse = in_range[:, index] & (ahead.sum(axis=1) < N_CONNECTIONS)
        reverse[index] = False

        chosen = [all_drones[k] for k in nearest] + [all_drones[k] for k in np.flatnonzero(reverse)]
        self.neighbors = list(dict.fromkeys(chosen))

    def update_neghbors(self, all_drones: List[FriendDrone]) -> None:
        """Deprecated misspelling of :meth:`update_neighbors`, kept for compatibility."""
        self.update_neighbors(all_drones)

    def communication(self, all_drones: List[FriendDrone]) -> None:
        """Merge perception with every neighbour, ``CICLE_COMM_BY_STEP`` times.

        Each exchange is lost with probability ``MESSAGE_LOSS_PROBABILITY``;
        repeated cycles act as retransmissions.

        PT-BR: Funde a percepção com cada vizinho, ``CICLE_COMM_BY_STEP`` vezes,
        com perda de mensagem probabilística.
        """
        connections = 0
        messages = 0
        rng = FriendDrone.class_rng
        for _ in range(CICLE_COMM_BY_STEP):
            for other in self.neighbors:
                connections += 1
                messages += 2
                if rng.random() > MESSAGE_LOSS_PROBABILITY:
                    self.merge_enemy_matrix(other)
                    self.merge_friend_matrix(other)

        self.active_connections += connections
        self.messages_sent_this_cycle += messages

    def broadcast_passive_detection(self) -> None:
        """Sum the neighbours' passive-detection grids (experimental, unused by default)."""
        connections = 0
        messages = 0
        for _ in range(CICLE_COMM_BY_STEP):
            for other in self.neighbors:
                connections += 1
                messages += 1
                self.merged_passive_detection_matrix = self.passive_detection_matrix.copy()
                if FriendDrone.class_rng.random() > MESSAGE_LOSS_PROBABILITY and other is not self:
                    self.merged_passive_detection_matrix += other.passive_detection_matrix

        self.active_connections += connections
        self.messages_sent_this_cycle += messages

    # ------------------------------------------------------------------ motion

    def take_action(self) -> None:
        """Ask the behaviour for a velocity, move, and enforce the flight limits."""
        if self.return_to_base:
            to_base = self.interest_point_center - self.pos
            self.vel = to_base.normalize() * FRIEND_SPEED if to_base.length() > 0 else pygame.math.Vector2(0, 0)
            self.info = ("REGRESS", None, None, None)
        else:
            self.apply_behavior()

        self.pos += self.vel
        self._enforce_simulation_bounds()
        self._enforce_external_radius()

    def _enforce_simulation_bounds(self) -> None:
        """Keep the drone inside the simulation area, bouncing off the borders."""
        if self.pos.x < 0:
            self.pos.x = 0
            self.vel.x = abs(self.vel.x)
        elif self.pos.x > SIM_WIDTH:
            self.pos.x = SIM_WIDTH
            self.vel.x = -abs(self.vel.x)

        if self.pos.y < 0:
            self.pos.y = 0
            self.vel.y = abs(self.vel.y)
        elif self.pos.y > SIM_HEIGHT:
            self.pos.y = SIM_HEIGHT
            self.vel.y = -abs(self.vel.y)

    def _enforce_external_radius(self) -> None:
        """Never let the drone go farther than ``EXTERNAL_RADIUS`` from the interest point."""
        offset = self.pos - self.interest_point_center
        if offset.length() > EXTERNAL_RADIUS:
            self.pos = self.interest_point_center + offset.normalize() * EXTERNAL_RADIUS
            self.vel = pygame.math.Vector2(0, 0)

    def update(
        self,
        enemy_drones: List[Any],
        friend_drones: List[FriendDrone],
        use_triangulation: bool,
        return_to_base: bool = False,
    ) -> None:
        """Advance the drone by one simulation step (perceive, communicate, act).

        PT-BR: Avança o drone um passo (percebe, comunica e age).

        Args:
            enemy_drones: Enemies currently alive.
            friend_drones: Friendly drones currently alive (including this one).
            use_triangulation: Use passive detection instead of direct detection.
            return_to_base: Fly back to the interest point, ignoring the behaviour.
        """
        FriendDrone._logical_clock += 1
        self.total_steps += 1
        self.active_connections = 0
        self.messages_sent_this_cycle = 0
        self.return_to_base = return_to_base
        self.detection_mode = "triangulation" if use_triangulation else "direct"

        self.decay_matrices()
        self.update_neghbors(friend_drones)
        if self.detection_mode == "triangulation":
            self.update_passive_detection(enemy_drones)
        self.update_local_enemy_detection(friend_drones, enemy_drones)
        self.update_local_friend_detection(friend_drones)
        self.communication(friend_drones)
        self.take_action()

        self.distance_traveled += self.pos.distance_to(self.last_position)
        self.last_position = self.pos.copy()

        if len(self.trajectory) > 300:
            self.trajectory.pop(0)
        self.trajectory.append(self.pos.copy())

        if self.current_state:
            self.state_history[self.current_state] = self.state_history.get(self.current_state, 0) + 1

    def get_state_percentages(self) -> Dict[str, float]:
        """Percentage of the steps spent in each behaviour label."""
        if self.total_steps == 0:
            return {}
        return {state: (count / self.total_steps) * 100 for state, count in self.state_history.items()}

    def get_state(self) -> Dict[str, Any]:
        """Build the ``state`` dictionary passed to the behaviour (see ``behaviors.py``)."""
        return {
            "drone_id": self.drone_id,
            "pos": np.array([[self.pos.x, self.pos.y]], dtype=np.float32),
            "friend_intensity": np.expand_dims(self.friend_intensity, axis=0),
            "enemy_intensity": np.expand_dims(self.enemy_intensity, axis=0),
            "friend_direction": np.expand_dims(self.friend_direction, axis=0),
            "enemy_direction": np.expand_dims(self.enemy_direction, axis=0),
        }

    def apply_behavior(self) -> None:
        """Set :attr:`vel` and :attr:`info` from the behaviour (or stay still if fixed)."""
        if self.fixed:
            self.vel = pygame.math.Vector2(0, 0)
            self.info = ("FIXED", None, None, None)
        else:
            self.info, self.vel = self.behavior.apply(self.get_state(), self.joystick_controlled)
        self.current_state = self.info[0] if isinstance(self.info, tuple) else self.info

    # ------------------------------------------------------------------ drawing

    def draw(
        self,
        surface: pygame.Surface,
        show_detection: bool = True,
        show_comm_range: bool = True,
        show_trajectory: bool = False,
        show_debug: bool = False,
    ) -> None:
        """Draw the drone, its ranges, trajectory, id and (optionally) debug info.

        PT-BR: Desenha o drone, seus alcances, trajetória, id e informações de debug.
        """
        if show_trajectory and len(self.trajectory) > 1:
            self._draw_trajectory(surface)

        center = (int(self.pos.x), int(self.pos.y))
        if show_detection or self.broken:
            draw_dashed_circle(surface, (*self.color, 32), center, int(self._get_detection_range()), 5, 5, 1)
        if show_comm_range:
            draw_dashed_circle(surface, (255, 255, 0, 32), center, COMMUNICATION_RANGE, 5, 5, 1)

        surface.blit(self.image, self.image.get_rect(center=center))
        self._draw_drone_id(surface)
        if show_debug:
            self._draw_debug_info(surface)

    def _draw_trajectory(self, surface: pygame.Surface) -> None:
        draw_fading_trajectory(surface, self.trajectory, self.color, decay_rate=0.04)

    def _draw_drone_id(self, surface: pygame.Surface) -> None:
        font = get_font(10)
        x, y = int(self.pos.x), int(self.pos.y)
        label = font.render(f"ID: F{self.drone_id}", True, (255, 255, 255))
        label.set_alpha(128)
        surface.blit(label, (x + 20, y - 20))
        if self.is_leader:
            surface.blit(font.render("LEADER", True, (0, 255, 0)), (x + 20, y - 5))
        if self.selected:
            surface.blit(font.render("GRAPH", True, (0, 255, 0)), (x + 20, y + 10))

    def _draw_debug_info(self, surface: pygame.Surface) -> None:
        if self.detection_mode == "triangulation":
            for direction in self.direction_vectors.values():
                draw_dashed_line(surface, (255, 0, 0, 128), self.pos, self.pos + direction * FRIEND_DETECTION_RANGE,
                                 width=1, dash_length=5, space_length=5)

        if not (self.info and self.info[0]):
            return
        gold = (255, 215, 0)
        label = get_font(10).render(self.info[0], True, gold)
        surface.blit(label, (int(self.pos.x) - 3.5 * len(self.info[0]), int(self.pos.y) + 25))

        if self.info[1] is not None:
            target = (int(self.info[1].x), int(self.info[1].y))
            pygame.draw.circle(surface, gold, target, 4)
            pygame.draw.line(surface, gold, (int(self.pos.x), int(self.pos.y)), target, 2)
        if self.info[2] is not None:
            pygame.draw.line(surface, gold,
                             (int(self.interest_point_center[0]), int(self.interest_point_center[1])),
                             (int(self.info[2].x), int(self.info[2].y)), 2)
