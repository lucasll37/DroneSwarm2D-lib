"""Behaviour API and the default policies of the defensive drones.

A *behaviour* is the decision-making policy of a friendly drone. Every step the
drone builds a ``state`` dictionary from its **local** perception (it never sees
the true positions of the enemies) and calls ``behavior.apply(state)``, which
must return ``(info, velocity)``.

PT-BR: API de comportamentos e as políticas padrão dos drones defensivos. A cada
passo o drone monta um dicionário ``state`` com sua percepção **local** e chama
``behavior.apply(state)``, que deve retornar ``(info, velocidade)``.

``state`` keys (arrays keep a leading batch axis of size 1, use ``np.squeeze``):

=====================  ===========================  =====================================
key                    shape                        meaning
=====================  ===========================  =====================================
``drone_id``           ``int``                      id of the drone
``pos``                ``(1, 2)``                   position in pixels
``enemy_intensity``    ``(1, GRID_W, GRID_H)``      recency of enemy detections, 0..1
``enemy_direction``    ``(1, GRID_W, GRID_H, 2)``   unit heading of detected enemies
``friend_intensity``   ``(1, GRID_W, GRID_H)``      recency of friend detections, 0..1
``friend_direction``   ``(1, GRID_W, GRID_H, 2)``   unit heading of detected friends
=====================  ===========================  =====================================

``info`` is a 4-tuple ``(label, target_point, reference_point, extra)``; only
``label`` is required, the other items are optional debug data drawn on screen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pygame

from ..core.settings import (
    AEW_RANGE,
    AEW_SPEED,
    COMMUNICATION_RANGE,
    ENEMY_SPEED,
    EPSILON,
    FRIEND_SPEED,
    HOLD_SPREAD,
    INITIAL_DISTANCE,  # noqa: F401  (kept for backward compatibility of this namespace)
    INTEREST_POINT_CENTER,
    JOYSTICK,
    MIN_COMMUNICATION_HOLD,
    THRESHOLD_PROJECTION,
)
from ..core.utils import (
    active_cells,
    can_intercept,  # noqa: F401  (backward compatibility)
    cell_center,
    intercept_direction,
    pos_to_cell,
)

Info = Tuple[Any, Any, Any, Any]
Cell = Tuple[int, int]


class BehaviorType(StrEnum):
    """Kinds of friendly drone. Members compare equal to their string value.

    PT-BR: Tipos de drone amigo. Os membros são iguais ao seu valor em string
    (``BehaviorType.RADAR == "RADAR"``).

    * ``COMMON`` - mobile interceptor.
    * ``AEW`` - airborne early warning: orbits the interest point with a larger sensor.
    * ``RADAR`` - fixed ground sensor.
    * ``AI`` - reserved for learned policies (behaves like ``COMMON`` in the engine).
    """

    RADAR = "RADAR"
    AEW = "AEW"
    COMMON = "COMMON"
    AI = "AI"


class BaseBehavior(ABC):
    """Base class for every friendly-drone behaviour.

    Subclass it and implement :meth:`apply`.

    PT-BR: Classe base de todos os comportamentos; implemente :meth:`apply`.

    Args:
        behavior_type: A :class:`BehaviorType` (or its string value). It defines
            the drone's role: sensor range, sprite and combat rules.

    Attributes:
        type: The behaviour type as a plain string (``"COMMON"``, ``"AEW"``...).
        joystick: pygame joystick when ``JOYSTICK == "Friend"`` and one is connected.

    Example:
        >>> class Stay(BaseBehavior):
        ...     def __init__(self):
        ...         super().__init__(BehaviorType.COMMON)
        ...     def apply(self, state, joystick_controlled=False):
        ...         return ("STAY", None, None, None), pygame.math.Vector2(0, 0)
    """

    def __init__(self, behavior_type: Union[BehaviorType, str]):
        if not isinstance(behavior_type, BehaviorType):
            try:
                behavior_type = BehaviorType(behavior_type)
            except ValueError:
                raise ValueError(
                    f"behavior_type must be a BehaviorType, got: {behavior_type!r}"
                ) from None

        self.type: str = behavior_type.value
        self.joystick: Optional[pygame.joystick.JoystickType] = None

        if JOYSTICK == "Friend":
            self._init_joystick()

    def _init_joystick(self) -> None:
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Joystick '{self.joystick.get_name()}' connected.")

    @abstractmethod
    def apply(self, state: Dict[str, Any], joystick_controlled: bool = False) -> Tuple[Info, pygame.math.Vector2]:
        """Decide the drone's velocity for this step.

        PT-BR: Decide a velocidade do drone neste passo.

        Args:
            state: Local perception of the drone (see the module docstring).
            joystick_controlled: Whether the drone is being flown manually.

        Returns:
            ``(info, velocity)`` where ``info[0]`` is a short state label shown on
            screen and used for statistics, and ``velocity`` is a ``Vector2`` in
            pixels per step.
        """


def _unpack_state(state: Dict[str, Any]) -> Tuple[pygame.math.Vector2, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Squeeze the batch axis and return ``(pos, fi, ei, fd, ed)``."""
    pos = np.squeeze(state["pos"])
    return (
        pygame.math.Vector2(float(pos[0]), float(pos[1])),
        np.squeeze(state["friend_intensity"]),
        np.squeeze(state["enemy_intensity"]),
        np.squeeze(state["friend_direction"]),
        np.squeeze(state["enemy_direction"]),
    )


def _orbit_velocity(pos: pygame.math.Vector2, radius: float, speed: float) -> pygame.math.Vector2:
    """Tangential velocity around ``INTEREST_POINT_CENTER`` (counter-clockwise on screen)."""
    r_vec = pos - INTEREST_POINT_CENTER
    if r_vec.length() == 0:
        r_vec = pygame.math.Vector2(radius, 0)
    tangent = pygame.math.Vector2(-r_vec.y, r_vec.x).normalize()
    return tangent * speed


class FriendCommonBehaviorDefault(BaseBehavior):
    """Default interceptor policy: greedy assignment + interception + "hold" states.

    PT-BR: Política padrão de interceptação: atribuição gulosa + interceptação +
    estados de espera ("hold").

    Every drone runs the same deterministic algorithm on its own perception, so
    drones that share the same picture reach the same assignment without talking
    about it:

    1. Enemy cells with intensity ``>= enemy_activation_threshold_position`` are
       sorted by distance to the interest point (most urgent first).
    2. Friendly cells with intensity ``>= friend_activation_threshold_position``
       (plus the drone's own cell) are the candidates.
    3. Each enemy gets the closest still-free candidate.
    4. If the drone's own cell was assigned, it flies the interception course
       (``PURSUING``); otherwise it holds (see :meth:`_hold_position`).

    Args:
        friend_activation_threshold_position: Minimum friend intensity.
        enemy_activation_threshold_position: Minimum enemy intensity.
    """

    def __init__(self, friend_activation_threshold_position: float = 0.7,
                 enemy_activation_threshold_position: float = 0.4):
        super().__init__(behavior_type=BehaviorType.COMMON)
        self.friend_activation_threshold_position = friend_activation_threshold_position
        self.enemy_activation_threshold_position = enemy_activation_threshold_position

    def apply(self, state: Dict[str, Any], joystick_controlled: bool = False) -> Tuple[Info, pygame.math.Vector2]:
        pos, friend_intensity, enemy_intensity, _friend_direction, enemy_direction = _unpack_state(state)

        enemy_targets = []
        for cell in active_cells(enemy_intensity, self.enemy_activation_threshold_position):
            target_pos = cell_center(cell)
            enemy_targets.append((cell, target_pos, target_pos.distance_to(INTEREST_POINT_CENTER)))

        my_cell = pos_to_cell(pos)
        my_cell_center = cell_center(my_cell)

        friend_candidates = [(cell, cell_center(cell))
                             for cell in active_cells(friend_intensity, self.friend_activation_threshold_position)]
        # Always include the drone itself so it can be assigned too.
        if not any(cell == my_cell for cell, _ in friend_candidates):
            friend_candidates.append((my_cell, my_cell_center))

        if not enemy_targets:
            return self._hold_position(pos, friend_intensity, enemy_intensity, enemy_direction,
                                       list(friend_candidates))

        enemy_targets.sort(key=lambda t: t[2])

        engagement_assignment: Dict[Tuple[float, float], Cell] = {}
        assigned_friend_cells = set()
        for _cell, target_pos, _ in enemy_targets:
            for candidate_cell, _candidate_pos in sorted(friend_candidates,
                                                         key=lambda x: x[1].distance_to(target_pos)):
                if candidate_cell not in assigned_friend_cells:
                    engagement_assignment[tuple(target_pos)] = candidate_cell
                    assigned_friend_cells.add(candidate_cell)
                    break

        engaged_enemies = [
            (my_cell_center.distance_to(target_pos), cell, target_pos)
            for cell, target_pos, _ in enemy_targets
            if engagement_assignment.get(tuple(target_pos)) == my_cell
        ]

        if engaged_enemies:
            engaged_enemies.sort(key=lambda t: t[0])
            _, chosen_cell, chosen_target_pos = engaged_enemies[0]
            # The matrices store unit headings; the enemy flies at ENEMY_SPEED.
            enemy_velocity = pygame.math.Vector2(*enemy_direction[chosen_cell]) * ENEMY_SPEED
            vel = intercept_direction(pos, FRIEND_SPEED, chosen_target_pos, enemy_velocity)
            if vel.length() > 0:
                return ("PURSUING", None, None, None), vel
            return ("ERROR PURSUING", None, None, None), pygame.math.Vector2(0, 0)

        friends_hold = [(cell, candidate_pos) for cell, candidate_pos in friend_candidates
                        if cell not in assigned_friend_cells]
        return self._hold_position(pos, friend_intensity, enemy_intensity, enemy_direction, friends_hold)

    def _hold_position(
        self,
        pos: pygame.math.Vector2,
        friend_intensity: np.ndarray,
        enemy_intensity: np.ndarray,
        enemy_direction: np.ndarray,
        friends_hold: Optional[List[Tuple[Cell, pygame.math.Vector2]]] = None,
        activation_threshold_position: float = 1,
        enemy_threshold: float = 0.4,
    ) -> Tuple[Info, pygame.math.Vector2]:
        """Behaviour of a drone that is not assigned to any enemy.

        PT-BR: Comportamento do drone que não foi designado a nenhum inimigo.

        * ``HOLD - WAIT`` - at the centre, or nothing better to do: stay still.
        * ``HOLD - RETURN`` - ``MIN_COMMUNICATION_HOLD == 0`` and too far: go back.
        * ``HOLD - INTCPT`` / ``GO HOLD INTCPT`` - an enemy is heading to the
          centre: wait on (or move to) its line of approach.
        * ``HOLD - SPREAD`` - spread tangentially away from other idle friends.
        * ``HOLD - NO ENOUGH COMM`` - fewer than ``MIN_COMMUNICATION_HOLD`` friends
          in communication range: move towards the centre to regroup.
        """
        friends_hold = friends_hold if friends_hold is not None else []

        if pos.distance_to(INTEREST_POINT_CENTER) < EPSILON:
            return ("HOLD - WAIT", None, None, friends_hold), pygame.math.Vector2(0, 0)

        # Fresh (intensity == 1) friend detections inside communication range.
        active_friend_count = 0
        for cell in active_cells(friend_intensity, activation_threshold_position):
            if cell_center(cell).distance_to(pos) < COMMUNICATION_RANGE:
                active_friend_count += 1

        if MIN_COMMUNICATION_HOLD == 0 and pos.distance_to(INTEREST_POINT_CENTER) > INITIAL_DISTANCE:
            vel = (INTEREST_POINT_CENTER - pos).normalize() * FRIEND_SPEED
            return ("HOLD - RETURN", None, None, friends_hold), vel

        if active_friend_count >= MIN_COMMUNICATION_HOLD:
            # Enemies whose heading points (almost) straight at the interest point.
            candidate_intercepts = []
            for cell in active_cells(enemy_intensity, enemy_threshold):
                center = cell_center(cell)
                vec_to_interest = INTEREST_POINT_CENTER - center
                if vec_to_interest.length() == 0:
                    continue
                enemy_dir = pygame.math.Vector2(*enemy_direction[cell])
                if enemy_dir.length() == 0:
                    continue
                enemy_dir = enemy_dir.normalize()
                if enemy_dir.dot(vec_to_interest.normalize()) >= 0.9:
                    candidate_intercepts.append((center.distance_to(INTEREST_POINT_CENTER), center, enemy_dir))

            if candidate_intercepts:
                candidate_intercepts.sort(key=lambda t: t[0])
                _, chosen_cell_center, enemy_dir = candidate_intercepts[0]

                # Projection of the drone onto the enemy's line of approach.
                s = (pos - chosen_cell_center).dot(enemy_dir)
                projection_point = chosen_cell_center + s * enemy_dir
                feasible = (projection_point.distance_to(INTEREST_POINT_CENTER)
                            < chosen_cell_center.distance_to(INTEREST_POINT_CENTER))

                if pos.distance_to(projection_point) <= EPSILON and feasible:
                    return ("HOLD - INTCPT", None, None, friends_hold), pygame.math.Vector2(0, 0)

                # A projection behind the enemy is useless: fall back to the centre.
                beyond_centre = (chosen_cell_center.distance_to(projection_point)
                                 > chosen_cell_center.distance_to(INTEREST_POINT_CENTER))
                defensive_point = INTEREST_POINT_CENTER if beyond_centre else projection_point

                offset = defensive_point - pos
                if offset.length() < THRESHOLD_PROJECTION and feasible and offset.length() > 0:
                    vel = offset.normalize() * FRIEND_SPEED
                    return ("GO HOLD INTCPT", defensive_point, chosen_cell_center, friends_hold), vel

            if HOLD_SPREAD and len(friends_hold) > 2:
                vel = self._spread_velocity(pos, friends_hold)
                return ("HOLD - SPREAD", None, None, friends_hold), vel

            return ("HOLD - WAIT", None, None, friends_hold), pygame.math.Vector2(0, 0)

        vel = (INTEREST_POINT_CENTER - pos).normalize() * FRIEND_SPEED
        return ("HOLD - NO ENOUGH COMM", None, None, friends_hold), vel

    @staticmethod
    def _spread_velocity(pos: pygame.math.Vector2,
                         friends_hold: List[Tuple[Cell, pygame.math.Vector2]]) -> pygame.math.Vector2:
        """Tangential part of the repulsion from the other idle friends.

        Only the component perpendicular to the radius is kept, so drones slide
        around the interest point instead of drifting away from it.
        """
        repulsion = pygame.math.Vector2(0, 0)
        my_cell = pos_to_cell(pos)
        for cell, candidate_pos in friends_hold:
            if cell == my_cell:
                continue
            delta = pos - candidate_pos
            if delta.length() > 0:
                repulsion += delta.normalize()

        radial = pos - INTEREST_POINT_CENTER
        radial = radial.normalize() if radial.length() > 0 else pygame.math.Vector2(1, 0)
        direction = repulsion - repulsion.dot(radial) * radial
        if direction.length() > 1:
            direction = direction.normalize()
        return direction * FRIEND_SPEED


class FriendAEWBehaviorDefault(BaseBehavior):
    """Airborne early warning: orbit the interest point at ``AEW_RANGE``.

    PT-BR: Alerta aéreo antecipado: orbita o ponto de interesse a ``AEW_RANGE``.
    """

    def __init__(self, friend_activation_threshold_position: float = 0.7,
                 enemy_activation_threshold_position: float = 0.4):
        super().__init__(behavior_type=BehaviorType.AEW)
        self.friend_activation_threshold_position = friend_activation_threshold_position
        self.enemy_activation_threshold_position = enemy_activation_threshold_position

    def apply(self, state: Dict[str, Any], joystick_controlled: bool = False) -> Tuple[Info, pygame.math.Vector2]:
        pos = _unpack_state(state)[0]
        return ("AEW", None, None, None), _orbit_velocity(pos, AEW_RANGE, AEW_SPEED)


class FriendRadarBehaviorDefault(BaseBehavior):
    """Ground radar: never moves.

    PT-BR: Radar terrestre: nunca se move.
    """

    def __init__(self, friend_activation_threshold_position: float = 0.7,
                 enemy_activation_threshold_position: float = 0.4):
        super().__init__(behavior_type=BehaviorType.RADAR)
        self.friend_activation_threshold_position = friend_activation_threshold_position
        self.enemy_activation_threshold_position = enemy_activation_threshold_position

    def apply(self, state: Dict[str, Any], joystick_controlled: bool = False) -> Tuple[Info, pygame.math.Vector2]:
        return ("RADAR", None, None, None), pygame.math.Vector2(0, 0)
