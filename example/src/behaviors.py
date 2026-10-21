"""Custom behaviours for the friendly drones of this project.

PT-BR: Comportamentos customizados dos drones amigos deste projeto.

A behaviour subclasses ``BaseBehavior`` and implements ``apply(state)``, which
returns ``(info, velocity)``. ``state`` only contains what *this* drone knows
(its own detections plus what it heard from its neighbours) - never the true
positions of the enemies. See ``TUTORIAL.md`` for the full description.

Three behaviours are defined here, one per drone role:

* ``FriendCommonBehavior`` - intercept the most urgent enemy, otherwise patrol;
* ``FriendAEWBehavior`` - orbit the interest point at ``AEW_RANGE``;
* ``FriendRadarBehavior`` - stay still.

Import this module only *after* ``DroneSwarm2D.init(...)``.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pygame

from DroneSwarm2D.behaviorsDefault import BaseBehavior, BehaviorType
from DroneSwarm2D.core import settings
from DroneSwarm2D.core.utils import active_cells, cell_center, intercept_direction

Info = Tuple[Any, Any, Any, Any]


def orbit_velocity(pos: pygame.math.Vector2, radius: float, speed: float, k_radial: float = 0.0) -> pygame.math.Vector2:
    """Velocity that circles the interest point.

    PT-BR: Velocidade que circula o ponto de interesse.

    Args:
        pos: Current position.
        radius: Desired orbit radius.
        speed: Tangential speed.
        k_radial: Gain of the radial correction that pulls the drone back to
            ``radius`` (0 disables it).
    """
    r_vec = pos - settings.INTEREST_POINT_CENTER
    if r_vec.length() == 0:
        r_vec = pygame.math.Vector2(radius, 0)
    radial_correction = k_radial * (radius - r_vec.length()) * r_vec.normalize()
    tangent = pygame.math.Vector2(-r_vec.y, r_vec.x).normalize()
    return tangent * speed + radial_correction


class FriendCommonBehavior(BaseBehavior):
    """Chase the detected enemy closest to the interest point; patrol when there is none.

    PT-BR: Persegue o inimigo detectado mais próximo do ponto de interesse; sem
    ameaças, patrulha em órbita.

    This is intentionally simple: every drone chases the same enemy. Compare it
    with ``FriendCommonBehaviorDefault``, which assigns one drone per enemy.

    Args:
        friend_activation_threshold_position: Minimum intensity to trust a friend cell.
        enemy_activation_threshold_position: Minimum intensity to trust an enemy cell.
        patrol_radius: Radius of the patrol orbit, in pixels.
    """

    def __init__(
        self,
        friend_activation_threshold_position: float = 0.7,
        enemy_activation_threshold_position: float = 0.4,
        patrol_radius: float = 150,
    ) -> None:
        super().__init__(behavior_type=BehaviorType.COMMON)
        self.friend_activation_threshold_position = friend_activation_threshold_position
        self.enemy_activation_threshold_position = enemy_activation_threshold_position
        self.patrol_radius = patrol_radius

    def apply(self, state: Dict[str, Any], joystick_controlled: bool = False) -> Tuple[Info, pygame.math.Vector2]:
        pos = pygame.math.Vector2(*np.squeeze(state["pos"]).tolist())
        enemy_intensity = np.squeeze(state["enemy_intensity"])
        enemy_direction = np.squeeze(state["enemy_direction"])

        # Cells where an enemy was seen recently enough.
        targets = active_cells(enemy_intensity, self.enemy_activation_threshold_position)
        if targets:
            # Most urgent threat: the one closest to the interest point.
            cell = min(targets, key=lambda c: cell_center(c).distance_to(settings.INTEREST_POINT_CENTER))
            # The matrices store unit headings; enemies fly at ENEMY_SPEED.
            enemy_velocity = pygame.math.Vector2(*enemy_direction[cell]) * settings.ENEMY_SPEED
            vel = intercept_direction(pos, settings.FRIEND_SPEED, cell_center(cell), enemy_velocity)
            return ("PURSUING", None, None, None), vel

        vel = orbit_velocity(pos, self.patrol_radius, settings.FRIEND_SPEED, k_radial=0.05)
        return ("PATROLLING", None, None, None), vel


class FriendAEWBehavior(BaseBehavior):
    """Airborne early warning: orbit the interest point at ``AEW_RANGE``.

    PT-BR: Alerta aéreo antecipado: orbita o ponto de interesse a ``AEW_RANGE``.
    """

    def __init__(self) -> None:
        super().__init__(behavior_type=BehaviorType.AEW)

    def apply(self, state: Dict[str, Any], joystick_controlled: bool = False) -> Tuple[Info, pygame.math.Vector2]:
        pos = pygame.math.Vector2(*np.squeeze(state["pos"]).tolist())
        return ("AEW", None, None, None), orbit_velocity(pos, settings.AEW_RANGE, settings.AEW_SPEED)


class FriendRadarBehavior(BaseBehavior):
    """Ground radar: never moves.

    PT-BR: Radar terrestre: nunca se move.
    """

    def __init__(self) -> None:
        super().__init__(behavior_type=BehaviorType.RADAR)

    def apply(self, state: Dict[str, Any], joystick_controlled: bool = False) -> Tuple[Info, pygame.math.Vector2]:
        return ("RADAR", None, None, None), pygame.math.Vector2(0, 0)
