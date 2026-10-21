"""Offensive ("enemy") drone and its attack patterns.

PT-BR: Drone ofensivo ("inimigo") e seus padrões de ataque.

Enemies spawn on a random border of the simulation area (after a random delay
of up to 100 steps) and fly towards the interest point using one of the
patterns listed in :data:`ENEMY_BEHAVIORS`. When a friendly drone gets within
``ENEMY_DETECTION_RANGE`` the enemy either commits to a *desperate attack*
(straight line, probability = its aggressiveness) or *escapes* for
``ESCAPE_STEPS`` steps. Aggressiveness grows from ``INITIAL_AGGRESSIVENESS`` at
``EXTERNAL_RADIUS`` to 1 at ``INITIAL_DISTANCE`` from the interest point.
"""

from __future__ import annotations

import math
import random
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pygame

from .rendering import draw_dashed_circle, draw_fading_trajectory, get_font
from .settings import (
    ENEMY_COUNT,
    ENEMY_DETECTION_RANGE,
    ENEMY_SPEED,
    ESCAPE_STEPS,
    EXTERNAL_RADIUS,
    INITIAL_AGGRESSIVENESS,
    INITIAL_DISTANCE,
    SIM_HEIGHT,
    SIM_WIDTH,
)
from .utils import load_svg_as_surface

_ASSETS_DIR = Path(__file__).parent.parent / "assets"

#: Patterns drawn at random when ``behavior_type`` is ``None``.
RANDOM_ENEMY_BEHAVIORS: Tuple[str, ...] = (
    "direct", "zigzag", "zigzag_damped", "zigzag_unstable", "zigzag_variable_period",
    "spiral", "spiral_bounce", "spiral_oscillatory",
    "alternating", "bounce_approach", "circle_wait_advance",
)

#: Every valid ``behavior_type`` (``ENEMY_BEHAVIOR`` in the configuration).
ENEMY_BEHAVIORS: Tuple[str, ...] = RANDOM_ENEMY_BEHAVIORS + (
    "formation", "biformation", "focal-direct", "debug", "u-debug", "joystick",
)

# Patterns that start immediately (no random spawn delay).
_NO_START_DELAY = ("formation", "biformation", "focal-direct", "debug", "u-debug", "joystick")


class EnemyDrone:
    """An attacking drone.

    PT-BR: Um drone atacante.

    Args:
        interest_point_center: Centre of the target.
        position: Initial ``(x, y)``; a random border position if ``None``.
        behavior_type: One of :data:`ENEMY_BEHAVIORS`; random pattern if ``None``.
            Unknown names fall back to ``"direct"`` (with a warning).
        fixed: Only used by the ``"u-debug"`` pattern (stay still).

    Attributes:
        pos, vel: Position and velocity (``pygame.math.Vector2``).
        drone_id: Unique id within the episode.
        behavior_type: Current pattern (formation patterns switch to ``"direct"``).
        aggressiveness: Probability of attacking (instead of escaping) when detected.
        detector: Position of the friend that detected it this step, or ``None``.
        desperate_attack: Whether it committed to a straight-line attack.
        info: Short label shown on screen.
    """

    enemy_id_counter: int = 0
    original_drone_image: pygame.Surface = load_svg_as_surface(f"{_ASSETS_DIR}/drone_9.svg")
    original_drone_image_joystick: pygame.Surface = load_svg_as_surface(f"{_ASSETS_DIR}/drone_10.svg")

    class_seed: Optional[int] = None
    class_rng: Optional[random.Random] = None
    class_np_rng: Optional[np.random.RandomState] = None

    def __init__(
        self,
        interest_point_center: pygame.math.Vector2,
        position: Optional[Tuple[float, float]] = None,
        behavior_type: Optional[str] = None,
        fixed: bool = False,
    ) -> None:
        if EnemyDrone.class_seed is None:
            EnemyDrone.set_class_seed()

        self.interest_point_center: pygame.math.Vector2 = interest_point_center
        self.joystick: Optional[pygame.joystick.JoystickType] = None

        if position is None:
            self.pos: pygame.math.Vector2 = self.random_border_position()
        else:
            self.pos = pygame.math.Vector2(position[0], position[1])

        target_vector = self.interest_point_center - self.pos
        direction = target_vector.normalize() if target_vector.length() > 0 else pygame.math.Vector2(0, 0)
        self.vel: pygame.math.Vector2 = direction * ENEMY_SPEED
        self.info: str = ""
        self.aggressiveness: float = 0.0

        self.color: Tuple[int, int, int] = (255, 0, 0)
        self.drone_id: int = self._assign_id()

        # Auxiliary state of the oscillating patterns.
        self.phase: float = 0.0
        self.timer: int = 0

        if behavior_type is None:
            self.behavior_type: str = EnemyDrone.class_rng.choice(list(RANDOM_ENEMY_BEHAVIORS))
        else:
            if behavior_type not in ENEMY_BEHAVIORS:
                warnings.warn(f"Unknown enemy behaviour {behavior_type!r}; using 'direct'. "
                              f"Valid values: {', '.join(ENEMY_BEHAVIORS)}.", RuntimeWarning, stacklevel=2)
            self.behavior_type = behavior_type

        if behavior_type == "joystick":
            self._init_joystick()

        if self.behavior_type not in _NO_START_DELAY:
            self.start_delay: int = EnemyDrone.class_rng.randint(0, 100)
        else:
            self.start_delay = 0

        self.fixed: bool = fixed
        self.trajectory: List[pygame.math.Vector2] = []

        source = self.original_drone_image_joystick if self.behavior_type == "joystick" else self.original_drone_image
        desired_width = int(SIM_WIDTH * 0.02)
        aspect_ratio = source.get_height() / source.get_width()
        desired_size = (desired_width, int(desired_width * aspect_ratio))
        self.drone_image: pygame.Surface = pygame.transform.scale(source, desired_size)

        # Reaction to detection
        self.detector: Optional[pygame.math.Vector2] = None
        self.escape_steps: int = ESCAPE_STEPS
        self.escape_steps_count: int = 0
        self.desperate_attack: bool = False

    # ------------------------------------------------------------------ class API

    @classmethod
    def set_class_seed(cls, seed: Optional[int] = None) -> None:
        """Seed the RNGs shared by all enemy drones (``None`` = random seed).

        PT-BR: Define a semente dos geradores aleatórios dos inimigos.
        """
        cls.class_seed = seed if seed is not None else random.randint(0, 10000000)
        cls.class_rng = random.Random(cls.class_seed)
        cls.class_np_rng = np.random.RandomState(cls.class_seed)

    def _init_joystick(self) -> None:
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Joystick '{self.joystick.get_name()}' connected.")

    def random_border_position(self) -> pygame.math.Vector2:
        """Random point on one of the four borders of the simulation area."""
        rng = EnemyDrone.class_rng
        side = rng.choice(["top", "bottom", "left", "right"])
        if side == "top":
            return pygame.math.Vector2(rng.uniform(0, SIM_WIDTH), 0)
        if side == "bottom":
            return pygame.math.Vector2(rng.uniform(0, SIM_WIDTH), SIM_HEIGHT)
        if side == "left":
            return pygame.math.Vector2(0, rng.uniform(0, SIM_HEIGHT))
        return pygame.math.Vector2(SIM_WIDTH, rng.uniform(0, SIM_HEIGHT))

    def _assign_id(self) -> int:
        current_id = EnemyDrone.enemy_id_counter
        EnemyDrone.enemy_id_counter += 1
        return current_id

    # ------------------------------------------------------------------ update

    def update(self, detector: Optional[pygame.math.Vector2] = None) -> None:
        """Advance one step.

        PT-BR: Avança um passo.

        Args:
            detector: Position of a friendly drone within ``ENEMY_DETECTION_RANGE``
                (``None`` if the enemy is not being watched).
        """
        self.detector = detector
        self.timer += 1

        if self.timer < self.start_delay:
            return

        distance_to_interest = (self.interest_point_center - self.pos).length()
        if distance_to_interest < 1:
            self.pos = self.interest_point_center.copy()
            self.vel = pygame.math.Vector2(0, 0)
            return

        if distance_to_interest > INITIAL_DISTANCE:
            self.aggressiveness = max(
                INITIAL_AGGRESSIVENESS,
                1 + (distance_to_interest - INITIAL_DISTANCE)
                * ((INITIAL_AGGRESSIVENESS - 1) / (EXTERNAL_RADIUS - INITIAL_DISTANCE)),
            )
        else:
            self.aggressiveness = 1.0

        self.apply_behavior()
        self.pos += self.vel
        self._enforce_bounds()

        if len(self.trajectory) > 300:
            self.trajectory.pop(0)
        self.trajectory.append(self.pos.copy())

    def _enforce_bounds(self) -> None:
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

    def apply_behavior(self) -> None:
        """Choose this step's velocity: attack, escape, or the configured pattern."""
        if self.desperate_attack:
            target_vector = self.interest_point_center - self.pos
            self.vel = (target_vector.normalize() * ENEMY_SPEED if target_vector.length() > 0
                        else pygame.math.Vector2(0, 0))
            self.info = "DESPERATE ATTACK"
            return

        if self.escape_steps_count > 0:
            away = self.pos - (self.detector if self.detector is not None else self.interest_point_center)
            if away.length() == 0:
                away = self.pos - self.interest_point_center
            if away.length() > 0:
                self.vel = away.normalize() * ENEMY_SPEED
            self.escape_steps_count += 1
            self.info = f"ESCAPE {self.escape_steps_count}/{self.escape_steps} STEPS"
            if self.escape_steps_count >= self.escape_steps:
                self.escape_steps_count = 0
                self.detector = None
                if self.behavior_type in ("formation", "biformation", "focal-direct"):
                    self.behavior_type = "direct"
            return

        # Detected: attack with probability `aggressiveness`, otherwise start escaping.
        if self.detector and self.behavior_type != "joystick":
            if EnemyDrone.class_np_rng.rand() < self.aggressiveness:
                self.desperate_attack = True
                self.info = "DESPERATE ATTACK"
            else:
                self.escape_steps_count += 1
                return

        target_vector = self.interest_point_center - self.pos
        if target_vector.length() == 0:
            return

        self.info = f"aggr.: {self.aggressiveness:.2f}"

        behavior = self.behavior_type
        if behavior == "joystick":
            self._behavior_joystick()
        elif behavior == "direct":
            self._behavior_direct(target_vector)
        elif behavior.startswith("zigzag"):
            self._behavior_zigzag(target_vector)
        elif behavior.startswith("spiral"):
            self._behavior_spiral()
        elif behavior == "alternating":
            self._behavior_alternating(target_vector)
        elif behavior == "bounce_approach":
            self._behavior_bounce_approach()
        elif behavior == "circle_wait_advance":
            self._behavior_circle_wait_advance()
        elif behavior in ("formation", "biformation"):
            self._behavior_formation()
        elif behavior == "focal-direct":
            self._behavior_focal_direct()
        elif behavior == "debug":
            self._behavior_debug()
        elif behavior == "u-debug":
            self._behavior_u_debug(target_vector)
        else:
            self._behavior_direct(target_vector)

    # ------------------------------------------------------------------ patterns
    # The auxiliary attributes of some patterns are created lazily on first use,
    # which keeps the random-number stream identical to previous versions.

    def _behavior_joystick(self) -> None:
        """Manual control with a game pad (left stick)."""
        if self.joystick is None:
            self.info = "NO JOYSTICK"
            return
        direction = pygame.math.Vector2(self.joystick.get_axis(0), self.joystick.get_axis(1))
        direction = direction.normalize() if direction.length() > 0.1 else pygame.math.Vector2(0, 0)
        if self.detector is not None:
            self.joystick.rumble(1, 0, 50)
        else:
            self.joystick.rumble(0, self.aggressiveness, 50)
        self.vel = direction * ENEMY_SPEED
        self.info = "JOYSTICK"

    def _behavior_direct(self, target_vector: pygame.math.Vector2) -> None:
        """Straight line to the interest point."""
        self.vel = target_vector.normalize() * ENEMY_SPEED

    def _behavior_zigzag(self, target_vector: pygame.math.Vector2) -> None:
        """Sinusoidal lateral oscillation (``zigzag``, ``_damped``, ``_unstable``, ``_variable_period``)."""
        base_direction = target_vector.normalize()
        self.phase += 0.1
        perp = pygame.math.Vector2(-base_direction.y, base_direction.x)

        if self.behavior_type == "zigzag":
            offset = math.sin(self.phase) * 2.0
        elif self.behavior_type == "zigzag_damped":
            offset = math.sin(self.phase) * (5.0 * math.exp(-0.01 * self.timer))
        elif self.behavior_type == "zigzag_unstable":
            random_amplitude = EnemyDrone.class_rng.uniform(1.0, 10.0)
            random_frequency = EnemyDrone.class_rng.uniform(0.5, 2.0)
            offset = math.sin(random_frequency * self.timer) * random_amplitude
        else:  # zigzag_variable_period
            effective_frequency = 1.0 + 0.5 * math.sin(0.05 * self.timer)
            offset = math.sin(effective_frequency * self.phase) * 6.0

        self.vel = (base_direction + perp * offset).normalize() * ENEMY_SPEED

    def _behavior_spiral(self) -> None:
        """Spiral in (``spiral``), with periodic outward bounces (``spiral_bounce``)
        or with an extra lateral oscillation (``spiral_oscillatory``)."""
        r_vector = self.pos - self.interest_point_center
        current_distance = r_vector.length()
        angle = math.atan2(r_vector.y, r_vector.x) + 0.1

        if self.behavior_type == "spiral":
            new_radius = current_distance * 0.95
        else:
            if not hasattr(self, "sb_state"):
                self.sb_state = "spiral"
                self.sb_timer = 0
                self.spiral_duration = 60
                self.bounce_duration = 30

            if self.sb_state == "spiral":
                new_radius = current_distance * 0.9
                self.sb_timer += 1
                if self.sb_timer >= self.spiral_duration:
                    self.sb_state = "bounce"
                    self.sb_timer = 0
                    self.bounce_deviation = EnemyDrone.class_rng.uniform(20, 50) * EnemyDrone.class_rng.choice([-1, 1])
            else:
                direction = (-(self.interest_point_center - self.pos)).normalize().rotate(self.bounce_deviation)
                self.sb_timer += 1
                if self.sb_timer >= self.bounce_duration:
                    self.sb_state = "spiral"
                    self.sb_timer = 0
                self.vel = direction * ENEMY_SPEED
                return

        desired_pos = self.interest_point_center + pygame.math.Vector2(math.cos(angle), math.sin(angle)) * new_radius
        direction = desired_pos - self.pos

        if self.behavior_type == "spiral_oscillatory" and direction.length() > 0:
            base_direction = direction.normalize()
            self.phase += 0.05
            perp = pygame.math.Vector2(-base_direction.y, base_direction.x)
            direction = (base_direction + perp * math.sin(self.phase)).normalize() * direction.length()

        self.vel = direction.normalize() * ENEMY_SPEED if direction.length() != 0 else pygame.math.Vector2(0, 0)

    def _behavior_alternating(self, target_vector: pygame.math.Vector2) -> None:
        """Alternate 50 steps straight in and 50 steps sideways (with noise)."""
        cycle_length = 100
        target_direction = target_vector.normalize()
        if self.timer % cycle_length < cycle_length / 2:
            direction = target_direction
        else:
            direction = pygame.math.Vector2(-target_direction.y, target_direction.x)
            direction = direction.rotate_rad(EnemyDrone.class_rng.uniform(-0.1, 0.1)).normalize()
        self.vel = direction * ENEMY_SPEED

    def _behavior_bounce_approach(self) -> None:
        """Approach for 60 steps, then retreat 30 steps with a random deviation."""
        if not hasattr(self, "state"):
            self.state = "approach"
            self.state_timer = 0
            self.approach_duration = 60
            self.retreat_duration = 30

        if self.state == "approach":
            direction = (self.interest_point_center - self.pos).normalize()
            self.state_timer += 1
            if self.state_timer >= self.approach_duration:
                self.state = "retreat"
                self.state_timer = 0
                self.deviation_angle = EnemyDrone.class_rng.uniform(20, 50) * EnemyDrone.class_rng.choice([-1, 1])
        else:
            direction = (-(self.interest_point_center - self.pos)).normalize().rotate(self.deviation_angle)
            self.state_timer += 1
            if self.state_timer >= self.retreat_duration:
                self.state = "approach"
                self.state_timer = 0
        self.vel = direction * ENEMY_SPEED

    def _behavior_circle_wait_advance(self) -> None:
        """Circle in place for a random time, then advance for 60 steps."""
        if not hasattr(self, "cw_state"):
            self.cw_state = "wait"
            self.cw_timer = 0
            self.wait_duration = EnemyDrone.class_rng.randint(40, 120)
            self.advance_duration = 60
            self.wait_turn_rate = 2

        if self.cw_state == "wait":
            self.vel = self.vel.rotate(self.wait_turn_rate)
            self.cw_timer += 1
            if self.cw_timer >= self.wait_duration:
                self.cw_state = "advance"
                self.cw_timer = 0
        else:
            direction = (self.interest_point_center - self.pos).normalize()
            self.vel = direction.rotate(EnemyDrone.class_rng.uniform(-5, 5)) * ENEMY_SPEED
            self.cw_timer += 1
            if self.cw_timer >= self.advance_duration:
                self.cw_state = "wait"
                self.cw_timer = 0

    def _behavior_formation(self) -> None:
        """Rows of 5 drones entering from the right (``formation``) or from both sides
        (``biformation``); switch to ``direct`` once inside ``INITIAL_DISTANCE``."""
        if not hasattr(self, "formation_id"):
            self.formation_id = self.drone_id % ENEMY_COUNT
            front = 5

            if self.behavior_type == "formation":
                row = self.formation_id // front
                column = self.formation_id % front - front // 2
                self.pos = pygame.math.Vector2(SIM_WIDTH, SIM_HEIGHT // 2 + column * 60)
                self.start_delay = 40 * row
            else:
                if self.formation_id < ENEMY_COUNT // 2:
                    index, x, vel_x = self.formation_id, SIM_WIDTH, -1
                else:
                    index, x, vel_x = self.formation_id - ENEMY_COUNT // 2, 0, 1
                row = index // front
                column = index % front - front // 2
                self.pos = pygame.math.Vector2(x, SIM_HEIGHT // 2 + column * 60)
                self.vel = pygame.math.Vector2(vel_x, 0)
                self.start_delay = 40 * row

        if self.pos.distance_to(self.interest_point_center) > INITIAL_DISTANCE:
            self.info = f"#{self.formation_id}"
        else:
            self.behavior_type = "direct"

    def _behavior_focal_direct(self) -> None:
        """Every drone leaves from the top-right corner, one every 20 steps."""
        self.pos = pygame.math.Vector2(SIM_WIDTH, 0)
        self.start_delay = 20 * (self.drone_id % ENEMY_COUNT)
        self.behavior_type = "direct"

    def _behavior_debug(self) -> None:
        """Clockwise orbit around the interest point (testing aid)."""
        if not hasattr(self, "debug_orbit_radius"):
            center = pygame.math.Vector2(self.interest_point_center[0], self.interest_point_center[1])
            self.debug_orbit_radius = self.pos.distance_to(center)
            self.debug_angular_speed = 0.02

        radius_vector = self.pos - pygame.math.Vector2(self.interest_point_center[0], self.interest_point_center[1])
        tangent_vector = pygame.math.Vector2(radius_vector.y, -radius_vector.x)
        self.vel = (tangent_vector.normalize() * ENEMY_SPEED if tangent_vector.length() > 0
                    else pygame.math.Vector2(0, 0))
        self.info = "ORBIT"

    def _behavior_u_debug(self, target_vector: pygame.math.Vector2) -> None:
        """Go in for 300 steps, sideways for 40, out for 300, then stop (testing aid)."""
        if not hasattr(self, "u_debug_phase"):
            self.u_debug_phase = 0
            self.u_debug_timer = 0
            self.forward_steps = 300
            self.perp_steps = 40

        if self.fixed:
            self.vel = pygame.math.Vector2(0, 0)
            return

        target_direction = target_vector.normalize()
        if self.u_debug_phase == 0:
            self.vel = target_direction * ENEMY_SPEED
            self.u_debug_timer += 1
            if self.u_debug_timer >= self.forward_steps:
                self.u_debug_phase, self.u_debug_timer = 1, 0
        elif self.u_debug_phase == 1:
            self.vel = pygame.math.Vector2(-target_direction.y, target_direction.x) * ENEMY_SPEED
            self.u_debug_timer += 1
            if self.u_debug_timer >= self.perp_steps:
                self.u_debug_phase, self.u_debug_timer = 2, 0
        else:
            self.vel = (-target_direction) * ENEMY_SPEED
            self.u_debug_timer += 1
            if self.u_debug_timer >= self.forward_steps:
                self.vel = pygame.math.Vector2(0, 0)

    # ------------------------------------------------------------------ drawing

    def draw(
        self,
        surface: pygame.Surface,
        show_detection: bool = True,
        show_trajectory: bool = False,
        show_debug: bool = False,
    ) -> None:
        """Draw the drone, its trajectory, detection range, id and debug label.

        PT-BR: Desenha o drone, a trajetória, o alcance de detecção, o id e o rótulo.
        """
        if show_trajectory and len(self.trajectory) > 1:
            draw_fading_trajectory(surface, self.trajectory, self.color, decay_rate=0.05)

        center = (int(self.pos.x), int(self.pos.y))
        surface.blit(self.drone_image, self.drone_image.get_rect(center=center))

        if show_detection:
            draw_dashed_circle(surface, (*self.color, 64), center, ENEMY_DETECTION_RANGE,
                               dash_length=5, space_length=5, width=1)

        font = get_font(10)
        label = font.render(f"ID: E{self.drone_id}", True, (255, 255, 255))
        label.set_alpha(128)
        surface.blit(label, (center[0] + 20, center[1] - 20))

        if show_debug:
            debug_label = font.render(self.info, True, (255, 215, 0))
            surface.blit(debug_label, (center[0] - 3.5 * len(self.info), center[1] + 25))
