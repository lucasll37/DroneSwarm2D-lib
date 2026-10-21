"""The simulation environment (Gym-like ``reset`` / ``step`` API).

PT-BR: O ambiente de simulação (API no estilo Gym: ``reset`` / ``step``).

Example:
    >>> import DroneSwarm2D
    >>> settings = DroneSwarm2D.init("config.json", headless=True)
    >>> env = DroneSwarm2D.AirTrafficEnv(mode=None, seed=42)  # mode="human" opens the UI
    >>> obs, done = env.reset()
    >>> while not done:
    ...     obs, reward, done, info = env.step(None)

One step:

1. every friendly drone perceives, communicates and moves (``FriendDrone.update``);
2. every enemy moves, reacting if a friend is within ``ENEMY_DETECTION_RANGE``;
3. enemies inside ``INTEREST_POINT_ATTACK_RANGE`` hit the interest point and are
   removed (each hit also destroys one ground radar, if any);
4. every friend/enemy pair closer than ``NEUTRALIZATION_RANGE`` (enemy outside
   active DMZs) fights; the outcome is drawn with the ``NEUTRALIZATION_PROB_*``
   probabilities;
5. the reward is computed and the frame is rendered (``mode == "human"``).
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import mpl_toolkits.mplot3d  # noqa: F401  (registers the "3d" projection)
import numpy as np
import pygame
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

if not pygame.get_init():
    pygame.init()

from ..behaviorsDefault import FriendAEWBehaviorDefault, FriendCommonBehaviorDefault, FriendRadarBehaviorDefault
from . import rendering
from .DemilitarizedZone import CircleDMZ
from .EnemyDrone import EnemyDrone
from .FriendDrone import FriendDrone
from .InterestPoint import CircleInterestPoint
from .rendering import (  # noqa: F401  (re-exported for backward compatibility)
    Button,
    draw_direction,
    draw_friend_communication,
    draw_grid,
    draw_heatmap,
    get_font,
)
from .settings import (
    AEW_COUNT,
    AEW_RANGE,
    BROKEN_COUNT,
    CELL_SIZE,
    CENTER,
    DT_STEP,
    ENEMY_COUNT,
    ENEMY_DETECTION_RANGE,
    EPSILON,
    EXTERNAL_RADIUS,
    FRIEND_COUNT,
    FRIEND_DETECTION_RANGE,
    GRAPH_HEIGHT,
    GRAPH_WIDTH,
    GRID_HEIGHT,
    GRID_WIDTH,
    INITIAL_DISTANCE,
    INITIAL_N_LAYERS,
    INTEREST_POINT_ATTACK_RANGE,
    INTEREST_POINT_CENTER,
    INTEREST_POINT_DAMAGE,
    INTERNAL_RADIUS,
    JOYSTICK,
    N_LINE_SIGHT_CROSSING,
    NEUTRALIZATION_PROB_BOTH_DEAD,
    NEUTRALIZATION_PROB_ENEMY_ALIVE,
    NEUTRALIZATION_PROB_FRIEND_ALIVE,
    NEUTRALIZATION_RANGE,
    PASSIVE_DETECTION,
    PLOT_THRESHOLD,
    RADAR_COUNT,
    RADAR_RANGE,
    SIM_HEIGHT,
    SIM_WIDTH,
    TYPE_OF_SCENARIO,
    screen,
)
from .utils import draw_dashed_line, sim_to_geo, smooth_matrix_with_kernel_10x10  # noqa: F401

clock: pygame.time.Clock = pygame.time.Clock()


def compute_global_matrices(
    drones: List[Any],
    intensity_attr: str,
    direction_attr: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse the perception of several drones: per cell, keep the strongest detection.

    PT-BR: Funde a percepção de vários drones mantendo, em cada célula, a
    detecção mais intensa.

    Args:
        drones: Friendly drones.
        intensity_attr: e.g. ``"enemy_intensity"``.
        direction_attr: e.g. ``"enemy_direction"``.

    Returns:
        ``(intensity, direction)`` with shapes ``(GRID_W, GRID_H)`` and ``(GRID_W, GRID_H, 2)``.
    """
    global_intensity = np.zeros((GRID_WIDTH, GRID_HEIGHT))
    global_direction = np.zeros((GRID_WIDTH, GRID_HEIGHT, 2))
    if not drones:
        return global_intensity, global_direction

    intensities = np.stack([getattr(d, intensity_attr) for d in drones])
    directions = np.stack([getattr(d, direction_attr) for d in drones])
    best = np.argmax(intensities, axis=0)  # first drone wins ties, as before
    best_intensity = np.take_along_axis(intensities, best[None], axis=0)[0]
    positive = best_intensity > 0
    global_intensity[positive] = best_intensity[positive]
    best_direction = np.take_along_axis(directions, best[None, ..., None], axis=0)[0]
    global_direction[positive] = best_direction[positive]
    return global_intensity, global_direction


class AirTrafficEnv:
    """Simulation of a defensive drone swarm protecting an interest point.

    PT-BR: Simulação de um enxame defensivo protegendo um ponto de interesse.

    Args:
        max_steps: Episode length limit.
        mode: ``"human"`` renders the pygame UI (capped at :attr:`render_fps`);
            any other value (e.g. ``None``) runs headless and as fast as possible.
        friend_behavior: Behaviour of the interceptors (default: ``FriendCommonBehaviorDefault``).
        friend_aew_behavior: Behaviour of the AEW drones (default: ``FriendAEWBehaviorDefault``).
        friend_radar_behavior: Behaviour of the radars (default: ``FriendRadarBehaviorDefault``).
        enemy_behavior: Enemy pattern (see ``EnemyDrone.ENEMY_BEHAVIORS``), ``None`` = random per drone.
        demilitarized_zones: ``[(x, y, radius), ...]``; zones overlapping the interest point are ignored.
        seed: Random seed. The same seed and configuration always produce the same episode.

    Attributes:
        friend_drones, enemy_drones: Drones alive.
        interest_point: The defended :class:`CircleInterestPoint`.
        current_step, episode, accum_reward: Episode counters.
        enemies_neutralized: Enemies destroyed in combat this episode.
        friends_lost: Friendly drones destroyed this episode.
        friends_returned: Friendly drones that landed with "return to base".
        sucessful_attacks: Enemies that reached the interest point this episode.
    """

    def __init__(
        self,
        max_steps: int = 5_000,
        mode: Optional[str] = "human",
        friend_behavior: Optional[Any] = None,
        friend_aew_behavior: Optional[Any] = None,
        friend_radar_behavior: Optional[Any] = None,
        enemy_behavior: Optional[str] = None,
        demilitarized_zones: Optional[List[Tuple[float, float, float]]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.seed: int = seed if seed is not None else random.randint(0, 10000000)
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.RandomState(self.seed)

        self.simulation_start_time: datetime = _utcnow()
        self.current_time: datetime = _utcnow()
        self.max_steps: int = max_steps
        self.mode = mode
        self.render_fps: int = 30
        self.friend_behavior = friend_behavior if friend_behavior is not None else FriendCommonBehaviorDefault()
        self.friend_aew_behavior = (friend_aew_behavior if friend_aew_behavior is not None
                                    else FriendAEWBehaviorDefault())
        self.friend_radar_behavior = (friend_radar_behavior if friend_radar_behavior is not None
                                      else FriendRadarBehaviorDefault())
        self.enemy_behavior = enemy_behavior
        self.folder_name: Optional[str] = None

        self.episode: int = 0
        self.current_step: int = 0
        self.friend_drones: List[FriendDrone] = []
        self.enemy_drones: List[EnemyDrone] = []
        self.interest_point: Optional[CircleInterestPoint] = None
        self.selected_drone: Optional[FriendDrone] = None
        self.leader: Optional[FriendDrone] = None
        self.tacview_logs: Dict[str, List] = {}

        self.sim_surface = pygame.Surface((SIM_WIDTH, SIM_HEIGHT)).convert_alpha()
        self.sim_surface.fill((0, 0, 0))

        self.fig: Optional[Figure] = None
        self.canvas: Optional[FigureCanvas] = None
        self.accum_reward: float = 0.0
        self.attack_penalty: int = 0
        self.sucessful_attacks: int = 0
        self.enemies_neutralized: int = 0
        self.friends_lost: int = 0
        self.friends_returned: int = 0
        self.font: pygame.font.Font = get_font(14)

        # UI toggles
        self.save_frames: bool = False
        self.show_graph: bool = False
        self.paused: bool = False
        self.show_friend_detection_range: bool = True
        self.show_enemy_detection_range: bool = True
        self.show_dashed_lines: bool = True
        self.show_dmz: bool = True
        self.show_target_lines: bool = False
        self.show_friend_comm_range: bool = False
        self.show_trajectory: bool = True
        self.show_debug: bool = True
        self.return_to_base: bool = False
        self.export_to_tacview: bool = False
        self.use_triangulation: bool = PASSIVE_DETECTION
        self.frame_number: int = 0

        # Communication statistics
        self.messages_exchanged: float = 0
        self.messages_count_interval: List[float] = []
        self.last_message_count_time: int = pygame.time.get_ticks()
        self.messages_per_second: float = 0.0
        self.active_connections: float = 0
        self.total_distance_traveled: float = 0.0

        self.demilitarized_zones: List[CircleDMZ] = []
        for x, y, radius in demilitarized_zones or []:
            zone_center = pygame.math.Vector2(x, y)
            if zone_center.distance_to(INTEREST_POINT_CENTER) > INTERNAL_RADIUS + radius:
                self.demilitarized_zones.append(CircleDMZ(zone_center, radius, seed=self.seed))

        self._setup_ui_buttons()

    # ------------------------------------------------------------------ UI

    def _setup_ui_buttons(self) -> None:
        """Create the 4x4 grid of control buttons under the graph panel."""
        width, height, spacing = 130, 30, 5
        layout = [
            [("Tog. Graph", self.toggle_graph, False), ("Pause", self.toggle_pause, False),
             ("Reset", self.reset_simulation, None), ("Exit", self.exit_env, None)],
            [("Tog. Friend Range", self.toggle_friend_range, True), ("Tog. Enemy Range", self.toggle_enemy_range, True),
             ("Tog. Friend Comm.", self.toggle_dashed, True), ("Tog. DMZ", self.toggle_dmz, True)],
            [("Tog. Comm Range", self.toggle_comm_range, False), ("Export Tacview", self.toggle_tacview_export, False),
             ("Tog. Save Frames", self.toggle_save_frames, False),
             ("Tog. Target Lines", self.toggle_target_lines, False)],
            [("Tog. Trajetory", self.toggle_trajectory, True), ("Tog. Debug", self.toggle_debug, True),
             ("Tog. D. Passive.", self.toggle_triangulation, PASSIVE_DETECTION),
             ("Tog. Return", self.toggle_return, False)],
        ]
        self.buttons: List[Button] = []
        for row_index, row in enumerate(layout):
            y = GRAPH_HEIGHT - 170 + 40 * row_index
            for col_index, (text, callback, toggled) in enumerate(row):
                x = SIM_WIDTH + 10 + col_index * (width + spacing)
                color = (200, 0, 0) if text == "Exit" else (70, 70, 70)
                self.buttons.append(Button((x, y, width, height), text, callback, toggled=toggled, color=color))

    def is_in_demilitarized_zone(self, position: pygame.math.Vector2) -> bool:
        """Whether ``position`` is inside any demilitarised zone."""
        return any(zone.contains(position) for zone in self.demilitarized_zones)

    def toggle_graph(self) -> None:
        self.show_graph = not self.show_graph

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def reset_simulation(self) -> None:
        self.reset()

    def exit_env(self) -> None:
        if self.export_to_tacview:
            self.handle_tacview()
        self.close()
        sys.exit()

    def toggle_friend_range(self) -> None:
        self.show_friend_detection_range = not self.show_friend_detection_range

    def toggle_enemy_range(self) -> None:
        self.show_enemy_detection_range = not self.show_enemy_detection_range

    def toggle_dashed(self) -> None:
        self.show_dashed_lines = not self.show_dashed_lines

    def toggle_dmz(self) -> None:
        """Toggle the DMZs: both their drawing **and** the no-engagement rule."""
        self.show_dmz = not self.show_dmz

    def toggle_target_lines(self) -> None:
        self.show_target_lines = not self.show_target_lines

    def toggle_save_frames(self) -> None:
        self.save_frames = not self.save_frames

    def toggle_comm_range(self) -> None:
        self.show_friend_comm_range = not self.show_friend_comm_range

    def toggle_trajectory(self) -> None:
        self.show_trajectory = not self.show_trajectory

    def toggle_tacview_export(self) -> None:
        self.export_to_tacview = not self.export_to_tacview

    def toggle_debug(self) -> None:
        self.show_debug = not self.show_debug

    def toggle_return(self) -> None:
        self.return_to_base = not self.return_to_base

    def toggle_triangulation(self) -> None:
        self.use_triangulation = not self.use_triangulation

    # Deprecated misspellings kept for backward compatibility.
    toogle_save_frames = toggle_save_frames
    toggle_trajetory = toggle_trajectory
    toogle_return = toggle_return
    toogle_triangulation = toggle_triangulation

    def handle_tacview(self) -> None:
        """Write one Tacview-compatible CSV per drone to ``./output/tacview_export_<date>``."""
        filepath = os.path.join(".", "output", f"tacview_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(filepath, exist_ok=True)
        for drone_name, log in self.tacview_logs.items():
            color = "Green" if "Friend" in drone_name.split("_")[0] else "Red"
            filename = os.path.join(filepath, f" ({drone_name}) [{color}].csv")
            with open(filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Time", "Longitude", "Latitude", "Altitude", "Roll (deg)", "Pitch (deg)", "Yaw (deg)"])
                for current_time, _name, pos_x, pos_y, _, vel_x, vel_y, _ in log:
                    time = current_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-4] + "Z"
                    lon, lat = sim_to_geo(pos_x, pos_y)
                    yaw = math.degrees(math.atan2(vel_y, vel_x)) if (vel_x != 0 or vel_y != 0) else 0.0
                    writer.writerow([time, lon, lat, 1000.0, 0.0, 0.0, yaw])
            print(f"Simulation exported to {filename} for Tacview.")

    # ------------------------------------------------------------------ episode

    def _get_observation(self) -> np.ndarray:
        """``[x, y, vx, vy]`` of every friend followed by every enemy (float32)."""
        obs: List[float] = []
        for drone in self.friend_drones:
            obs.extend([drone.pos.x, drone.pos.y, drone.vel.x, drone.vel.y])
        for enemy in self.enemy_drones:
            obs.extend([enemy.pos.x, enemy.pos.y, enemy.vel.x, enemy.vel.y])
        return np.array(obs, dtype=np.float32)

    def reset(self) -> Tuple[np.ndarray, bool]:
        """Start a new episode.

        PT-BR: Inicia um novo episódio.

        Friendly drones are placed in rings around the interest point (radars at
        ``RADAR_RANGE``, AEWs at ``AEW_RANGE``, interceptors in ``INITIAL_N_LAYERS``
        rings from ``INITIAL_DISTANCE``); enemies spawn on the borders.

        Returns:
            ``(observation, done)``.
        """
        self.fig = Figure(figsize=(GRAPH_WIDTH / 100, GRAPH_HEIGHT / 100), dpi=100)
        self.canvas = FigureCanvas(self.fig)

        self.simulation_start_time = _utcnow()
        self.current_time = _utcnow()
        self.episode += 1
        self.current_step = 0
        self.accum_reward = 0.0
        self.attack_penalty = 0
        self.sucessful_attacks = 0
        self.enemies_neutralized = 0
        self.friends_lost = 0
        self.friends_returned = 0
        self.leader = None
        self.selected_drone = None
        self.tacview_logs = {}
        self.friend_drones = []
        self.enemy_drones = []

        FriendDrone.friend_id_counter = 0
        FriendDrone.reset_clock()
        EnemyDrone.enemy_id_counter = 0

        center = CENTER
        self.interest_point = CircleInterestPoint(CENTER, INTERNAL_RADIUS, EXTERNAL_RADIUS, seed=self.seed)
        # Seed both classes before creating any drone so every episode is reproducible.
        FriendDrone.set_class_seed(seed=self.seed)
        EnemyDrone.set_class_seed(seed=self.seed)

        def ring_position(radius: float, angle: float) -> pygame.math.Vector2:
            return center + pygame.math.Vector2(radius * math.cos(angle), radius * math.sin(angle))

        for i in range(RADAR_COUNT):
            pos = ring_position(RADAR_RANGE, 2 * math.pi * i / RADAR_COUNT)
            self.friend_drones.append(FriendDrone(self.interest_point.center, position=(pos.x, pos.y),
                                                  behavior=self.friend_radar_behavior, fixed=True))

        for j in range(AEW_COUNT):
            pos = ring_position(AEW_RANGE, 2 * math.pi * j / AEW_COUNT)
            self.friend_drones.append(FriendDrone(self.interest_point.center, position=(pos.x, pos.y),
                                                  behavior=self.friend_aew_behavior))

        n_common = FRIEND_COUNT - AEW_COUNT - RADAR_COUNT
        per_layer = max(1, n_common // INITIAL_N_LAYERS)
        for k in range(n_common):
            angle = (2 * INITIAL_N_LAYERS * math.pi * k) / n_common
            layer = k // per_layer
            pos = ring_position(INITIAL_DISTANCE + layer * 40, angle)
            drone = FriendDrone(self.interest_point.center, position=(pos.x, pos.y),
                                behavior=self.friend_behavior, broken=(k < BROKEN_COUNT))
            if k == 0:
                self.selected_drone = self.leader = drone
                drone.is_leader = True
                drone.selected = True
                if JOYSTICK == "Friend":
                    drone.joystick_controlled = True
            self.friend_drones.append(drone)

        n_enemies = ENEMY_COUNT
        if JOYSTICK == "Enemy":
            self.enemy_drones.append(EnemyDrone(self.interest_point.center, behavior_type="joystick"))
            n_enemies -= 1
        self.enemy_drones.extend(EnemyDrone(self.interest_point.center, behavior_type=self.enemy_behavior)
                                 for _ in range(n_enemies))

        done = self.current_step >= self.max_steps or len(self.enemy_drones) == 0
        return self._get_observation(), done

    def _remove_friend_references(self, drone: FriendDrone) -> None:
        """Drop the leader / selected-drone references to a removed drone."""
        if drone.is_leader:
            self.leader = None
        if drone.selected:
            self.selected_drone = None

    @staticmethod
    def _engagement_outcome(friend: FriendDrone, rand_val: float) -> Optional[str]:
        """Result of a friend/enemy encounter: ``"enemy"``, ``"friend"``, ``"both"`` or ``None``.

        * Interceptors: enemy destroyed with ``NEUTRALIZATION_PROB_FRIEND_ALIVE``,
          interceptor destroyed with ``NEUTRALIZATION_PROB_ENEMY_ALIVE``, both
          destroyed otherwise.
        * AEW and radars (not armed): both destroyed (collision) with
          ``NEUTRALIZATION_PROB_BOTH_DEAD``, otherwise nothing happens.
        """
        if friend.behavior_type in ("AEW", "RADAR"):
            return "both" if rand_val <= NEUTRALIZATION_PROB_BOTH_DEAD else None

        if rand_val <= NEUTRALIZATION_PROB_FRIEND_ALIVE:
            return "enemy"
        if rand_val <= NEUTRALIZATION_PROB_FRIEND_ALIVE + NEUTRALIZATION_PROB_ENEMY_ALIVE:
            return "friend"
        return "both"

    def _update_friends(self) -> None:
        for drone in self.friend_drones:
            drone.update(self.enemy_drones, self.friend_drones, self.use_triangulation, self.return_to_base)
            self.messages_exchanged += getattr(drone, "messages_sent_this_cycle", 0)
            self.active_connections += getattr(drone, "active_connections", 0)
            self.total_distance_traveled += drone.distance_traveled

            if self.return_to_base and drone.pos.distance_to(CENTER) <= EPSILON:
                self.friend_drones = [f for f in self.friend_drones if f.drone_id != drone.drone_id]
                self.friends_returned += 1
                self._remove_friend_references(drone)

    def _update_enemies(self) -> None:
        for enemy in self.enemy_drones:
            detector = None
            for friend in self.friend_drones:
                if enemy.pos.distance_to(friend.pos) <= ENEMY_DETECTION_RANGE:
                    detector = friend.pos
                    break
            enemy.update(detector)

    def _update_message_rate(self) -> None:
        now = pygame.time.get_ticks()
        elapsed = (now - self.last_message_count_time) / 1000
        self.messages_exchanged /= 2  # every exchange was counted by both ends
        self.active_connections /= 2
        if elapsed >= 1e-1:
            self.messages_count_interval.append(self.messages_exchanged / elapsed)
            if len(self.messages_count_interval) > 5:
                self.messages_count_interval.pop(0)
            self.messages_per_second = sum(self.messages_count_interval) / len(self.messages_count_interval)
            self.last_message_count_time = now

    def _resolve_attacks(self) -> None:
        """Enemies that reach the interest point damage it and are removed."""
        surviving: List[EnemyDrone] = []
        for enemy in self.enemy_drones:
            if enemy.pos.distance_to(self.interest_point.center) <= INTEREST_POINT_ATTACK_RANGE + EPSILON:
                self.interest_point.health -= INTEREST_POINT_DAMAGE
                self.attack_penalty += 1
                self.sucessful_attacks += 1
                radar = next((d for d in self.friend_drones if d.behavior_type == "RADAR"), None)
                if radar is not None:
                    self.friend_drones.remove(radar)
                    self.friends_lost += 1
                    self._remove_friend_references(radar)
            else:
                surviving.append(enemy)
        self.enemy_drones = surviving
        self.interest_point.health = max(self.interest_point.health, 0)

    def _resolve_engagements(self) -> None:
        """Friend/enemy pairs within ``NEUTRALIZATION_RANGE`` fight."""
        friends_to_remove: set = set()
        enemies_to_remove: set = set()
        for i, friend in enumerate(self.friend_drones):
            for j, enemy in enumerate(self.enemy_drones):
                if friend.pos.distance_to(enemy.pos) > NEUTRALIZATION_RANGE:
                    continue
                if self.show_dmz and self.is_in_demilitarized_zone(enemy.pos):
                    continue
                outcome = self._engagement_outcome(friend, self.rng.random())
                if outcome in ("enemy", "both"):
                    enemies_to_remove.add(j)
                if outcome in ("friend", "both"):
                    friends_to_remove.add(i)

        for i in friends_to_remove:
            self._remove_friend_references(self.friend_drones[i])
        self.friends_lost += len(friends_to_remove)
        self.enemies_neutralized += len(enemies_to_remove)
        self.friend_drones = [f for idx, f in enumerate(self.friend_drones) if idx not in friends_to_remove]
        self.enemy_drones = [e for idx, e in enumerate(self.enemy_drones) if idx not in enemies_to_remove]

        if self.selected_drone is None and self.friend_drones:
            self.selected_drone = self.friend_drones[0]
            self.selected_drone.selected = True
        if self.leader is None and self.friend_drones:
            self.leader = self.friend_drones[0]
            self.leader.is_leader = True

    def _log_tacview(self) -> None:
        for prefix, drones in (("Friend", self.friend_drones), ("Enemy", self.enemy_drones)):
            for drone in drones:
                name = f"{prefix}_{drone.drone_id}"
                self.tacview_logs.setdefault(name, []).append(
                    (self.current_time, name, drone.pos.x, drone.pos.y, 0, drone.vel.x, drone.vel.y, 0))

    def step(self, action: Any = None) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Advance the simulation by one step.

        PT-BR: Avança a simulação um passo.

        Args:
            action: Ignored (drones act through their behaviours); kept for a
                Gym-like signature.

        Returns:
            ``(observation, reward, done, info)``. ``info`` contains
            ``current_step``, ``accum_reward``, ``enemies_shotdown``,
            ``friends_shotdown``, ``sucessful_attacks``, ``interest_point_health``,
            ``state_percentages``, ``total_distance_traveled``,
            ``friends_returned``, ``enemies_remaining`` and ``friends_remaining``.
        """
        self.handle_events()
        frame_reward = 0.0

        if not self.paused:
            self.messages_exchanged = 0
            self.active_connections = 0
            self.total_distance_traveled = 0
            self.current_time += timedelta(seconds=DT_STEP)

            self._update_friends()
            self._update_enemies()
            self._update_message_rate()

            self.current_step += 1
            self._resolve_attacks()
            self._resolve_engagements()
            frame_reward = self.compute_reward()
            self.accum_reward += frame_reward
            self._log_tacview()

        self.render(self.mode)
        obs = self._get_observation()
        done = self.current_step >= self.max_steps or len(self.enemy_drones) == 0

        state_percentages: Dict[str, float] = {}
        for drone in self.friend_drones:
            for state, percentage in drone.get_state_percentages().items():
                state_percentages[state] = state_percentages.get(state, 0) + percentage / len(self.friend_drones)

        info: Dict[str, Any] = {
            "current_step": self.current_step,
            "accum_reward": self.accum_reward,
            "enemies_shotdown": self.enemies_neutralized,
            "friends_shotdown": self.friends_lost,
            "sucessful_attacks": self.sucessful_attacks,
            "interest_point_health": self.interest_point.health,
            "state_percentages": state_percentages,
            "total_distance_traveled": self.total_distance_traveled,
            "friends_returned": self.friends_returned,
            "enemies_remaining": len(self.enemy_drones),
            "friends_remaining": len(self.friend_drones),
        }

        if done and self.export_to_tacview:
            self.handle_tacview()

        return obs, frame_reward, done, info

    def compute_reward(
        self,
        normilizer_factor: float = 1 / 10_000,
        bonus_factor: float = 10,
        penalty_factor: float = 2000,
    ) -> float:
        """Reward of the current step.

        ``sum(enemy distances to the centre) * normalizer``
        ``+ bonus_factor * enemies_neutralized * |centre| * normalizer``
        ``- penalty_factor * attacks since the last call``.

        PT-BR: Recompensa do passo: distância dos inimigos ao centro + bônus por
        inimigos neutralizados - penalidade por ataques bem-sucedidos.
        """
        total_distance = sum(enemy.pos.distance_to(self.interest_point.center) for enemy in self.enemy_drones)
        max_distance = self.interest_point.center.distance_to(pygame.math.Vector2(0, 0))
        bonus = (bonus_factor * self.enemies_neutralized * max_distance) * normilizer_factor
        penalty = penalty_factor * self.attack_penalty
        self.attack_penalty = 0
        return total_distance * normilizer_factor + bonus - penalty

    # ------------------------------------------------------------------ rendering

    def plot_individual_states(self) -> pygame.Surface:
        """Render the 3-D perception plot of the selected drone (graph panel)."""
        if self.fig is None:
            self.fig = Figure(figsize=(GRAPH_WIDTH / 100, GRAPH_HEIGHT / 100), dpi=100)
            self.canvas = FigureCanvas(self.fig)
        self.fig.clear()
        cmap = matplotlib.colormaps["hsv"]

        drone = self.selected_drone or (self.friend_drones[0] if self.friend_drones else None)
        if drone is None:
            return pygame.Surface((GRAPH_WIDTH, GRAPH_HEIGHT))

        x = np.linspace(CELL_SIZE / 2, SIM_WIDTH - CELL_SIZE / 2, GRID_WIDTH)
        y = np.linspace(CELL_SIZE / 2, SIM_HEIGHT - CELL_SIZE / 2, GRID_HEIGHT)
        X, Y = np.meshgrid(x, y)

        views = (("Enemy Detection", drone.enemy_intensity, drone.enemy_direction),
                 ("Friend Detection", drone.friend_intensity, drone.friend_direction))
        for idx, (title, intensity, direction) in enumerate(views):
            ax = self.fig.add_subplot(2, 1, idx + 1, projection="3d")
            ax.set_box_aspect((1, 1, 0.4))

            Z, smoothed_direction = smooth_matrix_with_kernel_10x10(
                intensity.T, np.transpose(direction, (1, 0, 2)), sigma=2, flat_radius=1)
            angle = np.arctan2(smoothed_direction[..., 1], smoothed_direction[..., 0])
            facecolors = cmap((angle + math.pi) / (2 * math.pi))
            facecolors[Z < PLOT_THRESHOLD] = [1, 1, 1, 1]

            ax.plot_surface(X, Y, Z, facecolors=facecolors, linewidth=0, antialiased=True, shade=False)
            ax.contourf(X, Y, Z, zdir="x", offset=ax.get_xlim()[0], cmap="Grays")
            ax.contourf(X, Y, Z, zdir="y", offset=ax.get_ylim()[0], cmap="Grays")
            ax.plot_wireframe(X, Y, Z, color="black", linewidth=0.2, rstride=1, cstride=1)
            ax.plot([drone.pos.x] * 2, [drone.pos.y] * 2, [0, 1], color="black", linewidth=4, zorder=10)
            ax.plot([ax.get_xlim()[0]] * 2, [drone.pos.y] * 2, [0, 1], color="black", linewidth=4, zorder=10)
            ax.plot([drone.pos.x] * 2, [ax.get_ylim()[0]] * 2, [0, 1], color="black", linewidth=4, zorder=10)

            ax.set_title(title, fontsize=12)
            ax.set_xlabel("X", fontsize=10)
            ax.set_ylabel("Y", fontsize=10)
            ax.set_zlabel("Recency", fontsize=10)
            ax.invert_yaxis()
            ax.set_xlim(0, SIM_WIDTH)
            ax.set_ylim(SIM_HEIGHT, 0)
            ax.set_zlim(0, 1.5)

        self.fig.suptitle(f"Detection and Position Plot (drone #{drone.drone_id})", fontsize=16, y=0.98)
        self.fig.tight_layout(rect=[0, 0.15, 0.75, 1])

        mappable = ScalarMappable(norm=Normalize(vmin=-math.pi, vmax=math.pi), cmap=cmap)
        mappable.set_array([])
        cbar = self.fig.colorbar(mappable, cax=self.fig.add_axes([0.85, 0.2, 0.02, 0.7]), label="Angle (rad)")
        cbar.set_ticks([-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi])
        cbar.set_ticklabels(["-π", "-π/2", "0", "+π/2", "+π"])

        self.canvas.draw()
        size = self.canvas.get_width_height()
        image = pygame.image.frombytes(bytes(self.canvas.buffer_rgba()), size, "RGBA")
        if size == (GRAPH_WIDTH, GRAPH_HEIGHT):
            return image
        panel = pygame.Surface((GRAPH_WIDTH, GRAPH_HEIGHT))
        panel.blit(image, (0, 0))
        return panel

    def draw_target_lines(self, surface: pygame.Surface) -> None:
        """Draw a line from every enemy to the interest point, labelled with the distance."""
        if self.interest_point is None:
            return
        font = get_font(12)
        center = self.interest_point.center
        for enemy in self.enemy_drones:
            pygame.draw.line(surface, (0, 0, 255), (int(enemy.pos.x), int(enemy.pos.y)),
                             (int(center.x), int(center.y)), 2)
            text = font.render(f"{enemy.pos.distance_to(center):.2f}", True, (255, 255, 255))
            surface.blit(text, text.get_rect(center=((enemy.pos.x + center.x) / 2, (enemy.pos.y + center.y) / 2)))

    def draw_header(
        self,
        surface: pygame.Surface,
        episode: int,
        frame_count: int,
        current_time: datetime,
        friend_drones: List[Any],
        accum_reward: float,
    ) -> None:
        """Draw the episode / communication statistics overlay."""
        green = (0, 255, 0)
        header = (f"[{TYPE_OF_SCENARIO.upper()}] Air Traffic Env - Episode: {episode:4d} | "
                  f"Step: {frame_count:4d} | Accumulated Reward: {accum_reward:8.2f}")
        surface.blit(get_font(16, bold=True).render(header, True, green), (10, 40))

        lines = [
            f"Friend Drones: {len(friend_drones)}",
            f"Enemy Drones: {len(self.enemy_drones)}",
            f"Messages/s: {self.messages_per_second:.1f}",
            f"Connections: {self.active_connections}",
            f"Total distance traveled: {self.total_distance_traveled:.1f} px",
            f"Randomness seed: {self.seed}",
        ]
        for row, text in enumerate(lines):
            surface.blit(self.font.render(text, True, green), (10, 70 + 20 * row))

        duration_str = str(current_time - self.simulation_start_time).split(".")[0]
        time_str = current_time.strftime("%Y-%m-%dT %H:%M:%S.%f")[:-3] + "Z"
        surface.blit(self.font.render(f"{time_str} (duration: {duration_str})", True, green), (10, SIM_HEIGHT - 30))
        surface.blit(self.font.render(" X ", True, (255, 255, 255), green), (SIM_WIDTH // 2, SIM_HEIGHT - 30))
        surface.blit(self.font.render(" Y ", True, (255, 255, 255), green), (SIM_WIDTH - 40, SIM_HEIGHT // 2))

    def render(self, mode: Optional[str] = "human") -> None:
        """Draw one frame (only when ``mode == "human"``)."""
        if mode != "human":
            return

        self.sim_surface.fill((0, 0, 0))
        draw_grid(self.sim_surface)

        if self.show_debug and JOYSTICK is None:
            global_enemy_intensity = np.zeros((GRID_WIDTH, GRID_HEIGHT))
            global_enemy_direction = np.zeros((GRID_WIDTH, GRID_HEIGHT, 2))
            for drone in self.friend_drones:
                global_enemy_intensity = np.maximum(global_enemy_intensity, drone.enemy_intensity)
                global_enemy_direction = np.where(drone.enemy_intensity[..., None] > 0,
                                                  drone.enemy_direction, global_enemy_direction)
            draw_heatmap(self.sim_surface, global_enemy_intensity, "orange")
            draw_direction(self.sim_surface, global_enemy_intensity, global_enemy_direction, 0.1)

        if self.show_dashed_lines and JOYSTICK != "Enemy":
            draw_friend_communication(self.sim_surface, self.friend_drones, show_dashed=self.show_dashed_lines)
        if self.show_target_lines:
            self.draw_target_lines(self.sim_surface)

        friend_args = (self.show_friend_detection_range, self.show_friend_comm_range,
                       self.show_trajectory, self.show_debug)
        for friend in self.friend_drones:
            # When a human flies an enemy, only the friends close to an enemy are visible.
            if JOYSTICK != "Enemy" or any(friend.pos.distance_to(e.pos) <= ENEMY_DETECTION_RANGE
                                          for e in self.enemy_drones):
                friend.draw(self.sim_surface, *friend_args)

        enemy_args = (self.show_enemy_detection_range, self.show_trajectory, self.show_debug)
        for enemy in self.enemy_drones:
            # When a human flies a friend, only triangulated enemies are visible.
            if JOYSTICK != "Friend" or sum(friend.pos.distance_to(enemy.pos) <= FRIEND_DETECTION_RANGE
                                           for friend in self.friend_drones) >= N_LINE_SIGHT_CROSSING:
                enemy.draw(self.sim_surface, *enemy_args)

        self.interest_point.draw(self.sim_surface)
        if self.show_dmz:
            for zone in self.demilitarized_zones:
                zone.draw(self.sim_surface)

        self.draw_header(self.sim_surface, self.episode, self.current_step, self.current_time,
                         self.friend_drones, self.accum_reward)

        if self.show_graph:
            graph_image = self.plot_individual_states()
        else:
            graph_image = pygame.Surface((GRAPH_WIDTH, GRAPH_HEIGHT))
            graph_image.fill((0, 0, 0))
            text = self.font.render("Graph Disabled", True, (255, 255, 255))
            graph_image.blit(text, text.get_rect(center=(GRAPH_WIDTH // 2, GRAPH_HEIGHT // 2)).topleft)

        screen.fill((50, 50, 50))
        screen.blit(self.sim_surface, (0, 0))
        screen.blit(graph_image, (SIM_WIDTH, 0))

        if self.show_debug:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if 0 <= mouse_x <= SIM_WIDTH and 0 <= mouse_y < SIM_HEIGHT:
                message = f"X: {mouse_x:4d} ({mouse_x / SIM_WIDTH:.2f}), Y: {mouse_y:4d} ({mouse_y / SIM_HEIGHT:.2f})"
                screen.blit(self.font.render(message, True, (0, 255, 0)), (SIM_WIDTH - 300, SIM_HEIGHT - 30))

        for button in self.buttons:
            button.draw(screen)

        if self.save_frames:
            if self.folder_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.folder_name = f"./tmp/frames_{timestamp}-{TYPE_OF_SCENARIO}"
                os.makedirs(self.folder_name, exist_ok=True)
            pygame.image.save(screen, os.path.join(self.folder_name, f"frame_{self.frame_number}.png"))
            self.frame_number += 1

        pygame.display.flip()
        clock.tick(self.render_fps)

    def close(self) -> None:
        """Shut pygame down. The configuration cannot be re-initialised afterwards."""
        rendering.clear_caches()
        pygame.quit()

    def handle_events(self) -> None:
        """Process window events: quit, button clicks and drone selection."""
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
        for event in events:
            for button in self.buttons:
                button.handle_event(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.math.Vector2(event.pos)
                for drone in self.friend_drones:
                    if mouse_pos.distance_to(drone.pos) < 10:
                        if self.selected_drone is not None:
                            self.selected_drone.selected = False
                        self.selected_drone = drone
                        drone.selected = True
                        break


def _utcnow() -> datetime:
    """Naive UTC "now" (replacement for the deprecated ``datetime.utcnow``)."""
    return datetime.now(UTC).replace(tzinfo=None)
