# type: ignore
"""
AirTrafficEnv.py

Define o ambiente de simulação completo com TODA a lógica de negócio original.
"""

import os
os.environ["SDL_AUDIODRIVER"] = "dummy"

import sys
import csv
import random
import math
from datetime import datetime, timedelta
from typing import Tuple, List, Any, Optional, Dict

import numpy as np
import pygame

if not pygame.get_init():
    pygame.init()

from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from .FriendDrone import FriendDrone
from .EnemyDrone import EnemyDrone
from .InterestPoint import CircleInterestPoint
from .DemilitarizedZone import CircleDMZ
from .settings import *
from .utils import smooth_matrix_with_kernel_10x10, sim_to_geo, draw_dashed_line
from ..behaviorsDefault import FriendCommonBehaviorDefault, FriendRadarBehaviorDefault, FriendAEWBehaviorDefault

clock: pygame.time.Clock = pygame.time.Clock()


class Button:
    """Botão para UI."""
    
    def __init__(
        self,
        rect: Tuple[int, int, int, int],
        text: str,
        callback: callable,
        toggled: Optional[bool] = None,
        color: Optional[Tuple[int, int, int]] = (70, 70, 70)
    ) -> None:
        self.rect: pygame.Rect = pygame.Rect(rect)
        self.text: str = text
        self.callback: callable = callback
        self.original_color: Tuple[int, int, int] = color
        self.toggled_color: Tuple[int, int, int] = (0, 100, 0)
        self.toggled: Optional[bool] = toggled
        self.font: pygame.font.Font = pygame.font.SysFont(FONT_FAMILY, 11)

    def darken_color(self, color: Tuple[int, int, int], amount: int = 30) -> Tuple[int, int, int]:
        return tuple(max(c - amount, 0) for c in color)
        
    def draw(self, surface: pygame.Surface) -> None:
        mouse_pos: Tuple[int, int] = pygame.mouse.get_pos()
        
        if self.toggled is not None:
            base_color = self.toggled_color if self.toggled else self.original_color
            color = self.darken_color(base_color) if self.rect.collidepoint(mouse_pos) else base_color
        else:
            color = self.original_color
        
        pygame.draw.rect(surface, color, self.rect)
        label = self.font.render(self.text, True, (255, 255, 255))
        surface.blit(label, (self.rect.x + 5, self.rect.y + 10))
        
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                if self.toggled is not None:
                    self.toggled = not self.toggled
                self.callback()


def draw_grid(surface: pygame.Surface) -> None:
    """Desenha grid."""
    grid_color: Tuple[int, int, int] = (15, 15, 15)
    for x in range(0, SIM_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, grid_color, (x, 0), (x, SIM_HEIGHT), 1)
    for y in range(0, SIM_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, grid_color, (0, y), (SIM_WIDTH, y), 1)


def draw_heatmap(surface: pygame.Surface, global_intensity: np.ndarray, base_color: str) -> None:
    """Desenha heatmap."""
    for i in range(GRID_WIDTH):
        for j in range(GRID_HEIGHT):
            intensity: float = global_intensity[i, j]
            if intensity > 0.01:
                if base_color == "red":
                    color: Tuple[int, int, int] = (int(intensity * 255), 0, 0)
                elif base_color == "orange":
                    color = (int(intensity * 255), int(intensity * 165), 0)
                else:
                    color = (0, 0, int(intensity * 255))
                rect = pygame.Rect(i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(surface, color, rect)


def draw_friend_communication(
    surface: pygame.Surface,
    friend_drones: List[Any],
    show_dashed: bool = True
) -> None:
    """Desenha comunicação entre drones."""
    if not show_dashed:
        return
    for drone in friend_drones:
        for nbr in getattr(drone, "neighbors", []):
            if drone.drone_id < nbr.drone_id:
                draw_dashed_line(
                    surface,
                    (255, 255, 255, 128),
                    drone.pos,
                    nbr.pos,
                    width=2,
                    dash_length=5,
                    space_length=5
                )


def draw_direction(
    surface: pygame.Surface,
    global_intensity: np.ndarray,
    global_direction: np.ndarray,
    threshold: float
) -> None:
    """Desenha direções."""
    for i in range(GRID_WIDTH):
        for j in range(GRID_HEIGHT):
            intensity: float = global_intensity[i, j]
            if intensity > threshold:
                center_x: float = i * CELL_SIZE + CELL_SIZE / 2
                center_y: float = j * CELL_SIZE + CELL_SIZE / 2
                dir_vec = pygame.math.Vector2(*global_direction[i, j])
                if dir_vec.length() > 0:
                    dir_vec = dir_vec.normalize() * 5
                    end_x: float = center_x + dir_vec.x
                    end_y: float = center_y + dir_vec.y
                    pygame.draw.line(surface, (255, 255, 255), (center_x, center_y), (end_x, end_y), 1)


def compute_global_matrices(
    drones: List[Any],
    intensity_attr: str,
    direction_attr: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula matrizes globais."""
    global_intensity = np.zeros((GRID_WIDTH, GRID_HEIGHT))
    global_direction = np.zeros((GRID_WIDTH, GRID_HEIGHT, 2))
    for i in range(GRID_WIDTH):
        for j in range(GRID_HEIGHT):
            best_intensity: float = 0.0
            best_dir = np.array([0.0, 0.0])
            for drone in drones:
                intensity: float = getattr(drone, intensity_attr)[i, j]
                if intensity > best_intensity:
                    best_intensity = intensity
                    best_dir = getattr(drone, direction_attr)[i, j]
            global_intensity[i, j] = best_intensity
            global_direction[i, j] = best_dir
    return global_intensity, global_direction


class AirTrafficEnv:
    """Ambiente de simulação completo."""
    
    def __init__(
        self,
        max_steps: int = 5_000,
        mode: Optional[str] = 'human',
        friend_behavior: Optional[str] = None,
        friend_aew_behavior: Optional[str] = None,
        friend_radar_behavior: Optional[str] = None,
        enemy_behavior = None,
        demilitarized_zones: List[Tuple[float, float, float]] = None,
        seed: Optional[int] = None
    ) -> None:
        """Inicializa ambiente."""
        self.seed: int = seed if seed is not None else random.randint(0, 10000000)
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.RandomState(self.seed)
        
        self.simulation_start_time: datetime = datetime.utcnow()
        self.current_time: datetime = datetime.utcnow()
        self.max_steps: int = max_steps
        self.mode = mode
        self.friend_behavior = friend_behavior
        self.friend_aew_behavior = friend_aew_behavior
        self.friend_radar_behavior = friend_radar_behavior
        self.enemy_behavior = enemy_behavior
        self.folder_name = None
        
        self.episode: int = 0
        self.current_step: int = 0
        self.friend_drones: List[FriendDrone] = []
        self.enemy_drones: List[EnemyDrone] = []
        self.interest_point = None
        self.selected_drone = None
        self.leader = None
        
        if self.friend_behavior is None:
            self.friend_behavior = FriendCommonBehaviorDefault()
        if self.friend_aew_behavior is None:
            self.friend_aew_behavior = FriendAEWBehaviorDefault()
        if self.friend_radar_behavior is None:
            self.friend_radar_behavior = FriendRadarBehaviorDefault()
        
        surface: pygame.Surface = pygame.Surface((SIM_WIDTH, SIM_HEIGHT))
        self.sim_surface = surface.convert_alpha()
        self.sim_surface.fill((0, 0, 0))
        
        self.fig: Optional[Figure] = None
        self.canvas: Optional[FigureCanvas] = None
        self.accum_reward: float = 0.0
        self.attack_penalty: int = 0
        self.sucessful_attacks: int = 0
        self.font: pygame.font.Font = pygame.font.SysFont(FONT_FAMILY, 14, bold=False)

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
        
        self.messages_exchanged: int = 0
        self.messages_count_interval: List[float] = []
        self.last_message_count_time: int = pygame.time.get_ticks()
        self.messages_per_second: float = 0.0
        self.active_connections: int = 0
        self.total_distance_traveled: float = 0.0
        
        self.demilitarized_zones: List[CircleDMZ] = []
        if demilitarized_zones:
            for x, y, radius in demilitarized_zones:
                zone_center = pygame.math.Vector2(x, y)
                if zone_center.distance_to(INTEREST_POINT_CENTER) > INTERNAL_RADIUS + radius:
                    self.demilitarized_zones.append(CircleDMZ(zone_center, radius, seed=self.seed))
        
        self._setup_ui_buttons()

    def _setup_ui_buttons(self) -> None:
        """Setup buttons."""
        button_width: int = 130
        button_height: int = 30
        button_spacing: int = 5
        row1_y: int = GRAPH_HEIGHT - 170
        row2_y: int = GRAPH_HEIGHT - 130
        row3_y: int = GRAPH_HEIGHT - 90
        row4_y: int = GRAPH_HEIGHT - 50
        graph_x: int = SIM_WIDTH
        
        self.buttons: List[Button] = []
        self.buttons.append(Button((graph_x + 10, row1_y, button_width, button_height), "Tog. Graph", self.toggle_graph, toggled=False))
        self.buttons.append(Button((graph_x + 10 + (button_width + button_spacing), row1_y, button_width, button_height), "Pause", self.toggle_pause, toggled=False))
        self.buttons.append(Button((graph_x + 10 + 2*(button_width + button_spacing), row1_y, button_width, button_height), "Reset", self.reset_simulation))
        self.buttons.append(Button((graph_x + 10 + 3*(button_width + button_spacing), row1_y, button_width, button_height), "Exit", self.exit_env, color=(200, 0, 0)))
        self.buttons.append(Button((graph_x + 10, row2_y, button_width, button_height), "Tog. Friend Range", self.toggle_friend_range, toggled=True))
        self.buttons.append(Button((graph_x + 10 + (button_width + button_spacing), row2_y, button_width, button_height), "Tog. Enemy Range", self.toggle_enemy_range, toggled=True))
        self.buttons.append(Button((graph_x + 10 + 2*(button_width + button_spacing), row2_y, button_width, button_height), "Tog. Friend Comm.", self.toggle_dashed, toggled=True))
        self.buttons.append(Button((graph_x + 10 + 3*(button_width + button_spacing), row2_y, button_width, button_height), "Tog. DMZ", self.toggle_dmz, toggled=True))
        self.buttons.append(Button((graph_x + 10, row3_y, button_width, button_height), "Tog. Comm Range", self.toggle_comm_range, toggled=False))
        self.buttons.append(Button((graph_x + 10 + (button_width + button_spacing), row3_y, button_width, button_height), "Export Tacview", self.toggle_tacview_export, toggled=False))
        self.buttons.append(Button((graph_x + 10 + 2*(button_width + button_spacing), row3_y, button_width, button_height), "Tog. Save Frames", self.toogle_save_frames, toggled=False))
        self.buttons.append(Button((graph_x + 10 + 3*(button_width + button_spacing), row3_y, button_width, button_height), "Tog. Target Lines", self.toggle_target_lines, toggled=False))
        self.buttons.append(Button((graph_x + 10, row4_y, button_width, button_height), "Tog. Trajetory", self.toggle_trajetory, toggled=True))
        self.buttons.append(Button((graph_x + 10 + (button_width + button_spacing), row4_y, button_width, button_height), "Tog. Debug", self.toggle_debug, toggled=True))
        self.buttons.append(Button((graph_x + 10 + 2*(button_width + button_spacing), row4_y, button_width, button_height), "Tog. D. Passive.", self.toogle_triangulation, toggled=PASSIVE_DETECTION))
        self.buttons.append(Button((graph_x + 10 + 3*(button_width + button_spacing), row4_y, button_width, button_height), "Tog. Return", self.toogle_return, toggled=False))

    def is_in_demilitarized_zone(self, position: pygame.math.Vector2) -> bool:
        """Verifica DMZ."""
        for zone in self.demilitarized_zones:
            if position.distance_to(zone.center) <= zone.radius:
                return True
        return False

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
        self.show_dmz = not self.show_dmz
    def toggle_target_lines(self) -> None:
        self.show_target_lines = not self.show_target_lines
    def toogle_save_frames(self) -> None:
        self.save_frames = not self.save_frames
    def toggle_comm_range(self) -> None:
        self.show_friend_comm_range = not self.show_friend_comm_range
    def toggle_trajetory(self) -> None:
        self.show_trajectory = not self.show_trajectory
    def toggle_tacview_export(self) -> None:
        self.export_to_tacview = not self.export_to_tacview
    def toggle_debug(self) -> None:
        self.show_debug = not self.show_debug
    def toogle_return(self) -> None:
        self.return_to_base = not self.return_to_base
    def toogle_triangulation(self) -> None:
        self.use_triangulation = not self.use_triangulation

    def handle_tacview(self) -> None:
        """Exporta Tacview."""
        date_time: str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath: str = f"./output/tacview_export_{date_time}"
        os.makedirs(filepath, exist_ok=True)
        for drone_name, log in self.tacview_logs.items():
            color: str = "Green" if "Friend" in drone_name.split("_")[0] else "Red"
            filename: str = os.path.join(filepath, f" ({drone_name}) [{color}].csv")
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Time", "Longitude", "Latitude", "Altitude", "Roll (deg)", "Pitch (deg)", "Yaw (deg)"])
                for record in log:
                    current_time, name, pos_x, pos_y, _, vel_x, vel_y, _ = record
                    time: str = current_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4] + "Z"
                    lon, lat = sim_to_geo(pos_x, pos_y)
                    altitude: float = 1000.0
                    roll: float = 0.0
                    pitch: float = 0.0
                    yaw: float = math.degrees(math.atan2(vel_y, vel_x)) if (vel_x != 0 or vel_y != 0) else 0.0
                    writer.writerow([time, lon, lat, altitude, roll, pitch, yaw])
            print(f"Simulation exported to {filename} for Tacview.")

    def _get_observation(self) -> np.ndarray:
        """Observação."""
        obs: List[float] = []
        for drone in self.friend_drones:
            obs.extend([drone.pos.x, drone.pos.y, drone.vel.x, drone.vel.y])
        for enemy in self.enemy_drones:
            obs.extend([enemy.pos.x, enemy.pos.y, enemy.vel.x, enemy.vel.y])
        return np.array(obs, dtype=np.float32)

    def reset(self) -> Tuple[np.ndarray, bool]:
        """Reset completo."""
        if hasattr(self, 'fig') and self.fig is not None:
            plt.close(self.fig)
        
        self.fig = Figure(figsize=(GRAPH_WIDTH / 100, GRAPH_HEIGHT / 100), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        
        self.simulation_start_time = datetime.utcnow()
        self.current_time = datetime.utcnow()
        self.episode += 1
        self.current_step = 0
        self.accum_reward = 0.0
        self.attack_penalty = 0
        self.sucessful_attacks = 0
        self.leader = None
        self.selected_drone = None
        self.tacview_logs: Dict[str, List] = {}
        self.friend_drones = []
        self.enemy_drones = []
        
        FriendDrone.friend_id_counter = 0
        EnemyDrone.enemy_id_counter = 0
        
        center = CENTER
        self.interest_point = CircleInterestPoint(CENTER, INTERNAL_RADIUS, EXTERNAL_RADIUS, seed=self.seed)
        
        FriendDrone.set_class_seed(seed=self.seed)
        
        for i in range(RADAR_COUNT):
            angle: float = 2 * math.pi * i / RADAR_COUNT
            pos = center + pygame.math.Vector2(RADAR_RANGE * math.cos(angle), RADAR_RANGE * math.sin(angle))
            drone = FriendDrone(self.interest_point.center, position=(pos.x, pos.y), 
                              behavior=self.friend_radar_behavior, fixed=True)
            self.friend_drones.append(drone)
            
        for j in range(AEW_COUNT):
            angle = 2 * math.pi * j / AEW_COUNT
            pos = center + pygame.math.Vector2(AEW_RANGE * math.cos(angle), AEW_RANGE * math.sin(angle))
            drone = FriendDrone(self.interest_point.center, position=(pos.x, pos.y), 
                              behavior=self.friend_aew_behavior)
            self.friend_drones.append(drone)
            
        for k in range(FRIEND_COUNT - AEW_COUNT - RADAR_COUNT):
            angle = (2 * INITIAL_N_LAYERS * math.pi * k) / (FRIEND_COUNT - AEW_COUNT - RADAR_COUNT)
            layer: int = k // ((FRIEND_COUNT - AEW_COUNT - RADAR_COUNT) // INITIAL_N_LAYERS)
                
            pos = center + pygame.math.Vector2(
                (INITIAL_DISTANCE + layer * 40) * math.cos(angle),
                (INITIAL_DISTANCE + layer * 40) * math.sin(angle)
            )
            
            drone = FriendDrone(
                self.interest_point.center,
                position=(pos.x, pos.y),
                behavior=self.friend_behavior,
                broken=(k < BROKEN_COUNT)
            )
                
            if k == 0:
                self.selected_drone = drone
                self.leader = drone
                drone.is_leader = True
                drone.selected = True
                if JOYSTICK == "Friend":
                    drone.joystick_controlled = True
                
            self.friend_drones.append(drone)
        
        _enemy_count: int = ENEMY_COUNT
        self.enemy_drones = []
        
        if JOYSTICK == "Enemy":
            self.enemy_drones.append(EnemyDrone(self.interest_point.center, behavior_type='joystick'))
            _enemy_count -= 1
            
        EnemyDrone.set_class_seed(seed=self.seed)
        self.enemy_drones.extend([
            EnemyDrone(self.interest_point.center, behavior_type=self.enemy_behavior)
            for _ in range(_enemy_count)
        ])
        
        done: bool = self.current_step >= self.max_steps or len(self.enemy_drones) == 0
        return self._get_observation(), done

    def plot_individual_states(self) -> pygame.Surface:
        """Plot 3D."""
        self.fig.clear()
        cmap = plt.cm.hsv

        if self.selected_drone is not None:
            drone = self.selected_drone
        elif self.friend_drones:
            drone = self.friend_drones[0]
        else:
            return pygame.Surface((GRAPH_WIDTH, GRAPH_HEIGHT))

        for idx, plot_view in enumerate(["Enemy Detection", "Friend Detection"]):
            ax = self.fig.add_subplot(2, 1, idx + 1, projection='3d')
            ax.set_box_aspect((1, 1, 0.4))
            
            x = np.linspace(CELL_SIZE / 2, SIM_WIDTH - CELL_SIZE / 2, GRID_WIDTH)
            y = np.linspace(CELL_SIZE / 2, SIM_HEIGHT - CELL_SIZE / 2, GRID_HEIGHT)
            X, Y = np.meshgrid(x, y)
            
            if plot_view == "Enemy Detection":
                to_plot = drone.enemy_intensity.T
                direction = drone.enemy_direction
            elif plot_view == "Friend Detection":
                to_plot = drone.friend_intensity.T
                direction = drone.friend_direction

            Z_smoothed, result_direction = smooth_matrix_with_kernel_10x10(
                to_plot, np.transpose(direction, (1, 0, 2)), sigma=2, flat_radius=1
            )

            dir_x = result_direction[..., 0]
            dir_y = result_direction[..., 1]
            angle = np.arctan2(dir_y, dir_x)
            norm_angle = (angle + math.pi) / (2 * math.pi)
            facecolors = cmap(norm_angle)
            facecolors[Z_smoothed < PLOT_THRESHOLD] = [1, 1, 1, 1]
            
            ax.plot_surface(X, Y, Z_smoothed, facecolors=facecolors,
                          linewidth=0, antialiased=True, shade=False)
            ax.contourf(X, Y, Z_smoothed, zdir='x', offset=ax.get_xlim()[0], cmap="Grays")
            ax.contourf(X, Y, Z_smoothed, zdir='y', offset=ax.get_ylim()[0], cmap="Grays")
            ax.plot_wireframe(X, Y, Z_smoothed, color='black', linewidth=0.2, rstride=1, cstride=1)
            ax.plot([drone.pos.x, drone.pos.x], [drone.pos.y, drone.pos.y], [0, 1],
                   color='black', linewidth=4, zorder=10)
            
            x_offset = ax.get_xlim()[0]
            ax.plot([x_offset, x_offset], [drone.pos.y, drone.pos.y], [0, 1],
                   color='black', linewidth=4, zorder=10)
            y_offset = ax.get_ylim()[0]
            ax.plot([drone.pos.x, drone.pos.x], [y_offset, y_offset], [0, 1],
                   color='black', linewidth=4, zorder=10)
            
            ax.set_title(plot_view, fontsize=12)
            ax.set_xlabel('X', fontsize=10)
            ax.set_ylabel('Y', fontsize=10)
            ax.set_zlabel('Recency', fontsize=10)
            ax.invert_yaxis()
            ax.set_xlim(0, SIM_WIDTH)
            ax.set_ylim(SIM_HEIGHT, 0)
            ax.set_zlim(0, 1.5)
        
        self.fig.suptitle(f"Detection and Position Plot (drone #{drone.drone_id})", fontsize=16, y=0.98)
        self.fig.tight_layout(rect=[0, 0.15, 0.75, 1])
        
        norm = plt.Normalize(vmin=-math.pi, vmax=math.pi)
        mappable_for_colorbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable_for_colorbar.set_array([])
        cbar_ax = self.fig.add_axes([0.85, 0.2, 0.02, 0.7])
        cbar = self.fig.colorbar(mappable_for_colorbar, cax=cbar_ax, label='Angle (rad)')
        cbar.set_ticks([-math.pi, -math.pi/2, 0, math.pi/2, math.pi])
        cbar.set_ticklabels(['-π', '-π/2', '0', '+π/2', '+π'])
        
        self.canvas.draw()
        graph_string = self.canvas.tostring_rgb()
        return pygame.image.frombuffer(graph_string, (GRAPH_WIDTH, GRAPH_HEIGHT), "RGB")
    
    def draw_target_lines(self, surface: pygame.Surface) -> None:
        """Target lines."""
        font = pygame.font.SysFont(FONT_FAMILY, 12)
        if self.interest_point is not None:
            for enemy in self.enemy_drones:
                start_pos = (int(enemy.pos.x), int(enemy.pos.y))
                end_pos = (int(self.interest_point.center.x), int(self.interest_point.center.y))
                pygame.draw.line(surface, (0, 0, 255), start_pos, end_pos, 2)
                dist: float = enemy.pos.distance_to(self.interest_point.center)
                mid_x: float = (enemy.pos.x + self.interest_point.center.x) / 2
                mid_y: float = (enemy.pos.y + self.interest_point.center.y) / 2
                distance_text = font.render(f"{dist:.2f}", True, (255, 255, 255))
                text_rect = distance_text.get_rect(center=(mid_x, mid_y))
                surface.blit(distance_text, text_rect)

    def step(self, action: List[float]) -> Tuple[np.ndarray, float, bool, dict]:
        """Step COMPLETO com toda lógica original."""
        self.handle_events()
        frame_reward: float = 0.0
        
        if not self.paused:
            self.messages_exchanged = 0
            self.active_connections = 0
            self.total_distance_traveled = 0
            
            self.current_time += timedelta(seconds=DT_STEP)
            
            for drone in self.friend_drones:
                drone.update(self.enemy_drones, self.friend_drones, self.use_triangulation, self.return_to_base)
                self.messages_exchanged += getattr(drone, 'messages_sent_this_cycle', 0)
                self.active_connections += getattr(drone, 'active_connections', 0)
                self.total_distance_traveled += drone.distance_traveled
                
                if self.return_to_base and drone.pos.distance_to(CENTER) <= EPSILON:
                    self.friend_drones = [f for f in self.friend_drones if drone.drone_id != f.drone_id]
                    if drone.is_leader:
                        self.leader = None
                    if drone.selected:
                        self.selected_drone = None
                    
            for enemy in self.enemy_drones:
                detector = None
                for friend in self.friend_drones:
                    if enemy.pos.distance_to(friend.pos) <= ENEMY_DETECTION_RANGE:
                        detector = friend.pos
                        break
                enemy.update(detector)
                
            current_time: int = pygame.time.get_ticks()
            time_elapsed: float = (current_time - self.last_message_count_time) / 1000
            
            self.messages_exchanged /= 2
            self.active_connections /= 2
            
            if time_elapsed >= 1e-1:
                msgs_per_sec: float = self.messages_exchanged / time_elapsed
                self.messages_count_interval.append(msgs_per_sec)
                if len(self.messages_count_interval) > 5:
                    self.messages_count_interval.pop(0)
                self.messages_per_second = sum(self.messages_count_interval) / len(self.messages_count_interval)
                self.last_message_count_time = current_time
                
            frame_reward = self.compute_reward()
            self.accum_reward += frame_reward
            self.current_step += 1

            surviving_enemies: List[EnemyDrone] = []
            for enemy in self.enemy_drones:
                if enemy.pos.distance_to(self.interest_point.center) <= INTEREST_POINT_ATTACK_RANGE + EPSILON:
                    self.interest_point.health -= INTEREST_POINT_DAMAGE
                    self.attack_penalty += 1
                    self.sucessful_attacks += 1
                    radar_to_remove = next(
                        (d for d in self.friend_drones 
                        if hasattr(d, 'behavior') and d.behavior and d.behavior.type == "RADAR"), 
                        None
                    )
                    if radar_to_remove:
                        self.friend_drones.remove(radar_to_remove)
                        if radar_to_remove.is_leader:
                            self.leader = None
                        if radar_to_remove.selected:
                            self.selected_drone = None
                else:
                    surviving_enemies.append(enemy)
                    
            self.enemy_drones = surviving_enemies
            if self.interest_point.health < 0:
                self.interest_point.health = 0

            friends_to_remove: set = set()
            enemies_to_remove: set = set()
            
            for i, friend in enumerate(self.friend_drones):
                for j, enemy in enumerate(self.enemy_drones):
                    if self.is_in_demilitarized_zone(enemy.pos) and self.show_dmz:
                        continue
                    
                    if friend.pos.distance_to(enemy.pos) <= NEUTRALIZATION_RANGE:
                        rand_val: float = self.rng.random()
                        
                        if (getattr(friend, 'behavior', None) and 
                            getattr(friend.behavior, 'type', None) == "AEW" and 
                            rand_val <= NEUTRALIZATION_PROB_BOTH_DEAD):
                            friends_to_remove.add(i)
                            enemies_to_remove.add(j)
                            if friend.is_leader:
                                self.leader = None
                            if friend.selected:
                                self.selected_drone = None
                                
                        if (getattr(friend, 'behavior', None) and 
                            getattr(friend.behavior, 'type', None) == "RADAR" and 
                            rand_val <= NEUTRALIZATION_PROB_BOTH_DEAD):
                            friends_to_remove.add(i)
                            enemies_to_remove.add(j)
                            if friend.is_leader:
                                self.leader = None
                            if friend.selected:
                                self.selected_drone = None
                        else:
                            if rand_val <= NEUTRALIZATION_PROB_FRIEND_ALIVE:
                                enemies_to_remove.add(j)
                            elif rand_val <= NEUTRALIZATION_PROB_FRIEND_ALIVE + NEUTRALIZATION_PROB_ENEMY_ALIVE:
                                friends_to_remove.add(i)
                                if friend.is_leader:
                                    self.leader = None
                                if friend.selected:
                                    self.selected_drone = None
                            else:
                                friends_to_remove.add(i)
                                enemies_to_remove.add(j)
                                if friend.is_leader:
                                    self.leader = None
                                if friend.selected:
                                    self.selected_drone = None
                                
            self.friend_drones = [f for idx, f in enumerate(self.friend_drones) if idx not in friends_to_remove]
            self.enemy_drones = [e for idx, e in enumerate(self.enemy_drones) if idx not in enemies_to_remove]
            
            if self.selected_drone is None and len(self.friend_drones) > 0:
                self.selected_drone = self.friend_drones[0]
                self.selected_drone.selected = True
            if self.leader is None and len(self.friend_drones) > 0:
                self.leader = self.friend_drones[0]
                self.leader.is_leader = True

            for drone in self.friend_drones:
                name: str = f"Friend_{drone.drone_id}"
                if name not in self.tacview_logs:
                    self.tacview_logs[name] = []
                self.tacview_logs[name].append((
                    self.current_time, name,
                    drone.pos.x, drone.pos.y, 0,
                    drone.vel.x, drone.vel.y, 0
                ))
            for drone in self.enemy_drones:
                name = f"Enemy_{drone.drone_id}"
                if name not in self.tacview_logs:
                    self.tacview_logs[name] = []
                self.tacview_logs[name].append((
                    self.current_time, name,
                    drone.pos.x, drone.pos.y, 0,
                    drone.vel.x, drone.vel.y, 0
                ))
                
        self.render(self.mode)
        obs: np.ndarray = self._get_observation()
        done: bool = self.current_step >= self.max_steps or len(self.enemy_drones) == 0

        state_percentages: Dict[str, float] = {}
        for drone in self.friend_drones:
            drone_percentages = drone.get_state_percentages()
            for state, percentage in drone_percentages.items():
                if state in state_percentages:
                    state_percentages[state] += percentage / len(self.friend_drones)
                else:
                    state_percentages[state] = percentage / len(self.friend_drones)
        
        # CRÍTICO: info dict COMPLETO com TODOS os campos originais
        info: dict = {
            "current_step": self.current_step,
            "accum_reward": self.accum_reward,
            "enemies_shotdown": ENEMY_COUNT - len(self.enemy_drones),
            "friends_shotdown": FRIEND_COUNT - len(self.friend_drones),
            "sucessful_attacks": self.sucessful_attacks,
            "interest_point_health": self.interest_point.health,
            "state_percentages": state_percentages,
            "total_distance_traveled": self.total_distance_traveled
        }
        
        if done and self.export_to_tacview:
            self.handle_tacview()
            
        return obs, frame_reward, done, info

    def draw_header(
        self,
        surface: pygame.Surface,
        episode: int,
        frame_count: int,
        current_time: datetime,
        friend_drones: List[Any],
        accum_reward: float
    ) -> None:
        """Draw header."""
        header_font = pygame.font.SysFont(FONT_FAMILY, 16, bold=True)
        header_text = f"[{TYPE_OF_SCENARIO.upper()}] Air Traffic Env - Episode: {episode:4d} | Step: {frame_count:4d} | Accumulated Reward: {accum_reward:8.2f}"
        header_label = header_font.render(header_text, True, (0, 255, 0))
        surface.blit(header_label, (10, 40))
        
        friend_count = f"Friend Drones: {len(friend_drones)}"
        friend_label = self.font.render(friend_count, True, (0, 255, 0))
        surface.blit(friend_label, (10, 70))
        
        enemy_count = f"Enemy Drones: {len(self.enemy_drones)}"
        enemy_label = self.font.render(enemy_count, True, (0, 255, 0))
        surface.blit(enemy_label, (10, 90))
                
        msg_text = f"Messages/s: {self.messages_per_second:.1f}"
        msg_label = self.font.render(msg_text, True, (0, 255, 0))
        surface.blit(msg_label, (10, 110))
        
        conn_text = f"Connections: {self.active_connections}"
        conn_label = self.font.render(conn_text, True, (0, 255, 0))
        surface.blit(conn_label, (10, 130))
        
        dist_text = f"Total distance traveled: {self.total_distance_traveled:.1f} px"
        dist_label = self.font.render(dist_text, True, (0, 255, 0))
        surface.blit(dist_label, (10, 150))
        
        seed_text = f"Randomness seed: {self.seed}"
        seed_label = self.font.render(seed_text, True, (0, 255, 0))
        surface.blit(seed_label, (10, 170))
        
        duration = current_time - self.simulation_start_time
        duration_str = str(duration).split(".")[0]
        time_str = current_time.strftime('%Y-%m-%dT %H:%M:%S.%f')[:-3] + "Z"
        header_time = self.font.render(f"{time_str} (duration: {duration_str})", True, (0, 255, 0))
        surface.blit(header_time, (10, SIM_HEIGHT - 30))
        
        x_label = self.font.render(" X ", True, (255, 255, 255), (0, 255, 0))
        surface.blit(x_label, (SIM_WIDTH // 2, SIM_HEIGHT - 30))
        
        y_label = self.font.render(" Y ", True, (255, 255, 255), (0, 255, 0))
        surface.blit(y_label, (SIM_WIDTH - 40, SIM_HEIGHT // 2))

    def render(self, mode: str = "human") -> None:
        """Render completo."""
        if mode == 'human':
            self.sim_surface.fill((0, 0, 0))
            draw_grid(self.sim_surface)

            if self.show_debug and JOYSTICK is None:
                # global_triangulation = np.zeros((GRID_WIDTH * TRIANGULATION_GRANULARITY, GRID_HEIGHT * TRIANGULATION_GRANULARITY))
                global_enemy_intensity = np.zeros((GRID_WIDTH, GRID_HEIGHT))
                global_enemy_direction = np.zeros((GRID_WIDTH, GRID_HEIGHT, 2))
                
                for drone in self.friend_drones:
                    # global_triangulation = np.add(global_triangulation, drone.passive_detection_matrix)
                    global_enemy_intensity = np.maximum(global_enemy_intensity, drone.enemy_intensity)
                    global_enemy_direction = np.where(np.expand_dims(drone.enemy_intensity, axis=2) > 0,
                                                    drone.enemy_direction, global_enemy_direction)

                draw_heatmap(self.sim_surface, global_enemy_intensity, "orange")
                draw_direction(self.sim_surface, global_enemy_intensity, global_enemy_direction, 0.1)
            
            if self.show_dashed_lines and JOYSTICK != "Enemy":
                draw_friend_communication(self.sim_surface, self.friend_drones, show_dashed=self.show_dashed_lines)
                
            if self.show_target_lines:
                self.draw_target_lines(self.sim_surface)
            
            if JOYSTICK == "Enemy":
                for friend in self.friend_drones:
                    for enemy in self.enemy_drones:
                        if friend.pos.distance_to(enemy.pos) <= ENEMY_DETECTION_RANGE:
                            friend.draw(self.sim_surface, self.show_friend_detection_range, 
                                      self.show_friend_comm_range, self.show_trajectory, self.show_debug)
                            break
            else:
                for friend in self.friend_drones:
                    friend.draw(self.sim_surface, self.show_friend_detection_range, 
                              self.show_friend_comm_range, self.show_trajectory, self.show_debug)
                
            if JOYSTICK == "Friend":
                for enemy in self.enemy_drones:
                    _count: int = 0
                    for friend in self.friend_drones:
                        if friend.pos.distance_to(enemy.pos) <= FRIEND_DETECTION_RANGE:
                            _count += 1
                        if _count == N_LINE_SIGHT_CROSSING:
                            enemy.draw(self.sim_surface, self.show_enemy_detection_range, 
                                     self.show_trajectory, self.show_debug)
                            break
            else:
                for enemy in self.enemy_drones:
                    enemy.draw(self.sim_surface, self.show_enemy_detection_range, 
                              self.show_trajectory, self.show_debug)
                
            self.interest_point.draw(self.sim_surface)
            
            if self.show_dmz:
                for zone in self.demilitarized_zones:
                    zone.draw(self.sim_surface)
                
            self.draw_header(self.sim_surface, self.episode, self.current_step, 
                           self.current_time, self.friend_drones, self.accum_reward)
            
            if self.show_graph:
                graph_image = self.plot_individual_states()
            else:
                graph_image = pygame.Surface((GRAPH_WIDTH, GRAPH_HEIGHT))
                graph_image.fill((0, 0, 0))
                text = self.font.render("Graph Disabled", True, (255, 255, 255))
                graph_image.blit(text, (GRAPH_WIDTH // 2 - text.get_width() // 2, 
                                       GRAPH_HEIGHT // 2 - text.get_height() // 2))
                
            screen.fill((50, 50, 50))
            screen.blit(self.sim_surface, (0, 0))
            screen.blit(graph_image, (SIM_WIDTH, 0))
            
            if self.show_debug:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                rel_x: float = mouse_x / SIM_WIDTH
                rel_y: float = mouse_y / SIM_HEIGHT
                if 0 <= mouse_x <= SIM_WIDTH and 0 <= mouse_y < SIM_HEIGHT:
                    message = f"X: {mouse_x:4d} ({rel_x:.2f}), Y: {mouse_y:4d} ({rel_y:.2f})"
                    coords_text = self.font.render(message, True, (0, 255, 0))
                    screen.blit(coords_text, (SIM_WIDTH - 300, SIM_HEIGHT - 30))
    
            for button in self.buttons:
                button.draw(screen)
            
            if self.save_frames:
                if self.folder_name is None:
                    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.folder_name = f"./tmp/frames_{timestamp}-{TYPE_OF_SCENARIO}"
                    os.makedirs(self.folder_name, exist_ok=True)
                filename: str = os.path.join(self.folder_name, f"frame_{self.frame_number}.png")
                pygame.image.save(screen, filename)
                self.frame_number += 1
            
            pygame.display.flip()
            clock.tick(30)

    def close(self) -> None:
        """Close."""
        pygame.quit()

    def handle_events(self) -> None:
        """Events."""
        events: List[pygame.event.Event] = pygame.event.get()
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
                        self.selected_drone.selected = False
                        self.selected_drone = drone
                        self.selected_drone.selected = True
                        break

    def compute_reward(
        self,
        normilizer_factor: int = 1/10_000,
        bonus_factor: int = 10,
        penalty_factor: int = 2000
    ) -> float:
        """Reward."""
        total_distance: float = sum(
            enemy.pos.distance_to(self.interest_point.center) 
            for enemy in self.enemy_drones
        )
        normalized_total_distance: float = total_distance * normilizer_factor
        max_distance: float = self.interest_point.center.distance_to(pygame.math.Vector2(0, 0))
        bonus_killed_enemy_earlier: float = (
            bonus_factor * (ENEMY_COUNT - len(self.enemy_drones)) * max_distance
        ) * normilizer_factor
        penalty_value: float = penalty_factor * self.attack_penalty
        self.attack_penalty = 0
        return normalized_total_distance + bonus_killed_enemy_earlier - penalty_value