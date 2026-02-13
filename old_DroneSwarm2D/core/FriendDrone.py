# type: ignore
"""
FriendDrone.py

This module defines the FriendDrone class used in the simulation. The FriendDrone class handles
local detection (enemy and friend), communication, merging of detection matrices,
triangulation of targets, action execution, and rendering. It also provides planning and
debug behaviors for drone motion.
"""

# -----------------------------------------------------------------------------
# Imports and Setup
# -----------------------------------------------------------------------------
import random
import math
import os
import sys
import numpy as np
import pygame
import itertools

from typing import Optional, Tuple, List, Any, Dict
from pathlib import Path

from .settings import *
from .utils import (
    draw_dashed_circle, 
    load_svg_as_surface, 
    pos_to_cell, 
    intercept_direction, 
    generate_sparse_matrix, 
    draw_dashed_line
)

# Obter o diretório deste arquivo
_MODULE_DIR = Path(__file__).parent

# Construir caminhos absolutos para os assets
_ASSETS_DIR = _MODULE_DIR.parent / "assets"  # DroneSwarm2D/assets/


# -----------------------------------------------------------------------------
# FriendDrone Class Definition
# -----------------------------------------------------------------------------
class FriendDrone:
    """
    FriendDrone represents a friendly drone in the simulation environment.
    
    This class handles detection of enemy and friendly drones, communication
    between drones, movement behaviors, and visualization.
    """

    # Class variables
    friend_id_counter: int = 0
    original_drone_image: pygame.Surface = load_svg_as_surface(f"{_ASSETS_DIR}/drone_0.svg")
    original_broken_drone_image: pygame.Surface = load_svg_as_surface(f"{_ASSETS_DIR}/drone_broken.svg")
    original_aew_image: pygame.Surface = load_svg_as_surface(f"{_ASSETS_DIR}/radar_0.svg")
    original_radar_image: pygame.Surface = load_svg_as_surface(f"{_ASSETS_DIR}/radar_0.svg")
        
    # Seed da classe - inicialmente None, será gerada se não for definida explicitamente
    class_seed = None
    class_rng = None
    class_np_rng = None
    

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def __init__(
        self, 
        interest_point_center: pygame.math.Vector2, 
        position: Tuple[float, float], 
        behavior = None,
        fixed: bool = False, 
        broken: bool = False
    ) -> None:
        """
        Initialize the drone with its starting position, interest point, and behavior type.
        
        Args:
            interest_point_center: The center point of interest for the drone.
            position: Initial (x, y) position of the drone.
            behavior_type: The behavior strategy ("planning", "AI", "AEW", "RADAR", "debug", "u-debug").
            fixed: If True, the drone remains stationary.
            broken: If True, the drone will provide faulty detection information.
        """
        if FriendDrone.class_seed is None:
            FriendDrone.set_class_seed()
            
        # Basic properties
        self.pos: pygame.math.Vector2 = pygame.math.Vector2(position[0], position[1])
        self.interest_point_center = interest_point_center
        self.behavior = behavior
        self.fixed = fixed
        self.selected = False
        self.joystick_controlled = False
        self.vel: pygame.math.Vector2 = pygame.math.Vector2(0, 0)
        self.orbit_radius = None  # Used for AEW behavior
        self.trajectory: List[pygame.math.Vector2] = []
        self.return_to_base: bool = False
        self.info: Tuple[str, Any, Any, Any] = ("", None, None, None)
        self.detection_mode = None
        self.neighbors = []

        # Drone properties
        self.color: Tuple[int, int, int] = (255, 255, 255)
        self.drone_id: int = self.assign_id()
        self.in_election: bool = False
        self.is_leader: bool = False
        self.leader_id: int = self.drone_id
        self.broken: bool = broken
        
        # Broken drone state tracking
        self.timer_state_broken = 0
        self.update_state_broken = UPDATE_STATE_BROKEN
        self.broken_friend_intensity = None
        self.broken_friend_direction = None
        self.broken_enemy_intensity = None
        self.broken_enemy_direction = None

        # Detection dictionaries
        self.aux_enemy_detections: Dict[int, Tuple[int, int]] = {}
        self.aux_friend_detections: Dict[int, Tuple[int, int]] = {}
        self.current_enemy_pos_detection: Dict[int, pygame.math.Vector2] = {}
        self.current_friend_pos_detection: Dict[int, pygame.math.Vector2] = {}
                
        # State tracking
        self.state_history = {}
        self.current_state = ""
        self.total_steps = 0 
        self.messages_sent = 0
        self.distance_traveled = 0
        self.last_position = self.pos.copy()
        self.active_connections = 0
        self.messages_sent_this_cycle = 0
        
        
        # Triangulation matrices
        self.passive_detection_matrix = np.zeros((GRID_WIDTH * TRIANGULATION_GRANULARITY, GRID_HEIGHT * TRIANGULATION_GRANULARITY))
        self.merged_passive_detection_matrix = None
        self.direction_vectors = {}
        
        # Detection matrices
        self.enemy_intensity: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        self.enemy_direction: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT, 2))
        self.enemy_timestamp: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        self.friend_intensity: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        self.friend_direction: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT, 2))
        self.friend_timestamp: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        
        # Set appropriate image based on drone type
        self._setup_drone_image()
                    
    @classmethod
    def set_class_seed(cls, seed=None):            
        cls.class_seed = seed if seed is not None else random.randint(0, 10000000)
        cls.class_rng = random.Random(cls.class_seed)
        cls.class_np_rng = np.random.RandomState(cls.class_seed)
        
    def _setup_drone_image(self) -> None:
        """
        Set up the appropriate visual representation based on drone type.
        """
        if self.behavior.type == "RADAR":
            desired_width = int(SIM_WIDTH * 0.03)
            aspect_ratio = self.original_radar_image.get_height() / self.original_radar_image.get_width()
            desired_height = int(desired_width * aspect_ratio)
            self.image = pygame.transform.scale(self.original_radar_image, (desired_width, desired_height))
            
        elif self.broken:
            desired_width = int(SIM_WIDTH * 0.02)
            aspect_ratio = self.original_broken_drone_image.get_height() / self.original_broken_drone_image.get_width()
            desired_height = int(desired_width * aspect_ratio)
            self.image = pygame.transform.scale(self.original_broken_drone_image, (desired_width, desired_height))
            
        elif self.joystick_controlled:
            desired_width = int(SIM_WIDTH * 0.03)
            aspect_ratio = self.original_radar_image.get_height() / self.original_radar_image.get_width()
            desired_height = int(desired_width * aspect_ratio)
            self.image = pygame.transform.scale(self.original_radar_image, (desired_width, desired_height))
            
        else:
            desired_width = int(SIM_WIDTH * 0.02)
            aspect_ratio = self.original_drone_image.get_height() / self.original_drone_image.get_width()
            desired_height = int(desired_width * aspect_ratio)
            self.image = pygame.transform.scale(self.original_drone_image, (desired_width, desired_height))

    # -------------------------------------------------------------------------
    # Unique ID Assignment
    # -------------------------------------------------------------------------
    def assign_id(self) -> int:
        """
        Assign a unique ID to the drone.
        
        Returns:
            int: The unique drone ID.
        """
        current_id: int = self.__class__.friend_id_counter
        FriendDrone.friend_id_counter += 1
        return current_id
    
    # -------------------------------------------------------------------------
    # Detection Range
    # -------------------------------------------------------------------------
    def _get_detection_range(self) -> float:
        """
        Obtém o alcance de detecção apropriado com base no tipo de drone.
        
        Returns:
            float: Alcance de detecção em pixels.
        """
        if self.behavior.type == "AEW":
            return AEW_DETECTION_RANGE
        elif self.behavior.type == "RADAR":
            return RADAR_DETECTION_RANGE
        else:
            return FRIEND_DETECTION_RANGE

    # -------------------------------------------------------------------------
    # Matrix Decay
    # -------------------------------------------------------------------------
    def decay_matrices(self) -> None:
        """
        Apply exponential decay to both enemy and friend detection intensity matrices.
        This simulates information becoming less reliable over time.
        """
        self.enemy_intensity *= DECAY_FACTOR
        self.friend_intensity *= DECAY_FACTOR
        
        
    def _perform_direct_detection(self, enemy_drones: List[Any], detection_range: float, current_time: int) -> None:
        """
        Perform direct detection of enemy drones within detection range.
        
        Args:
            enemy_drones: List of enemy drones.
            detection_range: Maximum detection range in pixels.
            current_time: Current simulation time.
        """
        for enemy in enemy_drones:
            key: int = id(enemy)
            if self.pos.distance_to(enemy.pos) >= detection_range:
                self.current_enemy_pos_detection.pop(key, None)
                self.aux_enemy_detections.pop(key, None)
                continue
            
            cell: Tuple[int, int] = pos_to_cell(enemy.pos)
            if key not in self.current_enemy_pos_detection:
                self.current_enemy_pos_detection[key] = enemy.pos.copy()
            else:
                if key in self.current_enemy_pos_detection and key in self.aux_enemy_detections:
                    prev_cell: Tuple[int, int] = self.aux_enemy_detections[key]
                    if prev_cell != cell:
                        # Zero out values in the previous cell
                        self.enemy_intensity[prev_cell] = 0
                        self.enemy_direction[prev_cell] = [0, 0]
                        self.enemy_timestamp[prev_cell] = current_time
                self.aux_enemy_detections[key] = cell
                self.enemy_intensity[cell] = 1.0 # IndexError: index 54 is out of bounds for axis 1 with size 54
                delta: pygame.math.Vector2 = enemy.pos - self.current_enemy_pos_detection[key]
                self.current_enemy_pos_detection[key] = enemy.pos.copy()
                if delta.length() > 0:
                    self.enemy_direction[cell] = list(delta.normalize())
                self.enemy_timestamp[cell] = current_time
    
    # -------------------------------------------------------------------------
    # Detecção Passiva e Triangulação
    # -------------------------------------------------------------------------  
    def update_passive_detection(self, enemy_drones: List[Any]) -> None:
        self.direction_vectors = {}
        self.passive_detection_matrix = np.zeros((self.passive_detection_matrix.shape[0], self.passive_detection_matrix.shape[1]))
        
        detection_range = self._get_detection_range()
        for enemy in enemy_drones:
            delta = enemy.pos - self.pos
            distance = delta.length()
            
            # Verifica se o inimigo está dentro do alcance de detecção
            if distance <= detection_range and distance > 0:
                # Armazena apenas a direção normalizada (sem distância)
                direction = delta.normalize()
                
                # Cria um hash para esta direção (ângulo discretizado)
                angle = math.atan2(direction.y, direction.x)
                angle_discrete = round(angle / 0.01) * 0.01
                direction_hash = f"{angle_discrete:.2f}"
                self.direction_vectors[direction_hash] = direction
                
                # Linha de visada
                cell_size = CELL_SIZE / TRIANGULATION_GRANULARITY
                steps = int(math.floor(detection_range / cell_size))
                start_cell = pos_to_cell(self.pos, cell_size)
                pos = self.pos.copy()
                
                while True:
                    # Calcula posição ao longo da linha
                    pos = pos + direction * cell_size
                    
                    if self.pos.distance_to(pos) > detection_range:
                        break
                    
                    # Converte para célula
                    cell = pos_to_cell(pos, cell_size)
                    
                    # Verifica se está dentro dos limites da grade
                    if 0 <= cell[0] < GRID_WIDTH * TRIANGULATION_GRANULARITY and 0 <= cell[1] < GRID_HEIGHT * TRIANGULATION_GRANULARITY:
                        # Marca a célula
                        self.passive_detection_matrix[cell] = 1

    # -------------------------------------------------------------------------
    # Update Local Enemy Detection
    # -------------------------------------------------------------------------
    def update_local_enemy_detection(self, friend_drones: List[Any], enemy_drones: List[Any]) -> None:
    # def update_local_enemy_detection(self, enemy_drones: List[Any]) -> None:
        """
        Update local detection of enemy drones based on the selected detection mode.
        
        In "direct" mode, only direct detections are processed.
        In "triangulation" mode, only triangulated detections are processed.
        
        Args:
            enemy_drones: List of enemy drones.
        """
        current_time: int = pygame.time.get_ticks()
        detection_range = self._get_detection_range()
        
        # Direct detection (only in "direct" mode)
        if self.detection_mode == "direct":
            self._perform_direct_detection(enemy_drones, detection_range, current_time)
            
        else: # DEBUG
            for enemy in enemy_drones:
                # DEBUG
                i = 0
                
                for friend in friend_drones:
                    detection_range = friend._get_detection_range()
                    if friend.pos.distance_to(enemy.pos) <= detection_range:
                        i += 1
                        
                    if i >= N_LINE_SIGHT_CROSSING:
                        break
                
                if i < N_LINE_SIGHT_CROSSING:
                    continue
                # DEBUG
                
                key: int = id(enemy)
                if self.pos.distance_to(enemy.pos) >= detection_range:
                    self.current_enemy_pos_detection.pop(key, None)
                    self.aux_enemy_detections.pop(key, None)
                    continue
                
                cell: Tuple[int, int] = pos_to_cell(enemy.pos)
                if key not in self.current_enemy_pos_detection:
                    self.current_enemy_pos_detection[key] = enemy.pos.copy()
                else:
                    if key in self.current_enemy_pos_detection and key in self.aux_enemy_detections:
                        prev_cell: Tuple[int, int] = self.aux_enemy_detections[key]
                        if prev_cell != cell:
                            # Zero out values in the previous cell
                            self.enemy_intensity[prev_cell] = 0
                            self.enemy_direction[prev_cell] = [0, 0]
                            self.enemy_timestamp[prev_cell] = current_time
                    self.aux_enemy_detections[key] = cell
                    self.enemy_intensity[cell] = 1.0
                    delta: pygame.math.Vector2 = enemy.pos - self.current_enemy_pos_detection[key]
                    self.current_enemy_pos_detection[key] = enemy.pos.copy()
                    if delta.length() > 0:
                        self.enemy_direction[cell] = list(delta.normalize())
                    self.enemy_timestamp[cell] = current_time
        
        # In "triangulation" mode, detection is already done in update_passive_detection_and_triangulate()
        # and results are directly stored in detection matrices
        
        # --- Vectorized Update for Cells Without Detection ---
        # Clean up empty cells within detection radius (important in both modes)
        detection_range_cells = int(np.floor(detection_range / CELL_SIZE) * 0.8)
        
        # Get the central cell (drone's position)
        center_x, center_y = pos_to_cell(self.pos)
        
        # Define limits of rectangle enclosing detection circle
        x_min = max(center_x - detection_range_cells, 0)
        x_max = min(center_x + detection_range_cells, GRID_WIDTH - 1)
        y_min = max(center_y - detection_range_cells, 0)
        y_max = min(center_y + detection_range_cells, GRID_HEIGHT - 1)
        
        # Create grid of indices for the region
        x_indices = np.arange(x_min, x_max + 1)
        y_indices = np.arange(y_min, y_max + 1)
        xv, yv = np.meshgrid(x_indices, y_indices, indexing='ij')
        
        # Calculate distance of each cell from center
        distances = np.sqrt((xv - center_x) ** 2 + (yv - center_y) ** 2)
        
        # Extract region of matrices
        region_intensity = self.enemy_intensity[x_min:x_max+1, y_min:y_max+1]
        region_timestamp = self.enemy_timestamp[x_min:x_max+1, y_min:y_max+1]
        
        # Create mask for cells within detection circle with low intensity
        mask_empty = (distances <= detection_range_cells) & (region_intensity < 1)
        
        # Set intensities to 0 and update timestamps for empty cells
        np.putmask(region_intensity, mask_empty, 0)
        np.putmask(region_timestamp, mask_empty, current_time)
        
        # Apply broken detection behavior if drone is broken
        if self.broken:
            self.update_broken(x_min, x_max, y_min, y_max, distances, detection_range_cells)

    # -------------------------------------------------------------------------
    # Update Local Friend Detection
    # -------------------------------------------------------------------------
    def update_local_friend_detection(self, friend_drones: List[Any]) -> None:
        """
        Update local detection of friendly drones.
        
        For each friendly drone (excluding AEW drones and self), update the 
        corresponding cell in the detection matrices.
        
        Args:
            friend_drones: List of friendly drones.
        """
        current_time: int = pygame.time.get_ticks()
        for friend in friend_drones:
            # Skip AEW drones and self
            if friend.behavior.type == "AEW" or friend is self:
                continue
            
            key: int = id(friend)
            if self.pos.distance_to(friend.pos) >= COMMUNICATION_RANGE:
                self.current_friend_pos_detection.pop(key, None)
                self.aux_friend_detections.pop(key, None)
                continue
            
            cell: Tuple[int, int] = pos_to_cell(friend.pos)
            if key not in self.current_friend_pos_detection:
                self.current_friend_pos_detection[key] = friend.pos.copy()
            else:
                if key in self.current_friend_pos_detection and key in self.aux_friend_detections:
                    prev_cell: Tuple[int, int] = self.aux_friend_detections[key]
                    if prev_cell != cell:
                        self.friend_intensity[prev_cell] = 0
                        self.friend_direction[prev_cell] = [0, 0]
                        self.friend_timestamp[prev_cell] = current_time
                self.aux_friend_detections[key] = cell
                self.friend_intensity[cell] = 1.0
                delta: pygame.math.Vector2 = friend.pos - self.current_friend_pos_detection[key]
                self.current_friend_pos_detection[key] = friend.pos.copy()
                if delta.length() > 0:
                    self.friend_direction[cell] = delta.normalize()
                self.friend_timestamp[cell] = current_time
                
        # --- Vectorized Update for Cells Without Friend Detection ---
        # Convert friend detection range to cell units
        detection_range_cells = int(np.floor(FRIEND_DETECTION_RANGE / CELL_SIZE) * 0.8)
        
        # Get the central cell (drone's position)
        center_x, center_y = pos_to_cell(self.pos)
        
        # Define rectangle covering detection circle
        x_min = max(center_x - detection_range_cells, 0)
        x_max = min(center_x + detection_range_cells, GRID_WIDTH - 1)
        y_min = max(center_y - detection_range_cells, 0)
        y_max = min(center_y + detection_range_cells, GRID_HEIGHT - 1)
        
        # Create index grid for region
        x_indices = np.arange(x_min, x_max + 1)
        y_indices = np.arange(y_min, y_max + 1)
        xv, yv = np.meshgrid(x_indices, y_indices, indexing='ij')
        
        # Calculate cell distances from center
        distances = np.sqrt((xv - center_x)**2 + (yv - center_y)**2)
        
        # Extract sub-regions of friend detection matrices
        region_intensity = self.friend_intensity[x_min:x_max+1, y_min:y_max+1]
        region_timestamp = self.friend_timestamp[x_min:x_max+1, y_min:y_max+1]
        
        # Create mask for empty cells within detection circle
        mask_empty = (distances <= detection_range_cells) & (region_intensity < 1)
        
        # Reset intensities and timestamps for empty cells
        np.putmask(region_intensity, mask_empty, 0)
        np.putmask(region_timestamp, mask_empty, current_time)
        
        # Apply broken detection behavior if drone is broken
        if self.broken:
            self.update_broken(x_min, x_max, y_min, y_max, distances, detection_range_cells)
            
    # -------------------------------------------------------------------------
    # Merge Enemy Matrix
    # -------------------------------------------------------------------------
    def merge_enemy_matrix(self, neighbor) -> None:
        """
        Merge enemy detection data from a neighbor drone into this drone's matrices,
        propagating information globally based on timestamps.
        
        Args:
            neighbor: The neighbor drone to merge data from.
        """
        update_mask = neighbor.enemy_timestamp > self.enemy_timestamp
        np.putmask(self.enemy_intensity, update_mask, neighbor.enemy_intensity)
        np.putmask(
            self.enemy_direction,
            np.broadcast_to(update_mask[..., None], self.enemy_direction.shape),
            neighbor.enemy_direction
        )
        np.putmask(self.enemy_timestamp, update_mask, neighbor.enemy_timestamp)

    # -------------------------------------------------------------------------
    # Merge Friend Matrix
    # -------------------------------------------------------------------------
    def merge_friend_matrix(self, neighbor: "FriendDrone") -> None:
        """
        Merge friend detection data from a neighbor drone into this drone's matrices.
        
        Args:
            neighbor: The neighbor drone to merge data from.
        """
        update_mask = neighbor.friend_timestamp > self.friend_timestamp
        np.putmask(self.friend_intensity, update_mask, neighbor.friend_intensity)
        np.putmask(
            self.friend_direction,
            np.broadcast_to(update_mask[..., None], self.friend_direction.shape),
            neighbor.friend_direction
        )
        np.putmask(self.friend_timestamp, update_mask, neighbor.friend_timestamp)
        
    # -------------------------------------------------------------------------
    # Update Neghbors
    # -------------------------------------------------------------------------
    def update_neghbors(self, all_drones: List[Any]):
        candidates = [
            other for other in all_drones
            if other is not self
            and self.pos.distance_to(other.pos) < COMMUNICATION_RANGE
        ]
        # 2) ordena por distância e pega os 3 mais próximos
        candidates.sort(key=lambda o: self.pos.distance_to(o.pos))
        nearest = candidates[:N_CONNECTIONS]

        # 3) identifica quem te considera entre os 3 mais próximos deles
        reverse_neighbors = []
        for other in all_drones:
            if other is self:
                continue
            # candidatos de 'other' dentro do alcance
            others_cand = [
                o for o in all_drones
                if o is not other
                and other.pos.distance_to(o.pos) < COMMUNICATION_RANGE
            ]
            others_cand.sort(key=lambda o: other.pos.distance_to(o.pos))
            if self in others_cand[:N_CONNECTIONS]:
                reverse_neighbors.append(other)

        # 4) união dos dois conjuntos
        neighbors = set(nearest + reverse_neighbors)
        self.neighbors = neighbors

    # -------------------------------------------------------------------------
    # Communication
    # -------------------------------------------------------------------------
    def communication(self, all_drones: List[Any]) -> None:
        connections = 0
        messages = 0
        
        for _ in range(CICLE_COMM_BY_STEP):
            for other in self.neighbors:
                connections += 1
                messages += 2
                if FriendDrone.class_rng.random() > MESSAGE_LOSS_PROBABILITY:
                    self.merge_enemy_matrix(other)
                    self.merge_friend_matrix(other)

        self.active_connections += connections
        self.messages_sent_this_cycle += messages
        
    
    def broadcast_passive_detection(self) -> None:
        connections = 0
        messages = 0
        
        for _ in range(CICLE_COMM_BY_STEP):
            for other in self.neighbors:                
                connections += 1
                messages += 1
                
                self.merged_passive_detection_matrix = self.passive_detection_matrix.copy()
                if FriendDrone.class_rng.random() > MESSAGE_LOSS_PROBABILITY and other != self:
                    self.merged_passive_detection_matrix += other.passive_detection_matrix
                    
        self.active_connections += connections
        self.messages_sent_this_cycle += messages
    # -------------------------------------------------------------------------
    # Apply Behavior Broken
    # -------------------------------------------------------------------------   
    def update_broken(self, x_min: int, x_max: int, y_min: int, y_max: int,
                     distances: np.ndarray, detection_range_cells: int) -> None:
        """
        Update detection matrices with random values for broken drones.
        
        For broken drones, this function updates detection matrices (intensity and direction)
        only for cells within detection range, using random values. It maintains these values
        for a set period before generating new random values.
        
        Args:
            x_min, x_max, y_min, y_max: Region limits (in cells) covering detection radius.
            distances: Matrix with distances (in cell units) from each cell to center.
            detection_range_cells: Detection radius in cell units.
        """
        # Determine shape of region to update
        region_shape = (x_max - x_min + 1, y_max - y_min + 1)
        
        # Generate broken states if not already generated
        if self.broken_enemy_direction is None:
            self.broken_enemy_intensity, self.broken_enemy_direction = generate_sparse_matrix(region_shape, max_nonzero=10, seed=self.seed)
            self.broken_friend_intensity, self.broken_friend_direction = generate_sparse_matrix(region_shape, max_nonzero=10, seed=self.seed)
        
        # Extract submatrices for region of interest
        region_enemy_intensity = self.enemy_intensity[x_min:x_max+1, y_min:y_max+1]
        region_enemy_direction = self.enemy_direction[x_min:x_max+1, y_min:y_max+1]
        region_friend_intensity = self.friend_intensity[x_min:x_max+1, y_min:y_max+1]
        region_friend_direction = self.friend_direction[x_min:x_max+1, y_min:y_max+1]
        
        # Create mask for cells within detection radius
        mask = distances <= detection_range_cells

        if self.timer_state_broken < self.update_state_broken:
            # Update cells with random values using mask
            np.putmask(region_enemy_intensity, mask, self.broken_enemy_intensity[mask])
            np.putmask(region_enemy_direction,
                      np.broadcast_to(mask[..., None], region_enemy_direction.shape),
                      self.broken_enemy_direction)
            
            np.putmask(region_friend_intensity, mask, self.broken_friend_intensity[mask])
            np.putmask(region_friend_direction,
                      np.broadcast_to(mask[..., None], region_friend_direction.shape),
                      self.broken_friend_direction)
            
            self.timer_state_broken += 1
            return
        else:
            # Generate new random states when timer expires
            self.broken_enemy_intensity, self.broken_enemy_direction = generate_sparse_matrix(region_shape, max_nonzero=10, seed=self.seed)
            self.broken_friend_intensity, self.broken_friend_direction = generate_sparse_matrix(region_shape, max_nonzero=10, seed=self.seed)
            
            # Update timestamps for region
            current_time: int = pygame.time.get_ticks()
            self.enemy_timestamp[x_min:x_max+1, y_min:y_max+1].fill(current_time)
            self.friend_timestamp[x_min:x_max+1, y_min:y_max+1].fill(current_time)
            
            self.timer_state_broken = 0
            
    # -------------------------------------------------------------------------
    # Action Execution
    # -------------------------------------------------------------------------
    def take_action(self) -> None:
        """
        Execute the drone's action based on enemy detection.
        
        Apply appropriate behavior, update position, and ensure drone stays within
        simulation bounds and within the maximum allowed distance from interest point.
        """
        if self.return_to_base:
            # Head back to interest point center
            self.vel = (self.interest_point_center - self.pos).normalize() * FRIEND_SPEED
            self.info = ("REGRESS", None, None, None)
        else:
            # Apply behavior-specific movement
            self.apply_behavior()
            
        # Update position
        self.pos += self.vel

        # Keep drone within simulation bounds
        self._enforce_simulation_bounds()
            
        # Prevent drone from exceeding EXTERNAL_RADIUS from interest point
        self._enforce_external_radius()

    def _enforce_simulation_bounds(self) -> None:
        """
        Ensure the drone stays within the simulation boundaries.
        If a boundary is hit, reverse the corresponding velocity component.
        """
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
        """
        Ensure the drone stays within the maximum allowed distance from interest point.
        If the limit is exceeded, move the drone back to the boundary and stop it.
        """
        if self.pos.distance_to(self.interest_point_center) > EXTERNAL_RADIUS:
            direction = (self.pos - self.interest_point_center).normalize()
            self.pos = self.interest_point_center + direction * EXTERNAL_RADIUS
            self.vel = pygame.math.Vector2(0, 0)

    # -------------------------------------------------------------------------
    # Update Drone State
    # -------------------------------------------------------------------------
    def update(self, enemy_drones: List[Any], friend_drones: List[Any], use_triangulation: bool, return_to_base: bool = False) -> None:
        """
        Update the drone's state for the current simulation step.
        
        This includes decaying matrices, updating local detections, communicating with
        nearby drones, and executing actions.
        
        Args:
            enemy_drones: List of enemy drones.
            friend_drones: List of friendly drones.
            return_to_base: If True, drone will return to the interest point center.
        """
        self.total_steps += 1
        self.active_connections = 0
        self.messages_sent_this_cycle = 0
        self.return_to_base = return_to_base
        
        self.detection_mode = "triangulation" if use_triangulation else "direct"
        
        # Apply exponential decay to detection matrices
        self.decay_matrices()
        
        # Atualiza vizinhos próximos
        self.update_neghbors(friend_drones)
        
        if self.detection_mode == "triangulation":
            self.update_passive_detection(enemy_drones)
        #     self.broadcast_passive_detection()
            
        # A função update_local_enemy_detection verifica internamente o modo de detecção
        self.update_local_enemy_detection(friend_drones, enemy_drones)
        # self.update_local_enemy_detection(enemy_drones)
        self.update_local_friend_detection(friend_drones)
        
        # Communicate with nearby drones
        self.communication(friend_drones)
        
        # Execute movement action
        self.take_action()
        
        # Calculate distance traveled
        self.distance_traveled += self.pos.distance_to(self.last_position)
        self.last_position = self.pos.copy()
        
        # Record trajectory for visualization
        if len(self.trajectory) > 300:
            self.trajectory.pop(0)
        self.trajectory.append(self.pos.copy())
        
        # Update state history
        if self.current_state:
            if self.current_state in self.state_history:
                self.state_history[self.current_state] += 1
            else:
                self.state_history[self.current_state] = 1
                
    def get_state_percentages(self) -> dict:
        """
        Calculate the percentage of steps spent in each state.
        
        Returns:
            dict: Dictionary mapping state names to percentage of time spent.
        """
        if self.total_steps == 0:
            return {}
            
        # Calculate percentages
        percentages = {}
        for state, count in self.state_history.items():
            percentages[state] = (count / self.total_steps) * 100
                
        return percentages
        
    # -------------------------------------------------------------------------
    # Apply Behavior
    # -------------------------------------------------------------------------
    def apply_behavior(self) -> None:
        """
        Update drone velocity based on its behavior type.
        
        Different behaviors include:
        - planning: Use the planning policy for decision-making
        - AI: Use the AI model for decision-making
        - AEW: Orbit around the interest point to provide surveillance
        - RADAR: Stationary radar
        - debug: Move directly toward interest point
        - u-debug: Move in a U-shaped pattern for testing
        """
        if self.fixed:
            self.vel = pygame.math.Vector2(0, 0)
            self.info = ("FIXED", None, None, None)
            self.current_state = self.info[0] if isinstance(self.info, tuple) else self.info
        
        else:
            state = {
                'pos': np.array([[self.pos.x, self.pos.y]], dtype=np.float32),
                'friend_intensity': np.expand_dims(self.friend_intensity, axis=0),
                'enemy_intensity': np.expand_dims(self.enemy_intensity, axis=0),
                'friend_direction': np.expand_dims(self.friend_direction, axis=0),
                'enemy_direction': np.expand_dims(self.enemy_direction, axis=0)
            }
            
            self.info, self.vel = self.behavior.apply(state, self.joystick_controlled)
            
        self.current_state = self.info[0] if isinstance(self.info, tuple) else self.info
        
    # -------------------------------------------------------------------------
    # Rendering: Draw the Drone
    # -------------------------------------------------------------------------
    def draw(self, surface: pygame.Surface, show_detection: bool = True, 
             show_comm_range: bool = True, show_trajectory: bool = False, 
             show_debug: bool = False) -> None:
        """
        Draw the drone and its information on the provided surface.
        
        Args:
            surface: Surface to draw on.
            show_detection: If True, display detection range.
            show_comm_range: If True, display communication range.
            show_trajectory: If True, draw drone's trajectory.
            show_debug: If True, show additional debug information.
        """
        # Draw trajectory if enabled
        if show_trajectory and len(self.trajectory) > 1:
            self._draw_trajectory(surface)
        
        # Draw detection range
        if show_detection or self.broken:
            detection_range = self._get_detection_range()
            draw_dashed_circle(
                surface, 
                (self.color[0], self.color[1], self.color[2], 32), 
                (int(self.pos.x), int(self.pos.y)), 
                detection_range, 
                5, 5, 1
            )
            
        # Draw communication range
        if show_comm_range:
            draw_dashed_circle(
                surface, 
                (255, 255, 0, 32), 
                (int(self.pos.x), int(self.pos.y)), 
                COMMUNICATION_RANGE, 
                5, 5, 1
            )
        
        # Draw drone image
        image_rect = self.image.get_rect(center=(int(self.pos.x), int(self.pos.y)))
        surface.blit(self.image, image_rect)
        
        # Draw drone ID
        self._draw_drone_id(surface)
        
        # Draw debug information if enabled
        if show_debug:
            self._draw_debug_info(surface)
            
    def _draw_trajectory(self, surface: pygame.Surface) -> None:
        """
        Draw the drone's movement trajectory with fading effect.
        
        Args:
            surface: Surface to draw on.
        """
        traj_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        decay_rate = 0.04
        n = len(self.trajectory)
        
        for i in range(n - 1):
            d = n - 1 - i
            alpha = int(255 * math.exp(-decay_rate * d))
            min_alpha = 30
            alpha = max(alpha, min_alpha)
            
            color_with_alpha = self.color + (alpha,)
            start_pos = (int(self.trajectory[i].x), int(self.trajectory[i].y))
            end_pos = (int(self.trajectory[i+1].x), int(self.trajectory[i+1].y))
            
            pygame.draw.line(traj_surf, color_with_alpha, start_pos, end_pos, 2)
            
        surface.blit(traj_surf, (0, 0))
            
    def _draw_drone_id(self, surface: pygame.Surface) -> None:
        """
        Draw the drone's ID and status labels.
        
        Args:
            surface: Surface to draw on.
        """
        font: pygame.font.Font = pygame.font.SysFont(FONT_FAMILY, 10)
        
        # Draw drone ID
        label = font.render(f"ID: F{self.drone_id}", True, (255, 255, 255))
        label.set_alpha(128)
        surface.blit(label, (int(self.pos.x) + 20, int(self.pos.y) - 20))
        
        # Draw leader indicator if applicable
        if self.is_leader:
            leader_label = font.render("LEADER", True, (0, 255, 0))
            surface.blit(leader_label, (int(self.pos.x) + 20, int(self.pos.y) - 5))
        
        # Draw selection indicator if applicable
        if self.selected:
            selected_label = font.render("GRAPH", True, (0, 255, 0))
            surface.blit(selected_label, (int(self.pos.x) + 20, int(self.pos.y) + 10))
            
    def _draw_debug_info(self, surface: pygame.Surface) -> None:
        """
        Draw additional debug information.
        
        Args:
            surface: Surface to draw on.
        """
        # Draw passive detection information if in triangulation mode
        if self.detection_mode == "triangulation":
            for direction in self.direction_vectors.values(): 
                # Calculate end point (extending the direction)
                end_point = self.pos + direction * FRIEND_DETECTION_RANGE
                draw_dashed_line(
                    surface, 
                    (255, 0, 0, 128), 
                    self.pos, 
                    end_point,
                    width=1, 
                    dash_length=5, 
                    space_length=5
                )
                
        # Draw state information
        font = pygame.font.SysFont(FONT_FAMILY, 10)
        if self.info and self.info[0]:
            len_info = len(self.info[0])
            debug_label = font.render(self.info[0], True, (255, 215, 0))
            surface.blit(debug_label, (int(self.pos.x) - 3.5 * len_info, int(self.pos.y) + 25))
            
            # Draw target information if available
            if self.info[1] is not None:
                pygame.draw.circle(
                    surface, 
                    (255, 215, 0), 
                    (int(self.info[1].x), int(self.info[1].y)), 
                    4
                )
                pygame.draw.line(
                    surface,
                    (255, 215, 0),
                    (int(self.pos.x), int(self.pos.y)),
                    (int(self.info[1].x), int(self.info[1].y)),
                    2
                )
                
            # Draw interest point line if available
            if self.info[2] is not None:
                pygame.draw.line(
                    surface,
                    (255, 215, 0),
                    (int(self.interest_point_center[0]), int(self.interest_point_center[1])),
                    (int(self.info[2].x), int(self.info[2].y)),
                    2
                )