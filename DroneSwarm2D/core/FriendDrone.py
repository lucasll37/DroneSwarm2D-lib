 
"""
FriendDrone.py

Define a classe FriendDrone usada na simulação de defesa com enxames de drones.

A classe FriendDrone gerencia:
- Detecção local (inimigos e amigos)
- Comunicação entre drones
- Fusão de matrizes de detecção
- Triangulação de alvos
- Execução de ações
- Renderização e debug

Attributes principais:
    pos: Posição atual do drone
    vel: Velocidade atual
    behavior: Comportamento do drone (planning, AI, AEW, RADAR, etc.)
    enemy_intensity: Matriz de intensidade de detecção de inimigos
    friend_intensity: Matriz de intensidade de detecção de amigos
    neighbors: Lista de drones vizinhos para comunicação
"""

import random
import math
import numpy as np
import pygame
import itertools

from typing import Optional, Tuple, List, Any, Dict, Set
from pathlib import Path

from .settings import (
    SIM_WIDTH,
    SIM_HEIGHT,
    GRID_WIDTH,
    GRID_HEIGHT,
    CELL_SIZE,
    FRIEND_SPEED,
    COMMUNICATION_RANGE,
    FRIEND_DETECTION_RANGE,
    AEW_DETECTION_RANGE,
    RADAR_DETECTION_RANGE,
    DECAY_FACTOR,
    UPDATE_STATE_BROKEN,
    MESSAGE_LOSS_PROBABILITY,
    CICLE_COMM_BY_STEP,
    N_CONNECTIONS,
    EPSILON,
    CENTER,
    EXTERNAL_RADIUS,
    TRIANGULATION_GRANULARITY,
    N_LINE_SIGHT_CROSSING,
    FONT_FAMILY,
)
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


class FriendDrone:
    """
    Representa um drone amigo defensivo na simulação.
    
    Esta classe gerencia detecção de inimigos e amigos, comunicação
    entre drones, fusão de informações, movimentação e renderização.
    
    Attributes:
        pos: Posição atual como Vector2
        vel: Velocidade atual como Vector2
        drone_id: Identificador único do drone
        behavior: Objeto de comportamento do drone
        fixed: Se True, drone permanece estacionário
        broken: Se True, drone fornece informações falsas
        selected: Se True, drone está selecionado na UI
        is_leader: Se True, drone é líder do enxame
        joystick_controlled: Se True, drone é controlado por joystick
        neighbors: Lista de drones vizinhos para comunicação
        enemy_intensity: Matriz de intensidade de detecção de inimigos
        enemy_direction: Matriz de direção de inimigos detectados
        friend_intensity: Matriz de intensidade de detecção de amigos
        friend_direction: Matriz de direção de amigos detectados
        
    Class Attributes:
        friend_id_counter: Contador global para IDs únicos
        class_seed: Seed compartilhada pela classe
        class_rng: Random Number Generator da classe
        class_np_rng: Numpy RNG da classe
    """

    # Variáveis de classe
    friend_id_counter: int = 0
    original_drone_image: pygame.Surface = load_svg_as_surface(f"{_ASSETS_DIR}/drone_0.svg")
    original_broken_drone_image: pygame.Surface = load_svg_as_surface(f"{_ASSETS_DIR}/drone_broken.svg")
    original_aew_image: pygame.Surface = load_svg_as_surface(f"{_ASSETS_DIR}/radar_0.svg")
    original_radar_image: pygame.Surface = load_svg_as_surface(f"{_ASSETS_DIR}/radar_0.svg")
    
    # Seeds da classe
    class_seed: Optional[int] = None
    class_rng: Optional[random.Random] = None
    class_np_rng: Optional[np.random.RandomState] = None

    def __init__(
        self,
        interest_point_center: pygame.math.Vector2,
        position: Tuple[float, float],
        behavior: Optional[Any] = None,
        fixed: bool = False,
        broken: bool = False
    ) -> None:
        """
        Inicializa o drone com posição, ponto de interesse e tipo de comportamento.
        
        Args:
            interest_point_center: Centro do ponto de interesse a defender
            position: Posição inicial (x, y) do drone
            behavior: Objeto de comportamento (planning, AI, AEW, RADAR, debug, etc.)
            fixed: Se True, o drone permanece estacionário
            broken: Se True, o drone fornecerá informações de detecção falsas
        """
        if FriendDrone.class_seed is None:
            FriendDrone.set_class_seed()
        
        # Propriedades básicas
        self.pos: pygame.math.Vector2 = pygame.math.Vector2(position[0], position[1])
        self.interest_point_center: pygame.math.Vector2 = interest_point_center
        self.behavior: Any = behavior
        self.fixed: bool = fixed
        self.selected: bool = False
        self.joystick_controlled: bool = False
        self.vel: pygame.math.Vector2 = pygame.math.Vector2(0, 0)
        self.orbit_radius: Optional[float] = None  # Usado para comportamento AEW
        self.trajectory: List[pygame.math.Vector2] = []
        self.return_to_base: bool = False
        self.info: Tuple[str, Any, Any, Any] = ("", None, None, None)
        self.detection_mode: Optional[str] = None
        self.neighbors: List['FriendDrone'] = []

        # Propriedades do drone
        self.color: Tuple[int, int, int] = (255, 255, 255)
        self.drone_id: int = self._assign_id()
        self.in_election: bool = False
        self.is_leader: bool = False
        self.leader_id: int = self.drone_id
        self.broken: bool = broken
        
        # Rastreamento de estado de drones quebrados
        self.timer_state_broken: int = 0
        self.update_state_broken: int = UPDATE_STATE_BROKEN
        self.broken_friend_intensity: Optional[np.ndarray] = None
        self.broken_friend_direction: Optional[np.ndarray] = None
        self.broken_enemy_intensity: Optional[np.ndarray] = None
        self.broken_enemy_direction: Optional[np.ndarray] = None

        # Dicionários de detecção
        self.aux_enemy_detections: Dict[int, Tuple[int, int]] = {}
        self.aux_friend_detections: Dict[int, Tuple[int, int]] = {}
        self.current_enemy_pos_detection: Dict[int, pygame.math.Vector2] = {}
        self.current_friend_pos_detection: Dict[int, pygame.math.Vector2] = {}
        
        # Rastreamento de estado
        self.state_history: Dict[str, int] = {}
        self.current_state: str = ""
        self.total_steps: int = 0
        self.messages_sent: int = 0
        self.distance_traveled: float = 0.0
        self.last_position: pygame.math.Vector2 = self.pos.copy()
        self.active_connections: int = 0
        self.messages_sent_this_cycle: int = 0
        
        # Matrizes de triangulação
        self.passive_detection_matrix: np.ndarray = np.zeros(
            (GRID_WIDTH * TRIANGULATION_GRANULARITY, GRID_HEIGHT * TRIANGULATION_GRANULARITY)
        )
        self.merged_passive_detection_matrix: Optional[np.ndarray] = None
        self.direction_vectors: Dict[str, pygame.math.Vector2] = {}
        
        # Matrizes de detecção
        self.enemy_intensity: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        self.enemy_direction: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT, 2))
        self.enemy_timestamp: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        self.friend_intensity: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        self.friend_direction: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT, 2))
        self.friend_timestamp: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        
        # Configurar imagem apropriada baseada no tipo de drone
        self._setup_drone_image()
    
    @classmethod
    def set_class_seed(cls, seed: Optional[int] = None) -> None:
        """
        Define a seed compartilhada da classe para geração de números aleatórios.
        
        Args:
            seed: Seed a ser usada. Se None, gera uma seed aleatória.
            
        Note:
            Inicializa random.Random e np.random.RandomState
        """
        cls.class_seed = seed if seed is not None else random.randint(0, 10000000)
        cls.class_rng = random.Random(cls.class_seed)
        cls.class_np_rng = np.random.RandomState(cls.class_seed)
    
    def _setup_drone_image(self) -> None:
        """
        Configura a representação visual apropriada baseada no tipo de drone.
        
        Note:
            Tipos: RADAR (maior), AEW, broken (danificado), joystick, padrão
        """
        if self.behavior.type == "RADAR":
            desired_width: int = int(SIM_WIDTH * 0.03)
            aspect_ratio: float = self.original_radar_image.get_height() / self.original_radar_image.get_width()
            desired_height: int = int(desired_width * aspect_ratio)
            self.image: pygame.Surface = pygame.transform.scale(
                self.original_radar_image,
                (desired_width, desired_height)
            )
        elif self.broken:
            desired_width = int(SIM_WIDTH * 0.02)
            aspect_ratio = self.original_broken_drone_image.get_height() / self.original_broken_drone_image.get_width()
            desired_height = int(desired_width * aspect_ratio)
            self.image = pygame.transform.scale(
                self.original_broken_drone_image,
                (desired_width, desired_height)
            )
        elif self.joystick_controlled:
            desired_width = int(SIM_WIDTH * 0.03)
            aspect_ratio = self.original_radar_image.get_height() / self.original_radar_image.get_width()
            desired_height = int(desired_width * aspect_ratio)
            self.image = pygame.transform.scale(
                self.original_radar_image,
                (desired_width, desired_height)
            )
        else:
            desired_width = int(SIM_WIDTH * 0.02)
            aspect_ratio = self.original_drone_image.get_height() / self.original_drone_image.get_width()
            desired_height = int(desired_width * aspect_ratio)
            self.image = pygame.transform.scale(
                self.original_drone_image,
                (desired_width, desired_height)
            )

    def _assign_id(self) -> int:
        """
        Atribui um ID único ao drone.
        
        Returns:
            ID único do drone
        """
        current_id: int = self.__class__.friend_id_counter
        FriendDrone.friend_id_counter += 1
        return current_id
    
    def _get_detection_range(self) -> float:
        """
        Obtém o alcance de detecção apropriado baseado no tipo de drone.
        
        Returns:
            Alcance de detecção em pixels
            
        Note:
            - AEW: maior alcance (AEW_DETECTION_RANGE)
            - RADAR: alcance de radar (RADAR_DETECTION_RANGE)
            - Outros: alcance padrão (FRIEND_DETECTION_RANGE)
        """
        if self.behavior.type == "AEW":
            return AEW_DETECTION_RANGE
        elif self.behavior.type == "RADAR":
            return RADAR_DETECTION_RANGE
        else:
            return FRIEND_DETECTION_RANGE

    def decay_matrices(self) -> None:
        """
        Aplica decaimento exponencial às matrizes de intensidade de detecção.
        
        Simula informação se tornando menos confiável ao longo do tempo.
        As matrizes de inimigos e amigos decaem pelo fator DECAY_FACTOR.
        
        Note:
            DECAY_FACTOR geralmente está entre 0.9-0.99
        """
        self.enemy_intensity *= DECAY_FACTOR
        self.friend_intensity *= DECAY_FACTOR

    def _perform_direct_detection(
        self,
        enemy_drones: List[Any],
        detection_range: float,
        current_time: int
    ) -> None:
        """
        Realiza detecção direta de drones inimigos dentro do alcance.
        
        Args:
            enemy_drones: Lista de drones inimigos
            detection_range: Alcance máximo de detecção em pixels
            current_time: Tempo atual da simulação
            
        Note:
            Atualiza enemy_intensity, enemy_direction e enemy_timestamp
        """
        for enemy in enemy_drones:
            key: int = id(enemy)
            distance: float = self.pos.distance_to(enemy.pos)
            
            if distance >= detection_range:
                self.current_enemy_pos_detection.pop(key, None)
                self.aux_enemy_detections.pop(key, None)
                continue
            
            cell: Tuple[int, int] = pos_to_cell(enemy.pos)
            
            if key not in self.current_enemy_pos_detection:
                self.current_enemy_pos_detection[key] = enemy.pos.copy()
            else:
                if key in self.aux_enemy_detections:
                    prev_cell: Tuple[int, int] = self.aux_enemy_detections[key]
                    if prev_cell != cell:
                        # Zerar valores na célula anterior
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

    def update_passive_detection(self, enemy_drones: List[Any]) -> None:
        """
        Atualiza detecção passiva criando linhas de visada para cada inimigo detectado.
        
        A detecção passiva permite triangulação quando múltiplos drones detectam
        o mesmo alvo de diferentes ângulos.
        
        Args:
            enemy_drones: Lista de drones inimigos
            
        Note:
            Preenche passive_detection_matrix e direction_vectors
        """
        self.direction_vectors = {}
        self.passive_detection_matrix = np.zeros(self.passive_detection_matrix.shape)
        
        detection_range: float = self._get_detection_range()
        
        for enemy in enemy_drones:
            delta = enemy.pos - self.pos
            distance: float = delta.length()
            
            # Verificar se inimigo está dentro do alcance
            if distance <= detection_range and distance > 0:
                # Armazenar apenas direção normalizada (sem distância)
                direction = delta.normalize()
                
                # Hash para esta direção (ângulo discretizado)
                angle: float = math.atan2(direction.y, direction.x)
                angle_discrete: float = round(angle / 0.01) * 0.01
                direction_hash: str = f"{angle_discrete:.2f}"
                self.direction_vectors[direction_hash] = direction
                
                # Linha de visada
                cell_size: float = CELL_SIZE / TRIANGULATION_GRANULARITY
                pos = self.pos.copy()
                
                while True:
                    # Calcular posição ao longo da linha
                    pos = pos + direction * cell_size
                    
                    if self.pos.distance_to(pos) > detection_range:
                        break
                    
                    # Converter para célula
                    cell = pos_to_cell(pos, cell_size)
                    
                    # Verificar limites da grade
                    if (0 <= cell[0] < GRID_WIDTH * TRIANGULATION_GRANULARITY and
                        0 <= cell[1] < GRID_HEIGHT * TRIANGULATION_GRANULARITY):
                        # Marcar a célula
                        self.passive_detection_matrix[cell] = 1

    def update_local_enemy_detection(
        self,
        friend_drones: List['FriendDrone'],
        enemy_drones: List[Any]
    ) -> None:
        """
        Atualiza detecção local de drones inimigos baseado no modo de detecção selecionado.
        
        Em modo "direct", apenas detecções diretas são processadas.
        Em modo "triangulation", apenas detecções trianguladas são processadas.
        
        Args:
            friend_drones: Lista de drones amigos
            enemy_drones: Lista de drones inimigos
            
        Note:
            O modo de detecção é definido em self.detection_mode
        """
        current_time: int = pygame.time.get_ticks()
        detection_range: float = self._get_detection_range()
        
        # Detecção direta (apenas em modo "direct")
        if self.detection_mode == "direct":
            self._perform_direct_detection(enemy_drones, detection_range, current_time)
        else:
            # DEBUG: Verificar se inimigo foi detectado por N_LINE_SIGHT_CROSSING drones
            for enemy in enemy_drones:
                detection_count: int = 0
                
                for friend in friend_drones:
                    friend_range: float = friend._get_detection_range()
                    if friend.pos.distance_to(enemy.pos) <= friend_range:
                        detection_count += 1
                    if detection_count >= N_LINE_SIGHT_CROSSING:
                        break
                
                if detection_count < N_LINE_SIGHT_CROSSING:
                    continue
                
                key: int = id(enemy)
                if self.pos.distance_to(enemy.pos) >= detection_range:
                    self.current_enemy_pos_detection.pop(key, None)
                    self.aux_enemy_detections.pop(key, None)
                    continue
                
                cell: Tuple[int, int] = pos_to_cell(enemy.pos)
                
                if key not in self.current_enemy_pos_detection:
                    self.current_enemy_pos_detection[key] = enemy.pos.copy()
                else:
                    if key in self.aux_enemy_detections:
                        prev_cell: Tuple[int, int] = self.aux_enemy_detections[key]
                        if prev_cell != cell:
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
        
        # Atualização vetorizada para células sem detecção
        self._clean_empty_cells(detection_range, current_time, is_enemy=True)
        
        # Aplicar comportamento de drone quebrado se aplicável
        if self.broken:
            self._apply_broken_detection(detection_range, is_enemy=True)

    def _clean_empty_cells(
        self,
        detection_range: float,
        current_time: int,
        is_enemy: bool = True
    ) -> None:
        """
        Limpa células vazias dentro do raio de detecção.
        
        Args:
            detection_range: Raio de detecção em pixels
            current_time: Tempo atual da simulação
            is_enemy: Se True, limpa enemy_intensity; se False, limpa friend_intensity
        """
        detection_range_cells: int = int(np.floor(detection_range / CELL_SIZE) * 0.8)
        center_x, center_y = pos_to_cell(self.pos)
        
        # Definir limites do retângulo
        x_min: int = max(center_x - detection_range_cells, 0)
        x_max: int = min(center_x + detection_range_cells, GRID_WIDTH - 1)
        y_min: int = max(center_y - detection_range_cells, 0)
        y_max: int = min(center_y + detection_range_cells, GRID_HEIGHT - 1)
        
        # Criar grade de índices
        x_indices = np.arange(x_min, x_max + 1)
        y_indices = np.arange(y_min, y_max + 1)
        xv, yv = np.meshgrid(x_indices, y_indices, indexing='ij')
        
        # Calcular distâncias
        distances = np.sqrt((xv - center_x) ** 2 + (yv - center_y) ** 2)
        
        # Extrair região das matrizes
        if is_enemy:
            region_intensity = self.enemy_intensity[x_min:x_max+1, y_min:y_max+1]
            region_timestamp = self.enemy_timestamp[x_min:x_max+1, y_min:y_max+1]
        else:
            region_intensity = self.friend_intensity[x_min:x_max+1, y_min:y_max+1]
            region_timestamp = self.friend_timestamp[x_min:x_max+1, y_min:y_max+1]
        
        # Criar máscara para células vazias
        mask_empty = (distances <= detection_range_cells) & (region_intensity < 1)
        
        # Resetar intensidades e atualizar timestamps
        np.putmask(region_intensity, mask_empty, 0)
        np.putmask(region_timestamp, mask_empty, current_time)

    def _apply_broken_detection(
        self,
        detection_range: float,
        is_enemy: bool = True
    ) -> None:
        """
        Aplica comportamento de detecção falsa para drones quebrados.
        
        Args:
            detection_range: Raio de detecção em pixels
            is_enemy: Se True, aplica a enemy_intensity; se False, a friend_intensity
        """
        detection_range_cells: int = int(np.floor(detection_range / CELL_SIZE) * 0.8)
        center_x, center_y = pos_to_cell(self.pos)
        
        x_min: int = max(center_x - detection_range_cells, 0)
        x_max: int = min(center_x + detection_range_cells, GRID_WIDTH - 1)
        y_min: int = max(center_y - detection_range_cells, 0)
        y_max: int = min(center_y + detection_range_cells, GRID_HEIGHT - 1)
        
        region_shape: Tuple[int, int] = (x_max - x_min + 1, y_max - y_min + 1)
        
        # Gerar estados quebrados se necessário
        if is_enemy and self.broken_enemy_direction is None:
            self.broken_enemy_intensity, self.broken_enemy_direction = generate_sparse_matrix(
                region_shape, max_nonzero=10, seed=self.class_seed
            )
        elif not is_enemy and self.broken_friend_direction is None:
            self.broken_friend_intensity, self.broken_friend_direction = generate_sparse_matrix(
                region_shape, max_nonzero=10, seed=self.class_seed
            )
        
        # Calcular distâncias
        x_indices = np.arange(x_min, x_max + 1)
        y_indices = np.arange(y_min, y_max + 1)
        xv, yv = np.meshgrid(x_indices, y_indices, indexing='ij')
        distances = np.sqrt((xv - center_x) ** 2 + (yv - center_y) ** 2)
        mask = distances <= detection_range_cells
        
        if is_enemy:
            region_intensity = self.enemy_intensity[x_min:x_max+1, y_min:y_max+1]
            region_direction = self.enemy_direction[x_min:x_max+1, y_min:y_max+1]
            broken_intensity = self.broken_enemy_intensity
            broken_direction = self.broken_enemy_direction
        else:
            region_intensity = self.friend_intensity[x_min:x_max+1, y_min:y_max+1]
            region_direction = self.friend_direction[x_min:x_max+1, y_min:y_max+1]
            broken_intensity = self.broken_friend_intensity
            broken_direction = self.broken_friend_direction
        
        if self.timer_state_broken < self.update_state_broken:
            np.putmask(region_intensity, mask, broken_intensity[mask])
            np.putmask(
                region_direction,
                np.broadcast_to(mask[..., None], region_direction.shape),
                broken_direction
            )
            self.timer_state_broken += 1
        else:
            # Gerar novos estados aleatórios
            if is_enemy:
                self.broken_enemy_intensity, self.broken_enemy_direction = generate_sparse_matrix(
                    region_shape, max_nonzero=10, seed=self.class_seed
                )
            else:
                self.broken_friend_intensity, self.broken_friend_direction = generate_sparse_matrix(
                    region_shape, max_nonzero=10, seed=self.class_seed
                )
            
            current_time: int = pygame.time.get_ticks()
            if is_enemy:
                self.enemy_timestamp[x_min:x_max+1, y_min:y_max+1].fill(current_time)
            else:
                self.friend_timestamp[x_min:x_max+1, y_min:y_max+1].fill(current_time)
            
            self.timer_state_broken = 0

    def update_local_friend_detection(self, friend_drones: List['FriendDrone']) -> None:
        """
        Atualiza detecção local de drones amigos.
        
        Para cada drone amigo (excluindo drones AEW e self), atualiza a
        célula correspondente nas matrizes de detecção.
        
        Args:
            friend_drones: Lista de drones amigos
        """
        current_time: int = pygame.time.get_ticks()
        
        for friend in friend_drones:
            # Pular drones AEW e self
            if friend.behavior.type == "AEW" or friend is self:
                continue
            
            key: int = id(friend)
            distance: float = self.pos.distance_to(friend.pos)
            
            if distance >= COMMUNICATION_RANGE:
                self.current_friend_pos_detection.pop(key, None)
                self.aux_friend_detections.pop(key, None)
                continue
            
            cell: Tuple[int, int] = pos_to_cell(friend.pos)
            
            if key not in self.current_friend_pos_detection:
                self.current_friend_pos_detection[key] = friend.pos.copy()
            else:
                if key in self.aux_friend_detections:
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
        
        # Limpar células vazias
        self._clean_empty_cells(FRIEND_DETECTION_RANGE, current_time, is_enemy=False)
        
        # Aplicar detecção quebrada se aplicável
        if self.broken:
            self._apply_broken_detection(FRIEND_DETECTION_RANGE, is_enemy=False)

    def merge_enemy_matrix(self, neighbor: 'FriendDrone') -> None:
        """
        Mescla dados de detecção de inimigos de um drone vizinho.
        
        Propaga informação globalmente baseado em timestamps. Apenas atualiza
        células onde o vizinho tem informação mais recente.
        
        Args:
            neighbor: Drone vizinho para mesclar dados
            
        Note:
            Usa np.putmask para operações vetorizadas eficientes
        """
        update_mask = neighbor.enemy_timestamp > self.enemy_timestamp
        np.putmask(self.enemy_intensity, update_mask, neighbor.enemy_intensity)
        np.putmask(
            self.enemy_direction,
            np.broadcast_to(update_mask[..., None], self.enemy_direction.shape),
            neighbor.enemy_direction
        )
        np.putmask(self.enemy_timestamp, update_mask, neighbor.enemy_timestamp)

    def merge_friend_matrix(self, neighbor: 'FriendDrone') -> None:
        """
        Mescla dados de detecção de amigos de um drone vizinho.
        
        Args:
            neighbor: Drone vizinho para mesclar dados
        """
        update_mask = neighbor.friend_timestamp > self.friend_timestamp
        np.putmask(self.friend_intensity, update_mask, neighbor.friend_intensity)
        np.putmask(
            self.friend_direction,
            np.broadcast_to(update_mask[..., None], self.friend_direction.shape),
            neighbor.friend_direction
        )
        np.putmask(self.friend_timestamp, update_mask, neighbor.friend_timestamp)

    def update_neghbors(self, all_drones: List['FriendDrone']) -> None:
        """
        Atualiza lista de vizinhos próximos para comunicação.
        
        Seleciona os N_CONNECTIONS drones mais próximos dentro do alcance
        de comunicação, garantindo conexões bidirecionais.
        
        Args:
            all_drones: Lista de todos os drones amigos
            
        Note:
            Implementa estratégia de N vizinhos mais próximos com reciprocidade
        """
        # Candidatos dentro do alcance
        candidates = [
            other for other in all_drones
            if other is not self and self.pos.distance_to(other.pos) < COMMUNICATION_RANGE
        ]
        
        # Ordenar por distância e pegar os N mais próximos
        candidates.sort(key=lambda o: self.pos.distance_to(o.pos))
        nearest = candidates[:N_CONNECTIONS]

        # Identificar quem me considera entre seus N mais próximos
        reverse_neighbors: List['FriendDrone'] = []
        for other in all_drones:
            if other is self:
                continue
            
            others_cand = [
                o for o in all_drones
                if o is not other and other.pos.distance_to(o.pos) < COMMUNICATION_RANGE
            ]
            others_cand.sort(key=lambda o: other.pos.distance_to(o.pos))
            
            if self in others_cand[:N_CONNECTIONS]:
                reverse_neighbors.append(other)

        # União dos dois conjuntos
        neighbors: Set['FriendDrone'] = set(nearest + reverse_neighbors)
        self.neighbors = list(neighbors)

    def communication(self, all_drones: List['FriendDrone']) -> None:
        """
        Realiza comunicação com drones vizinhos.
        
        Troca informações de detecção (enemy e friend) com cada vizinho,
        considerando perda de mensagens probabilística.
        
        Args:
            all_drones: Lista de todos os drones amigos
            
        Note:
            Executa CICLE_COMM_BY_STEP ciclos de comunicação
        """
        connections: int = 0
        messages: int = 0
        
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
        """
        Transmite detecção passiva para vizinhos (não utilizado atualmente).
        
        Note:
            Método preparado para expansão futura do sistema de triangulação
        """
        connections: int = 0
        messages: int = 0
        
        for _ in range(CICLE_COMM_BY_STEP):
            for other in self.neighbors:
                connections += 1
                messages += 1
                
                self.merged_passive_detection_matrix = self.passive_detection_matrix.copy()
                if FriendDrone.class_rng.random() > MESSAGE_LOSS_PROBABILITY and other != self:
                    self.merged_passive_detection_matrix += other.passive_detection_matrix
                    
        self.active_connections += connections
        self.messages_sent_this_cycle += messages

    def take_action(self) -> None:
        """
        Executa ação do drone baseada em detecção de inimigos.
        
        Aplica comportamento apropriado, atualiza posição e garante que
        o drone permaneça dentro dos limites da simulação e do raio externo.
        """
        if self.return_to_base:
            # Retornar ao centro do ponto de interesse
            self.vel = (self.interest_point_center - self.pos).normalize() * FRIEND_SPEED
            self.info = ("REGRESS", None, None, None)
        else:
            # Aplicar movimento específico do comportamento
            self.apply_behavior()
            
        # Atualizar posição
        self.pos += self.vel

        # Manter drone dentro dos limites da simulação
        self._enforce_simulation_bounds()
            
        # Prevenir drone de exceder EXTERNAL_RADIUS do ponto de interesse
        self._enforce_external_radius()

    def _enforce_simulation_bounds(self) -> None:
        """
        Garante que o drone permaneça dentro dos limites da simulação.
        
        Se uma borda for atingida, inverte o componente correspondente da velocidade.
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
        Garante que o drone permaneça dentro do raio máximo do ponto de interesse.
        
        Se o limite for excedido, move o drone de volta para a borda e o para.
        """
        distance_to_center: float = self.pos.distance_to(self.interest_point_center)
        
        if distance_to_center > EXTERNAL_RADIUS:
            direction = (self.pos - self.interest_point_center).normalize()
            self.pos = self.interest_point_center + direction * EXTERNAL_RADIUS
            self.vel = pygame.math.Vector2(0, 0)

    def update(
        self,
        enemy_drones: List[Any],
        friend_drones: List['FriendDrone'],
        use_triangulation: bool,
        return_to_base: bool = False
    ) -> None:
        """
        Atualiza o estado do drone para o passo atual da simulação.
        
        Inclui decaimento de matrizes, atualização de detecções locais,
        comunicação com drones próximos e execução de ações.
        
        Args:
            enemy_drones: Lista de drones inimigos
            friend_drones: Lista de drones amigos
            use_triangulation: Se True, usa modo de triangulação
            return_to_base: Se True, drone retornará ao centro
        """
        self.total_steps += 1
        self.active_connections = 0
        self.messages_sent_this_cycle = 0
        self.return_to_base = return_to_base
        
        self.detection_mode = "triangulation" if use_triangulation else "direct"
        
        # Aplicar decaimento exponencial às matrizes de detecção
        self.decay_matrices()
        
        # Atualizar vizinhos próximos
        self.update_neghbors(friend_drones)
        
        if self.detection_mode == "triangulation":
            self.update_passive_detection(enemy_drones)
            
        # Atualizar detecções locais
        self.update_local_enemy_detection(friend_drones, enemy_drones)
        self.update_local_friend_detection(friend_drones)
        
        # Comunicar com drones próximos
        self.communication(friend_drones)
        
        # Executar ação de movimento
        self.take_action()
        
        # Calcular distância percorrida
        self.distance_traveled += self.pos.distance_to(self.last_position)
        self.last_position = self.pos.copy()
        
        # Registrar trajetória para visualização
        if len(self.trajectory) > 300:
            self.trajectory.pop(0)
        self.trajectory.append(self.pos.copy())
        
        # Atualizar histórico de estados
        if self.current_state:
            if self.current_state in self.state_history:
                self.state_history[self.current_state] += 1
            else:
                self.state_history[self.current_state] = 1

    def get_state_percentages(self) -> Dict[str, float]:
        """
        Calcula a porcentagem de passos gastos em cada estado.
        
        Returns:
            Dicionário mapeando nomes de estados para porcentagem de tempo
        """
        if self.total_steps == 0:
            return {}
            
        percentages: Dict[str, float] = {}
        for state, count in self.state_history.items():
            percentages[state] = (count / self.total_steps) * 100
                
        return percentages

    def apply_behavior(self) -> None:
        """
        Atualiza velocidade do drone baseado em seu tipo de comportamento.
        
        O behavior object é responsável por determinar a velocidade
        e informações de debug do drone.
        
        Note:
            Se fixed=True, o drone permanece estacionário
        """
        if self.fixed:
            self.vel = pygame.math.Vector2(0, 0)
            self.info = ("FIXED", None, None, None)
            self.current_state = self.info[0] if isinstance(self.info, tuple) else self.info
        else:
            state = {
                'drone_id': self.drone_id,
                'pos': np.array([[self.pos.x, self.pos.y]], dtype=np.float32),
                'friend_intensity': np.expand_dims(self.friend_intensity, axis=0),
                'enemy_intensity': np.expand_dims(self.enemy_intensity, axis=0),
                'friend_direction': np.expand_dims(self.friend_direction, axis=0),
                'enemy_direction': np.expand_dims(self.enemy_direction, axis=0)
            }
            
            self.info, self.vel = self.behavior.apply(state, self.joystick_controlled)
            
        self.current_state = self.info[0] if isinstance(self.info, tuple) else self.info

    def draw(
        self,
        surface: pygame.Surface,
        show_detection: bool = True,
        show_comm_range: bool = True,
        show_trajectory: bool = False,
        show_debug: bool = False
    ) -> None:
        """
        Desenha o drone e suas informações na superfície fornecida.
        
        Args:
            surface: Superfície para desenhar
            show_detection: Se True, exibe alcance de detecção
            show_comm_range: Se True, exibe alcance de comunicação
            show_trajectory: Se True, desenha trajetória do drone
            show_debug: Se True, mostra informações de debug adicionais
        """
        # Desenhar trajetória se habilitado
        if show_trajectory and len(self.trajectory) > 1:
            self._draw_trajectory(surface)
        
        # Desenhar alcance de detecção
        if show_detection or self.broken:
            detection_range: float = self._get_detection_range()
            draw_dashed_circle(
                surface,
                (self.color[0], self.color[1], self.color[2], 32),
                (int(self.pos.x), int(self.pos.y)),
                int(detection_range),
                5, 5, 1
            )
            
        # Desenhar alcance de comunicação
        if show_comm_range:
            draw_dashed_circle(
                surface,
                (255, 255, 0, 32),
                (int(self.pos.x), int(self.pos.y)),
                COMMUNICATION_RANGE,
                5, 5, 1
            )
        
        # Desenhar imagem do drone
        image_rect = self.image.get_rect(center=(int(self.pos.x), int(self.pos.y)))
        surface.blit(self.image, image_rect)
        
        # Desenhar ID do drone
        self._draw_drone_id(surface)
        
        # Desenhar informações de debug se habilitado
        if show_debug:
            self._draw_debug_info(surface)

    def _draw_trajectory(self, surface: pygame.Surface) -> None:
        """
        Desenha a trajetória de movimento do drone com efeito de desvanecimento.
        
        Args:
            surface: Superfície para desenhar
        """
        traj_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        decay_rate: float = 0.04
        n: int = len(self.trajectory)
        
        for i in range(n - 1):
            d: int = n - 1 - i
            alpha: int = int(255 * math.exp(-decay_rate * d))
            alpha = max(alpha, 30)  # Alpha mínimo
            
            color_with_alpha = self.color + (alpha,)
            start_pos = (int(self.trajectory[i].x), int(self.trajectory[i].y))
            end_pos = (int(self.trajectory[i+1].x), int(self.trajectory[i+1].y))
            
            pygame.draw.line(traj_surf, color_with_alpha, start_pos, end_pos, 2)
            
        surface.blit(traj_surf, (0, 0))

    def _draw_drone_id(self, surface: pygame.Surface) -> None:
        """
        Desenha o ID do drone e labels de status.
        
        Args:
            surface: Superfície para desenhar
        """
        font: pygame.font.Font = pygame.font.SysFont(FONT_FAMILY, 10)
        
        # Desenhar ID do drone
        label = font.render(f"ID: F{self.drone_id}", True, (255, 255, 255))
        label.set_alpha(128)
        surface.blit(label, (int(self.pos.x) + 20, int(self.pos.y) - 20))
        
        # Desenhar indicador de líder se aplicável
        if self.is_leader:
            leader_label = font.render("LEADER", True, (0, 255, 0))
            surface.blit(leader_label, (int(self.pos.x) + 20, int(self.pos.y) - 5))
        
        # Desenhar indicador de seleção se aplicável
        if self.selected:
            selected_label = font.render("GRAPH", True, (0, 255, 0))
            surface.blit(selected_label, (int(self.pos.x) + 20, int(self.pos.y) + 10))

    def _draw_debug_info(self, surface: pygame.Surface) -> None:
        """
        Desenha informações de debug adicionais.
        
        Args:
            surface: Superfície para desenhar
        """
        # Desenhar informações de detecção passiva se em modo triangulação
        if self.detection_mode == "triangulation":
            for direction in self.direction_vectors.values():
                # Calcular ponto final (estendendo a direção)
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
                
        # Desenhar informações de estado
        font = pygame.font.SysFont(FONT_FAMILY, 10)
        if self.info and self.info[0]:
            len_info: int = len(self.info[0])
            debug_label = font.render(self.info[0], True, (255, 215, 0))
            surface.blit(debug_label, (int(self.pos.x) - 3.5 * len_info, int(self.pos.y) + 25))
            
            # Desenhar informações de alvo se disponíveis
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
                
            # Desenhar linha do ponto de interesse se disponível
            if self.info[2] is not None:
                pygame.draw.line(
                    surface,
                    (255, 215, 0),
                    (int(self.interest_point_center[0]), int(self.interest_point_center[1])),
                    (int(self.info[2].x), int(self.info[2].y)),
                    2
                )