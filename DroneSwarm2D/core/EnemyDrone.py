"""
EnemyDrone.py

Define a classe EnemyDrone que representa drones ofensivos na simulação.

O drone exibe diversos comportamentos (determinísticos e estocásticos) para
se aproximar ou orbitar um ponto de interesse, mantendo-se dentro dos limites
da simulação.

Behaviors disponíveis:
- direct: Movimento direto ao alvo
- zigzag: Aproximação com oscilações
- spiral: Movimento em espiral
- alternating: Alternância entre direções
- bounce_approach: Avanço e recuo
- circle_wait_advance: Espera circular e avanço
- formation/biformation: Movimentos em formação
- joystick: Controle manual via joystick
"""

import math
import random
from typing import Tuple, Optional
from pathlib import Path

import numpy as np
import pygame

from .settings import (
    ENEMY_SPEED,
    ENEMY_DETECTION_RANGE,
    SIM_WIDTH,
    SIM_HEIGHT,
    ESCAPE_STEPS,
    INITIAL_AGGRESSIVENESS,
    INITIAL_DISTANCE,
    EXTERNAL_RADIUS,
    ENEMY_COUNT,
    FONT_FAMILY,
)
from .utils import draw_dashed_circle, load_svg_as_surface

# Obter o diretório deste arquivo
_MODULE_DIR = Path(__file__).parent

# Construir caminhos absolutos para os assets
_ASSETS_DIR = _MODULE_DIR.parent / "assets"  # DroneSwarm2D/assets/


class EnemyDrone:
    """
    Representa um drone inimigo com diversos comportamentos de aproximação/órbita.
    
    O drone pode ter vários tipos de comportamento (ex: "direct", "zigzag", "spiral", etc.)
    e recebe um identificador único. Atualiza sua posição baseado em sua velocidade e
    permanece dentro da área de simulação.
    
    Attributes:
        interest_point_center: Centro do ponto de interesse alvo
        pos: Posição atual como Vector2
        vel: Velocidade atual como Vector2
        drone_id: Identificador único do drone
        behavior_type: Tipo de comportamento atual
        color: Cor RGB para renderização
        joystick: Objeto joystick pygame (se aplicável)
        detector: Posição do drone amigo que detectou este (ou None)
        escape_steps_count: Contador de passos de fuga
        desperate_attack: Flag para ataque desesperado
        aggressiveness: Nível de agressividade (0-1+)
        trajectory: Lista de posições históricas
        
    Class Attributes:
        enemy_id_counter: Contador global para IDs únicos
        class_seed: Seed compartilhada pela classe
        class_rng: Random Number Generator da classe
        class_np_rng: Numpy RNG da classe
    """
    
    enemy_id_counter: int = 0
    original_drone_image: pygame.Surface = load_svg_as_surface(f"{_ASSETS_DIR}/drone_9.svg")
    original_drone_image_joystick: pygame.Surface = load_svg_as_surface(f"{_ASSETS_DIR}/drone_10.svg")

    # Seed da classe - inicialmente None, será gerada se não definida explicitamente
    class_seed: Optional[int] = None
    class_rng: Optional[random.Random] = None
    class_np_rng: Optional[np.random.RandomState] = None
    
    def __init__(
        self, 
        interest_point_center: pygame.math.Vector2, 
        position: Optional[Tuple[float, float]] = None, 
        behavior_type: Optional[str] = None, 
        fixed: bool = False
    ) -> None:
        """
        Inicializa o drone inimigo com ponto alvo e comportamento.

        Args:
            interest_point_center: Centro do ponto de interesse alvo.
            position: Posição inicial (x, y). Se None, usa posição aleatória na borda.
            behavior_type: Tipo de comportamento (ex: "direct", "zigzag", etc.).
                          Se None, um comportamento é escolhido aleatoriamente.
            fixed: Se True, o drone permanece estacionário.
        """
        if EnemyDrone.class_seed is None:
            EnemyDrone.set_class_seed()
            
        self.interest_point_center: pygame.math.Vector2 = interest_point_center
        self.joystick: Optional[pygame.joystick.Joystick] = None
        
        # Definir posição inicial
        if position is None:
            self.pos: pygame.math.Vector2 = self.random_border_position()
        else:
            self.pos = pygame.math.Vector2(position[0], position[1])
                
        # Inicializar velocidade direcionada ao ponto de interesse
        target_vector: pygame.math.Vector2 = self.interest_point_center - self.pos
        direction = target_vector.normalize() if target_vector.length() > 0 else pygame.math.Vector2(0, 0)
        self.vel: pygame.math.Vector2 = direction * ENEMY_SPEED
        self.info: str = ""
        self.aggressiveness: float = 0.0
        
        # Definir cor e atribuir ID único
        self.color: Tuple[int, int, int] = (255, 0, 0)
        self.drone_id: int = self._assign_id()
        
        # Atributos auxiliares para comportamentos específicos
        self.phase: float = 0.0
        self.timer: int = 0
        
        # Atribuir tipo de comportamento
        if behavior_type is None:
            self.behavior_type: str = EnemyDrone.class_rng.choice([
                "direct", "zigzag", "zigzag_damped", "zigzag_unstable", "zigzag_variable_period",
                "spiral", "spiral_bounce", "spiral_oscillatory",
                "alternating", "bounce_approach", "circle_wait_advance"
            ])
        else:
            self.behavior_type = behavior_type
            
        if behavior_type == "joystick":
            self._init_joystick()

        # Atraso inicial (exceto para comportamentos especiais)
        if not self.behavior_type in ["formation", "biformation", "focal-direct", "debug", "u-debug", "joystick"]:
            self.start_delay: int = EnemyDrone.class_rng.randint(0, 100)
        else:
            self.start_delay = 0
            
        self.fixed: bool = fixed
        self.trajectory: list[pygame.math.Vector2] = []
        
        # Configurar imagem do drone
        if self.behavior_type == "joystick":
            desired_width: int = int(SIM_WIDTH * 0.02)
            aspect_ratio: float = self.original_drone_image_joystick.get_height() / self.original_drone_image_joystick.get_width()
            desired_height: int = int(desired_width * aspect_ratio)
            self.drone_image: pygame.Surface = pygame.transform.scale(
                self.original_drone_image_joystick, 
                (desired_width, desired_height)
            )
        else:
            desired_width = int(SIM_WIDTH * 0.02)
            aspect_ratio = self.original_drone_image.get_height() / self.original_drone_image.get_width()
            desired_height = int(desired_width * aspect_ratio)
            self.drone_image = pygame.transform.scale(
                self.original_drone_image, 
                (desired_width, desired_height)
            )
        
        # Novos atributos para detecção e agressividade
        self.detector: Optional[pygame.math.Vector2] = None  # Indica detecção por drones amigos
        self.escape_steps: int = ESCAPE_STEPS  # Número de passos de fuga
        self.escape_steps_count: int = 0  # Contador de passos de fuga
        self.desperate_attack: bool = False  # Indica se está em ataque desesperado
            
    @classmethod
    def set_class_seed(cls, seed: Optional[int] = None) -> None:
        """
        Define a seed compartilhada da classe para geração de números aleatórios.
        
        Args:
            seed: Seed a ser usada. Se None, gera uma seed aleatória.
            
        Note:
            Inicializa tanto random.Random quanto np.random.RandomState
        """
        cls.class_seed = seed if seed is not None else random.randint(0, 10000000)
        cls.class_rng = random.Random(cls.class_seed)
        cls.class_np_rng = np.random.RandomState(cls.class_seed)
        
    def _init_joystick(self) -> None:
        """
        Inicializa controle por joystick se disponível.
        
        Note:
            Imprime mensagem no console se joystick for conectado
        """
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Joystick '{self.joystick.get_name()}' conectado.")

    def random_border_position(self) -> pygame.math.Vector2:
        """
        Gera uma posição inicial aleatória ao longo de uma das bordas da simulação.

        Returns:
            Vector2 com posição na borda (top, bottom, left ou right)
        """
        side: str = EnemyDrone.class_rng.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top':
            return pygame.math.Vector2(EnemyDrone.class_rng.uniform(0, SIM_WIDTH), 0)
        elif side == 'bottom':
            return pygame.math.Vector2(EnemyDrone.class_rng.uniform(0, SIM_WIDTH), SIM_HEIGHT)
        elif side == 'left':
            return pygame.math.Vector2(0, EnemyDrone.class_rng.uniform(0, SIM_HEIGHT))
        else:  # right
            return pygame.math.Vector2(SIM_WIDTH, EnemyDrone.class_rng.uniform(0, SIM_HEIGHT))

    def _assign_id(self) -> int:
        """
        Atribui um identificador único a este drone.

        Returns:
            ID único do drone
        """
        current_id: int = self.__class__.enemy_id_counter
        EnemyDrone.enemy_id_counter += 1
        return current_id
    

    def update(self, detector: Optional[pygame.math.Vector2] = None) -> None: 
        """
        Atualiza a posição do drone baseado em seu comportamento e velocidade,
        garantindo que permaneça dentro dos limites da simulação.
        
        O drone para de se mover quando atinge o ponto de interesse.
        
        Args:
            detector: Posição de um drone amigo que detectou este drone.
                     Se não None, pode ativar comportamento de fuga.
        """
        self.detector = detector
        self.timer += 1

        # Aguardar pelo atraso inicial antes de mover
        if self.timer < self.start_delay:
            return

        # Snap ao ponto de interesse se muito próximo
        distance_to_interest: float = (self.interest_point_center - self.pos).length()
        if distance_to_interest < 1:
            self.pos = self.interest_point_center.copy()
            self.vel = pygame.math.Vector2(0, 0)
            return

        # Calcular agressividade baseada na distância
        if distance_to_interest > INITIAL_DISTANCE:
            self.aggressiveness = max(
                INITIAL_AGGRESSIVENESS,
                1 + (distance_to_interest - INITIAL_DISTANCE) * 
                ((INITIAL_AGGRESSIVENESS - 1) / (EXTERNAL_RADIUS - INITIAL_DISTANCE))
            )
        else:
            self.aggressiveness = 1.0
        
        # Atualizar velocidade baseada no comportamento selecionado
        self.apply_behavior()
        self.pos += self.vel
        
        # Manter o drone dentro dos limites da simulação
        self._enforce_bounds()
            
        # Atualizar trajetória
        if len(self.trajectory) > 300:
            self.trajectory.pop(0)
        self.trajectory.append(self.pos.copy())

    def _enforce_bounds(self) -> None:
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

    def apply_behavior(self) -> None:
        """
        Atualiza a velocidade do drone de acordo com seu comportamento escolhido,
        combinando elementos determinísticos e estocásticos.
        
        Comportamentos disponíveis:
        - Ataque desesperado: movimento direto ao ponto de interesse
        - Fuga: afastamento do detector
        - Comportamentos específicos: direct, zigzag, spiral, etc.
        
        Note:
            A agressividade determina se o drone atacará ou fugirá quando detectado.
        """
        # Ataque desesperado: mover diretamente ao ponto de interesse
        if self.desperate_attack:
            target_vector = self.interest_point_center - self.pos
            self.vel = target_vector.normalize() * ENEMY_SPEED if target_vector.length() > 0 else pygame.math.Vector2(0, 0)
            self.info = "DESPERATE ATTACK"
            return

        # Comportamento de fuga: afastar-se do detector ou ponto de interesse
        if self.escape_steps_count > 0:
            if self.detector is not None:
                direction = (self.pos - self.detector).normalize()
            else:
                direction = (self.pos - self.interest_point_center).normalize()
                
            self.vel = direction * ENEMY_SPEED
            self.escape_steps_count += 1
            self.info = f"ESCAPE {self.escape_steps_count}/{self.escape_steps} STEPS"
            
            if self.escape_steps_count >= self.escape_steps:
                self.escape_steps_count = 0
                self.detector = None
                if self.behavior_type in ["formation", "biformation", "focal-direct"]:
                    self.behavior_type = "direct"
            return

        # Agressividade: se detectado, decidir entre atacar ou fugir
        if self.detector and self.behavior_type != 'joystick':
            if EnemyDrone.class_np_rng.rand() < self.aggressiveness:
                self.desperate_attack = True
                self.info = "DESPERATE ATTACK"
            else:
                self.escape_steps_count += 1
                return

        # Comportamento padrão: calcular vetor ao ponto de interesse
        target_vector = self.interest_point_center - self.pos
        if target_vector.length() == 0:
            return  # Evitar divisão por zero
        
        self.info = f"aggr.: {self.aggressiveness:.2f}"

        # Aplicar comportamento específico
        if self.behavior_type == "joystick":
            self._behavior_joystick()
        elif self.behavior_type == "direct":
            self._behavior_direct(target_vector)
        elif self.behavior_type.startswith("zigzag"):
            self._behavior_zigzag(target_vector)
        elif self.behavior_type.startswith("spiral"):
            self._behavior_spiral()
        elif self.behavior_type == "alternating":
            self._behavior_alternating(target_vector)
        elif self.behavior_type == "bounce_approach":
            self._behavior_bounce_approach()
        elif self.behavior_type == "circle_wait_advance":
            self._behavior_circle_wait_advance()
        elif self.behavior_type in ["formation", "biformation"]:
            self._behavior_formation()
        elif self.behavior_type == "focal-direct":
            self._behavior_focal_direct()
        elif self.behavior_type == "debug":
            self._behavior_debug()
        elif self.behavior_type == "u-debug":
            self._behavior_u_debug(target_vector)
        else:
            # Comportamento padrão: direto
            self._behavior_direct(target_vector)

    def _behavior_joystick(self) -> None:
        """Comportamento controlado por joystick."""
        if self.joystick is not None:
            x_axis = self.joystick.get_axis(0) 
            y_axis = self.joystick.get_axis(1)
            direction = pygame.math.Vector2(x_axis, y_axis)
            
            if direction.length() > 0.1:
                direction = direction.normalize()
            else:
                direction = pygame.math.Vector2(0, 0)
                
            if self.detector is not None:
                self.joystick.rumble(1, 0, 50)
            else:
                self.joystick.rumble(0, self.aggressiveness, 50)

            self.vel = direction * ENEMY_SPEED
            self.info = "JOYSTICK"
        else:
            self.info = "NO JOYSTICK"

    def _behavior_direct(self, target_vector: pygame.math.Vector2) -> None:
        """Movimento direto ao ponto de interesse."""
        self.vel = target_vector.normalize() * ENEMY_SPEED

    def _behavior_zigzag(self, target_vector: pygame.math.Vector2) -> None:
        """Aproximação com oscilações senoidais."""
        base_direction = target_vector.normalize()
        self.phase += 0.1
        perp = pygame.math.Vector2(-base_direction.y, base_direction.x)
        
        if self.behavior_type == "zigzag":
            amplitude = 2.0
            offset = math.sin(self.phase) * amplitude
        elif self.behavior_type == "zigzag_damped":
            decay_factor = math.exp(-0.01 * self.timer)
            amplitude = 5.0 * decay_factor
            offset = math.sin(self.phase) * amplitude
        elif self.behavior_type == "zigzag_unstable":
            random_amplitude = EnemyDrone.class_rng.uniform(1.0, 10.0)
            random_frequency = EnemyDrone.class_rng.uniform(0.5, 2.0)
            offset = math.sin(random_frequency * self.timer) * random_amplitude
        else:  # zigzag_variable_period
            base_frequency = 1.0
            frequency_variation = 0.5 * math.sin(0.05 * self.timer)
            effective_frequency = base_frequency + frequency_variation
            amplitude = 6.0
            offset = math.sin(effective_frequency * self.phase) * amplitude
            
        direction = (base_direction + perp * offset).normalize()
        self.vel = direction * ENEMY_SPEED

    def _behavior_spiral(self) -> None:
        """Movimento em espiral em direção ao ponto de interesse."""
        r_vector = self.pos - self.interest_point_center
        current_distance = r_vector.length()
        angle = math.atan2(r_vector.y, r_vector.x) + 0.1
        
        if self.behavior_type == "spiral":
            new_radius = current_distance * 0.95
        else:  # spiral_bounce ou spiral_oscillatory
            if not hasattr(self, 'sb_state'):
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
            else:  # bounce
                direction = (-(self.interest_point_center - self.pos)).normalize()
                direction = direction.rotate(self.bounce_deviation)
                self.sb_timer += 1
                if self.sb_timer >= self.bounce_duration:
                    self.sb_state = "spiral"
                    self.sb_timer = 0
                self.vel = direction * ENEMY_SPEED
                return
        
        new_offset = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * new_radius
        desired_pos = self.interest_point_center + new_offset
        direction = (desired_pos - self.pos)
        
        if self.behavior_type == "spiral_oscillatory" and direction.length() > 0:
            base_direction = direction.normalize()
            self.phase += 0.05
            perp = pygame.math.Vector2(-base_direction.y, base_direction.x)
            offset = math.sin(self.phase)
            direction = (base_direction + perp * offset).normalize() * direction.length()
        
        self.vel = direction.normalize() * ENEMY_SPEED if direction.length() != 0 else pygame.math.Vector2(0, 0)

    def _behavior_alternating(self, target_vector: pygame.math.Vector2) -> None:
        """Alterna entre aproximação direta e órbita perpendicular."""
        cycle_length = 100
        phase_in_cycle = self.timer % cycle_length
        target_direction = target_vector.normalize()
        
        if phase_in_cycle < cycle_length / 2:
            direction = target_direction
        else:
            direction = pygame.math.Vector2(-target_direction.y, target_direction.x)
            noise_angle = EnemyDrone.class_rng.uniform(-0.1, 0.1)
            direction = direction.rotate_rad(noise_angle).normalize()
        
        self.vel = direction * ENEMY_SPEED

    def _behavior_bounce_approach(self) -> None:
        """Alterna entre aproximação e recuo com desvio."""
        if not hasattr(self, 'state'):
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
            self.vel = direction * ENEMY_SPEED
        else:  # retreat
            direction = (-(self.interest_point_center - self.pos)).normalize()
            direction = direction.rotate(self.deviation_angle)
            self.state_timer += 1
            if self.state_timer >= self.retreat_duration:
                self.state = "approach"
                self.state_timer = 0
            self.vel = direction * ENEMY_SPEED

    def _behavior_circle_wait_advance(self) -> None:
        """Alterna entre espera (movimento circular) e avanço."""
        if not hasattr(self, 'cw_state'):
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
        else:  # advance
            direction = (self.interest_point_center - self.pos).normalize()
            offset_angle = EnemyDrone.class_rng.uniform(-5, 5)
            direction = direction.rotate(offset_angle)
            self.vel = direction * ENEMY_SPEED
            self.cw_timer += 1
            if self.cw_timer >= self.advance_duration:
                self.cw_state = "wait"
                self.cw_timer = 0

    def _behavior_formation(self) -> None:
        """Movimento em formação (formation ou biformation)."""
        if not hasattr(self, 'formation_id'):
            self.formation_id = self.drone_id % ENEMY_COUNT
            FRONT_FORMATION = 5
            
            if self.behavior_type == "formation":
                row = self.formation_id // FRONT_FORMATION
                column = self.formation_id % FRONT_FORMATION - FRONT_FORMATION // 2
                self.pos = pygame.math.Vector2(SIM_WIDTH, SIM_HEIGHT // 2 + column * 60)
                self.start_delay = 40 * row
            else:  # biformation
                if self.formation_id < ENEMY_COUNT // 2:
                    formation_index = self.formation_id
                    row = formation_index // FRONT_FORMATION
                    column = formation_index % FRONT_FORMATION - FRONT_FORMATION // 2
                    self.pos = pygame.math.Vector2(SIM_WIDTH, SIM_HEIGHT // 2 + column * 60)
                    self.vel = pygame.math.Vector2(-1, 0)
                    self.start_delay = 40 * row
                else:
                    formation_index = self.formation_id - (ENEMY_COUNT // 2)
                    row = formation_index // FRONT_FORMATION
                    column = formation_index % FRONT_FORMATION - FRONT_FORMATION // 2
                    self.pos = pygame.math.Vector2(0, SIM_HEIGHT // 2 + column * 60)
                    self.vel = pygame.math.Vector2(1, 0)
                    self.start_delay = 40 * row

        if self.pos.distance_to(self.interest_point_center) > INITIAL_DISTANCE:
            self.info = f"#{self.formation_id}"
        else:
            self.behavior_type = "direct"

    def _behavior_focal_direct(self) -> None:
        """Comportamento focal-direct."""
        self.pos = pygame.math.Vector2(SIM_WIDTH, 0)
        self.start_delay = 20 * (self.drone_id % ENEMY_COUNT)
        self.behavior_type = "direct"

    def _behavior_debug(self) -> None:
        """Comportamento de debug: movimento circular horário."""
        if not hasattr(self, 'debug_orbit_radius'):
            center_vec = pygame.math.Vector2(
                self.interest_point_center.x if hasattr(self.interest_point_center, 'x') else self.interest_point_center[0],
                self.interest_point_center.y if hasattr(self.interest_point_center, 'y') else self.interest_point_center[1]
            )
            self.debug_orbit_radius = self.pos.distance_to(center_vec)
            self.debug_angular_speed = 0.02
        
        center_x = self.interest_point_center.x if hasattr(self.interest_point_center, 'x') else self.interest_point_center[0]
        center_y = self.interest_point_center.y if hasattr(self.interest_point_center, 'y') else self.interest_point_center[1]
        
        radius_vector = self.pos - pygame.math.Vector2(center_x, center_y)
        tangent_vector = pygame.math.Vector2(radius_vector.y, -radius_vector.x)
        
        if tangent_vector.length() > 0:
            direction = tangent_vector.normalize()
            self.vel = direction * ENEMY_SPEED
        else:
            self.vel = pygame.math.Vector2(0, 0)
        
        self.info = "ORBIT"

    def _behavior_u_debug(self, target_vector: pygame.math.Vector2) -> None:
        """Comportamento U-debug: avançar, perpendicular, recuar."""
        if not hasattr(self, 'u_debug_phase'):
            self.u_debug_phase = 0
            self.u_debug_timer = 0
            self.forward_steps = 300
            self.perp_steps = 40

        target_direction = target_vector.normalize()
        if not self.fixed:
            if self.u_debug_phase == 0:
                self.vel = target_direction * ENEMY_SPEED
                self.u_debug_timer += 1
                if self.u_debug_timer >= self.forward_steps:
                    self.u_debug_phase = 1
                    self.u_debug_timer = 0
            elif self.u_debug_phase == 1:
                perp_direction = pygame.math.Vector2(-target_direction.y, target_direction.x)
                self.vel = perp_direction * ENEMY_SPEED
                self.u_debug_timer += 1
                if self.u_debug_timer >= self.perp_steps:
                    self.u_debug_phase = 2
                    self.u_debug_timer = 0
            else:  # phase 2
                self.vel = (-target_direction) * ENEMY_SPEED
                self.u_debug_timer += 1
                if self.u_debug_timer >= self.forward_steps:
                    self.vel = pygame.math.Vector2(0, 0)
        else:
            self.vel = pygame.math.Vector2(0, 0)

    def draw(
        self, 
        surface: pygame.Surface, 
        show_detection: bool = True, 
        show_trajectory: bool = False, 
        show_debug: bool = False
    ) -> None:
        """
        Desenha o drone inimigo na superfície fornecida.

        Args:
            surface: Superfície pygame onde o drone será desenhado.
            show_detection: Se True, exibe o alcance de detecção do drone.
            show_trajectory: Se True, exibe a trajetória do drone.
            show_debug: Se True, exibe informações de debug.
        """
        # Desenhar trajetória se habilitado
        if show_trajectory and len(self.trajectory) > 1:
            traj_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            decay_rate = 0.05
            n = len(self.trajectory)
            for i in range(n - 1):
                d = n - 1 - i
                alpha = int(255 * math.exp(-decay_rate * d))
                alpha = max(alpha, 30)  # Mínimo de alpha
                color_with_alpha = self.color + (alpha,)
                start_pos = (int(self.trajectory[i].x), int(self.trajectory[i].y))
                end_pos = (int(self.trajectory[i+1].x), int(self.trajectory[i+1].y))
                pygame.draw.line(traj_surf, color_with_alpha, start_pos, end_pos, 2)
            surface.blit(traj_surf, (0, 0))
        
        # Desenhar imagem do drone
        image_rect = self.drone_image.get_rect(center=(int(self.pos.x), int(self.pos.y)))
        surface.blit(self.drone_image, image_rect)
        
        # Opcionalmente desenhar alcance de detecção
        if show_detection:
            draw_dashed_circle(
                surface, 
                (self.color[0], self.color[1], self.color[2], 64), 
                (int(self.pos.x), int(self.pos.y)),
                ENEMY_DETECTION_RANGE, 
                dash_length=5, 
                space_length=5, 
                width=1
            )
        
        # Renderizar ID do drone
        font = pygame.font.SysFont(FONT_FAMILY, 10)
        label = font.render(f"ID: E{self.drone_id}", True, (255, 255, 255))
        label.set_alpha(128)
        surface.blit(label, (int(self.pos.x) + 20, int(self.pos.y) - 20))
        
        # Informações de debug
        if show_debug:
            len_info = len(self.info)
            debug_label = font.render(self.info, True, (255, 215, 0))
            surface.blit(debug_label, (int(self.pos.x) - 3.5 * len_info, int(self.pos.y) + 25))