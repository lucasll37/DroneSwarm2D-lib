# type: ignore
"""
enemy_drone.py

This module defines the EnemyDrone class which represents an enemy drone in the simulation.
The drone exhibits various behaviors (deterministic and stochastic) to approach or orbit an 
interest point while remaining within simulation boundaries.
"""

# Standard libraries
import math
import random
from typing import Tuple, List, Optional
from pathlib import Path

# Third-party libraries
import numpy as np
import pygame


# Project-specific imports
from .settings import *
from .utils import draw_dashed_circle, load_svg_as_surface

# Obter o diretório deste arquivo
_MODULE_DIR = Path(__file__).parent

# Construir caminhos absolutos para os assets
_ASSETS_DIR = _MODULE_DIR.parent / "assets"  # DroneSwarm2D/assets/

# -----------------------------------------------------------------------------
# EnemyDrone Class
# -----------------------------------------------------------------------------
class EnemyDrone:
    """
    Represents an enemy drone with various behaviors to approach or orbit an interest point.
    
    The drone can have several behavior types (e.g., "direct", "zigzag", "spiral", etc.) and
    is assigned a unique identifier. It updates its position based on its velocity and remains
    within the simulation area.
    """
    
    enemy_id_counter: int = 0
    original_drone_image: pygame.Surface = load_svg_as_surface(f"{_ASSETS_DIR}/drone_9.svg")
    original_drone_image_joystick: pygame.Surface = load_svg_as_surface(f"{_ASSETS_DIR}/drone_10.svg")

    # Seed da classe - inicialmente None, será gerada se não for definida explicitamente
    class_seed = None
    class_rng = None
    class_np_rng = None
    
    def __init__(self, interest_point_center: pygame.math.Vector2, 
                 position: Optional[Tuple] = None, 
                 behavior_type: Optional[str] = None, 
                 fixed: bool = False) -> None:
        """
        Initialize the enemy drone with a target interest point and behavior.

        Args:
            interest_point_center (pygame.math.Vector2): Center of the target interest point.
            position (Optional[Tuple]): Starting position; if None, a random border position is used.
            behavior_type (Optional[str]): Type of behavior (e.g., "direct", "zigzag", etc.). If None, one is randomly selected.
            fixed (bool): If True, the drone remains stationary.
        """
        
        if EnemyDrone.class_seed is None:
            EnemyDrone.set_class_seed()
            
        self.interest_point_center = interest_point_center
        self.joystick = None
        
        # Set starting position: use provided position or generate a random border position.
        if position is None:
            self.pos = self.random_border_position()
        else:
            self.pos = pygame.math.Vector2(position[0], position[1])
                
        # Initialize velocity directed toward the interest point.
        target_vector = self.interest_point_center - self.pos
        direction = target_vector.normalize() if target_vector.length() > 0 else pygame.math.Vector2(0, 0)
        self.vel = direction * ENEMY_SPEED
        self.info: str = ""
        self.aggressiveness: float = 0.0
        
        # Set display color and assign a unique drone ID.
        self.color: Tuple[int, int, int] = (255, 0, 0)
        self.drone_id: int = self.assign_id()
        
        # Auxiliary attributes for specific behaviors.
        self.phase = 0
        self.timer = 0
        
        # Assign behavior type; if none is provided, randomly choose one.
        if behavior_type is None:
            self.behavior_type = EnemyDrone.class_rng.choice([
                "direct", "zigzag", "zigzag_damped", "zigzag_unstable", "zigzag_variable_period",
                "spiral", "spiral_bounce", "spiral_oscillatory",
                "alternating", "bounce_approach", "circle_wait_advance"
            ])
        else:
            self.behavior_type = behavior_type
            
        if behavior_type == "joystick":
            self._init_joystick()

        if not self.behavior_type in ["formation", "biformation", "focal-direct", "debug", "u-debug", "joystick"]:
            self.start_delay = EnemyDrone.class_rng.randint(0, 100)
        else:
            self.start_delay = 0
            
        self.fixed = fixed
        self.trajectory: List[pygame.math.Vector2] = []
        
        if self.behavior_type == "joystick":
            desired_width = int(SIM_WIDTH * 0.02)
            aspect_ratio = self.original_drone_image_joystick.get_height() / self.original_drone_image_joystick.get_width()
            desired_height = int(desired_width * aspect_ratio)
            self.drone_image = pygame.transform.scale(self.original_drone_image_joystick, (desired_width, desired_height))
            
        else:
            desired_width = int(SIM_WIDTH * 0.02)
            aspect_ratio = self.original_drone_image.get_height() / self.original_drone_image.get_width()
            desired_height = int(desired_width * aspect_ratio)
            self.drone_image = pygame.transform.scale(self.original_drone_image, (desired_width, desired_height))
        
        # New attributes for detection and aggressiveness
        self.detector: Optional[pygame.math.Vector2] = None  # Indicates detection by friendly drones
        self.escape_steps: int = ESCAPE_STEPS                # Number of escape steps
        self.escape_steps_count: int = 0                     # Counter for escape steps taken
        self.desperate_attack: bool = False                  # Indicates if drone is in desperate attack mode
            
    @classmethod
    def set_class_seed(cls, seed=None):            
        cls.class_seed = seed if seed is not None else random.randint(0, 10000000)
        cls.class_rng = random.Random(cls.class_seed)
        cls.class_np_rng = np.random.RandomState(cls.class_seed)
        
    def _init_joystick(self):
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Joystick '{self.joystick.get_name()}' conectado.")

    # -----------------------------------------------------------------------------
    # Random Border Position
    # -----------------------------------------------------------------------------
    def random_border_position(self) -> pygame.math.Vector2:
        """
        Generate a random starting position along one of the simulation area's borders.

        Returns:
            pygame.math.Vector2: Random border position.
        """
        side = EnemyDrone.class_rng.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top':
            return pygame.math.Vector2(EnemyDrone.class_rng.uniform(0, SIM_WIDTH), 0)
        elif side == 'bottom':
            return pygame.math.Vector2(EnemyDrone.class_rng.uniform(0, SIM_WIDTH), SIM_HEIGHT)
        elif side == 'left':
            return pygame.math.Vector2(0, EnemyDrone.class_rng.uniform(0, SIM_HEIGHT))
        elif side == 'right':
            return pygame.math.Vector2(SIM_WIDTH, EnemyDrone.class_rng.uniform(0, SIM_HEIGHT))

    # -----------------------------------------------------------------------------
    # Unique ID Assignment
    # -----------------------------------------------------------------------------
    def assign_id(self) -> int:
        """
        Assign a unique identifier to this drone.

        Returns:
            int: The unique drone ID.
        """
        current_id = self.__class__.enemy_id_counter
        EnemyDrone.enemy_id_counter += 1
        return current_id
    

    # -----------------------------------------------------------------------------
    # Update Method
    # -----------------------------------------------------------------------------
    def update(self, detector: Optional[pygame.math.Vector2] = None) -> None: 
        """
        Update the drone's position based on its behavior and velocity while ensuring
        it remains within simulation bounds. The drone stops moving when it reaches the interest point.
        
        Args:
            detector (Optional[pygame.math.Vector2]): Position of a detecting friendly drone.
        """
        self.detector = detector
        self.timer += 1

        # Wait for the start delay before moving.
        if self.timer < self.start_delay:
            return

        # Snap to interest point if very close.
        if (self.interest_point_center - self.pos).length() < 1:
            self.pos = self.interest_point_center.copy()
            self.vel = pygame.math.Vector2(0, 0)
            return

        distance_to_interest = self.pos.distance_to(self.interest_point_center)
        if distance_to_interest > INITIAL_DISTANCE:
            self.aggressiveness = max(INITIAL_AGGRESSIVENESS,
                                      1 + (distance_to_interest - INITIAL_DISTANCE) * ((INITIAL_AGGRESSIVENESS - 1) / (EXTERNAL_RADIUS - INITIAL_DISTANCE)))
        else:
            self.aggressiveness = 1
        
        
        # Update velocity based on the selected behavior.
        self.apply_behavior()
        self.pos += self.vel
        
        # Keep the drone within simulation bounds.
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
            
        if len(self.trajectory) > 300:
            self.trajectory.pop(0)
        self.trajectory.append(self.pos.copy())

    # -----------------------------------------------------------------------------
    # Behavior Application
    # -----------------------------------------------------------------------------
    def apply_behavior(self) -> None:
        """
        Update the drone's velocity according to its chosen behavior,
        blending deterministic and stochastic elements.
        """
        # Desperate attack: move directly toward interest point.
        if self.desperate_attack:
            target_vector = self.interest_point_center - self.pos
            self.vel = target_vector.normalize() * ENEMY_SPEED if target_vector.length() > 0 else pygame.math.Vector2(0, 0)
            self.info = "DESPERATE ATTACK"
            return

        # Escape behavior: move away from detector or interest point.
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

        # Aggressiveness: if detected, decide whether to attack or escape.
        if self.detector and self.behavior_type != 'joystick':
            if EnemyDrone.class_np_rng.rand() < self.aggressiveness:
                self.desperate_attack = True
                self.info = "DESPERATE ATTACK"
            else:
                self.escape_steps_count += 1
                return

        # Default behavior: compute vector toward the interest point.
        target_vector = self.interest_point_center - self.pos
        if target_vector.length() == 0:
            return  # Avoid division by zero
        
        self.info = f"aggr.: {self.aggressiveness:.2f}"

        # ---------------------- Behavior Implementations ----------------------
        if self.behavior_type == "joystick":
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

        elif self.behavior_type == "direct":
            # Move directly toward the interest point.
            self.vel = target_vector.normalize() * ENEMY_SPEED

        elif self.behavior_type == "zigzag":
            # Approach target with sinusoidal oscillations.
            base_direction = target_vector.normalize()
            self.phase += 0.1
            perp = pygame.math.Vector2(-base_direction.y, base_direction.x)
            amplitude = 2
            offset = math.sin(self.phase) * amplitude
            direction = (base_direction + perp * offset).normalize()
            self.vel = direction * ENEMY_SPEED

        elif self.behavior_type == "zigzag_damped":
            # Zigzag with decaying amplitude.
            base_direction = target_vector.normalize()
            self.phase += 0.1
            decay_factor = math.exp(-0.01 * self.timer)
            amplitude = 5.0 * decay_factor
            perp = pygame.math.Vector2(-base_direction.y, base_direction.x)
            offset = math.sin(self.phase) * amplitude
            direction = (base_direction + perp * offset).normalize()
            self.vel = direction * ENEMY_SPEED

        elif self.behavior_type == "zigzag_unstable":
            # Zigzag with random amplitude and frequency.
            base_direction = target_vector.normalize()
            random_amplitude = EnemyDrone.class_rng.uniform(1.0, 10.0)
            random_frequency = EnemyDrone.class_rng.uniform(0.5, 2.0)
            offset = math.sin(random_frequency * self.timer) * random_amplitude
            perp = pygame.math.Vector2(-base_direction.y, base_direction.x)
            direction = (base_direction + perp * offset).normalize()
            self.vel = direction * ENEMY_SPEED

        elif self.behavior_type == "zigzag_variable_period":
            # Zigzag with variable oscillation frequency.
            base_direction = target_vector.normalize()
            self.phase += 0.1
            base_frequency = 1.0
            frequency_variation = 0.5 * math.sin(0.05 * self.timer)
            effective_frequency = base_frequency + frequency_variation
            amplitude = 6.0
            perp = pygame.math.Vector2(-base_direction.y, base_direction.x)
            offset = math.sin(effective_frequency * self.phase) * amplitude
            direction = (base_direction + perp * offset).normalize()
            self.vel = direction * ENEMY_SPEED

        elif self.behavior_type == "spiral":
            # Spiral toward the interest point.
            r_vector = self.pos - self.interest_point_center
            current_distance = r_vector.length()
            angle = math.atan2(r_vector.y, r_vector.x) + 0.1
            new_radius = current_distance * 0.95
            new_offset = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * new_radius
            desired_pos = self.interest_point_center + new_offset
            direction = (desired_pos - self.pos)
            self.vel = direction.normalize() * ENEMY_SPEED if direction.length() != 0 else pygame.math.Vector2(0, 0)

        elif self.behavior_type == "alternating":
            # Alternate between straight approach and perpendicular orbit.
            cycle_length = 100
            half_cycle = cycle_length / 2
            phase_in_cycle = self.timer % cycle_length
            target_direction = target_vector.normalize()
            if phase_in_cycle < half_cycle:
                direction = target_direction
            else:
                direction = pygame.math.Vector2(-target_direction.y, target_direction.x)
                noise_angle = EnemyDrone.class_rng.uniform(-0.1, 0.1)
                direction = direction.rotate_rad(noise_angle).normalize()
            self.vel = direction * ENEMY_SPEED

        elif self.behavior_type == "bounce_approach":
            # Alternate between approaching and retreating with a deviation.
            if not hasattr(self, 'state'):
                self.state = "approach"
                self.state_timer = 0
                self.approach_duration = 60
                self.retreat_duration = 30
                self.deviation_angle = 0
            if self.state == "approach":
                direction = (self.interest_point_center - self.pos).normalize()
                self.state_timer += 1
                if self.state_timer >= self.approach_duration:
                    self.state = "retreat"
                    self.state_timer = 0
                    self.deviation_angle = EnemyDrone.class_rng.uniform(20, 50)
                    if EnemyDrone.class_rng.random() < 0.5:
                        self.deviation_angle = -self.deviation_angle
                self.vel = direction * ENEMY_SPEED
            elif self.state == "retreat":
                direction = (-(self.interest_point_center - self.pos)).normalize()
                direction = direction.rotate(self.deviation_angle)
                self.state_timer += 1
                if self.state_timer >= self.retreat_duration:
                    self.state = "approach"
                    self.state_timer = 0
                self.vel = direction * ENEMY_SPEED

        elif self.behavior_type == "circle_wait_advance":
            # Alternate between waiting (circular motion) and advancing (direct approach).
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
            elif self.cw_state == "advance":
                direction = (self.interest_point_center - self.pos).normalize()
                offset_angle = EnemyDrone.class_rng.uniform(-5, 5)
                direction = direction.rotate(offset_angle)
                self.vel = direction * ENEMY_SPEED
                self.cw_timer += 1
                if self.cw_timer >= self.advance_duration:
                    self.cw_state = "wait"
                    self.cw_timer = 0

        elif self.behavior_type == "spiral_bounce":
            # Combine spiral movement with a bounce approach.
            if not hasattr(self, 'sb_state'):
                self.sb_state = "spiral"
                self.sb_timer = 0
                self.spiral_duration = 60
                self.bounce_duration = 30
                self.bounce_deviation = 0
            if self.sb_state == "spiral":
                r_vector = self.pos - self.interest_point_center
                current_distance = r_vector.length()
                angle = math.atan2(r_vector.y, r_vector.x) + 0.1
                new_radius = current_distance * 0.9
                new_offset = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * new_radius
                desired_pos = self.interest_point_center + new_offset
                direction = (desired_pos - self.pos)
                if direction.length() != 0:
                    direction = direction.normalize()
                self.sb_timer += 1
                if self.sb_timer >= self.spiral_duration:
                    self.sb_state = "bounce"
                    self.sb_timer = 0
                    self.bounce_deviation = EnemyDrone.class_rng.uniform(20, 50)
                    if EnemyDrone.class_rng.random() < 0.5:
                        self.bounce_deviation = -self.bounce_deviation
                self.vel = direction * ENEMY_SPEED
            elif self.sb_state == "bounce":
                direction = (-(self.interest_point_center - self.pos)).normalize()
                direction = direction.rotate(self.bounce_deviation)
                self.sb_timer += 1
                if self.sb_timer >= self.bounce_duration:
                    self.sb_state = "spiral"
                    self.sb_timer = 0
                self.vel = direction * ENEMY_SPEED

        elif self.behavior_type == "spiral_oscillatory":
            # Spiral movement combined with oscillatory perturbations.
            r_vector = self.pos - self.interest_point_center
            current_distance = r_vector.length()
            angle = math.atan2(r_vector.y, r_vector.x) + 0.1
            new_radius = current_distance * 0.95
            new_offset = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * new_radius
            desired_pos = self.interest_point_center + new_offset
            direction = (desired_pos - self.pos)
            if direction.length() != 0:
                direction = direction.normalize()
            base_direction = direction
            self.phase += 0.05
            perp = pygame.math.Vector2(-base_direction.y, base_direction.x)
            amplitude = 1.0
            offset = math.sin(self.phase) * amplitude
            new_direction = (base_direction + perp * offset).normalize()
            self.vel = new_direction * ENEMY_SPEED
            
        elif self.behavior_type == "formation":
            if not hasattr(self, 'formation_id'):
                self.formation_id = self.drone_id % ENEMY_COUNT
                FRONT_FORMATION = 5
                row = self.formation_id // FRONT_FORMATION
                column = self.formation_id % FRONT_FORMATION - FRONT_FORMATION // 2
                
                self.pos = pygame.math.Vector2(SIM_WIDTH, SIM_HEIGHT // 2 + column * 60)
                self.start_delay = 40 * row

            # self.behavior_type = "direct"
            if self.pos.distance_to(self.interest_point_center) > INITIAL_DISTANCE:
                self.vel = pygame.math.Vector2(-1, 0)
                self.info = f"#{self.formation_id}"
                
            else:
                self.behavior_type = "direct"
                
        elif self.behavior_type == "biformation":
            if not hasattr(self, 'formation_id'):
                # Atribui um identificador de formação baseado no drone_id e no total de inimigos
                self.formation_id = self.drone_id % ENEMY_COUNT
                FRONT_FORMATION = 5

                # Se formation_id for menor que metade do total, o drone inicia pela direita;
                # caso contrário, inicia pela esquerda.
                if self.formation_id < ENEMY_COUNT // 2:
                    # Grupo da direita:
                    formation_index = self.formation_id
                    row = formation_index // FRONT_FORMATION
                    column = formation_index % FRONT_FORMATION - FRONT_FORMATION // 2
                    self.pos = pygame.math.Vector2(SIM_WIDTH, SIM_HEIGHT // 2 + column * 60)
                    self.vel = pygame.math.Vector2(-1, 0)
                    self.start_delay = 40 * row
                else:
                    # Grupo da esquerda:
                    formation_index = self.formation_id - (ENEMY_COUNT // 2)
                    row = formation_index // FRONT_FORMATION
                    column = formation_index % FRONT_FORMATION - FRONT_FORMATION // 2
                    self.pos = pygame.math.Vector2(0, SIM_HEIGHT // 2 + column * 60)
                    self.vel = pygame.math.Vector2(1, 0)
                    self.start_delay = 40 * row

            # Enquanto não alcançam o ponto de interesse, os drones mantêm o movimento inicial.
            if self.pos.distance_to(self.interest_point_center) > INITIAL_DISTANCE:
                self.info = f"#{self.formation_id}"
            else:
                self.behavior_type = "direct"
                
        elif self.behavior_type == "focal-direct":
            # Atribui um identificador de formação baseado no drone_id e no total de inimigos
            formation_id = self.drone_id % ENEMY_COUNT
            self.pos = pygame.math.Vector2(SIM_WIDTH, 0)
            self.start_delay = 20 * formation_id
            self.behavior_type = "direct"
                    
        elif self.behavior_type == "debug":
            # Debug behavior: circular movement clockwise around interest point at original distance
            self.p = (0.5 * SIM_WIDTH, 0.5 * SIM_HEIGHT)
            
            # Initialize debug parameters if not already set
            if not hasattr(self, 'debug_orbit_radius'):
                # Get center coordinates (handle both Vector2 and tuple)
                center = self.interest_point_center
                center_x = center.x if hasattr(center, 'x') else center[0]
                center_y = center.y if hasattr(center, 'y') else center[1]
                center_vec = pygame.math.Vector2(center_x, center_y)
                
                # Calculate and store the original distance from center
                self.debug_orbit_radius = self.pos.distance_to(center_vec)
                
                # Angular speed for clockwise rotation (negative for clockwise)
                self.debug_angular_speed = 0.02  # Negative for clockwise
            
            # Get center coordinates
            center = self.interest_point_center
            center_x = center.x if hasattr(center, 'x') else center[0]
            center_y = center.y if hasattr(center, 'y') else center[1]
            
            # Calculate radius vector from center to drone
            radius_vector = self.pos - pygame.math.Vector2(center_x, center_y)
            
            # Get tangent vector (perpendicular to radius) for clockwise motion
            # Rotating radius vector -90 degrees gives clockwise tangent
            tangent_vector = pygame.math.Vector2(radius_vector.y, -radius_vector.x)
            
            # Normalize to get unit vector and apply speed
            if tangent_vector.length() > 0:
                direction = tangent_vector.normalize()
                self.vel = direction * ENEMY_SPEED
            else:
                self.vel = pygame.math.Vector2(0, 0)
            
            self.info = f"ORBIT"
                
        elif self.behavior_type == "u-debug":
            # U-debug behavior: move forward, then perpendicular, and finally reverse.
            if not hasattr(self, 'u_debug_phase'):
                self.u_debug_phase = 0  # 0: moving forward, 1: moving perpendicular, 2: moving backward
                self.u_debug_timer = 0
                self.forward_steps = 300  # frames for forward/backward movement
                self.perp_steps = 40      # frames for perpendicular movement

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
                elif self.u_debug_phase == 2:
                    self.vel = (-target_direction) * ENEMY_SPEED
                    self.u_debug_timer += 1
                    if self.u_debug_timer >= self.forward_steps:
                        self.vel = pygame.math.Vector2(0, 0)
            else:
                self.vel = pygame.math.Vector2(0, 0)

        else:
            # Default behavior: move straight toward the interest point.
            self.vel = target_vector.normalize() * ENEMY_SPEED

    # -----------------------------------------------------------------------------
    # Draw Method
    # -----------------------------------------------------------------------------
    def draw(self, surface: pygame.Surface, show_detection: bool = True, show_trajectory: bool = False, show_debug: bool = False) -> None:
        """
        Draw the enemy drone on the given surface.

        Args:
            surface (pygame.Surface): Surface on which to draw the drone.
            show_detection (bool): If True, displays the drone's detection range.
            show_trajectory (bool): If True, displays the drone's trajectory.
        """
        # Draw trajectory if enabled
        if show_trajectory and len(self.trajectory) > 1:
            traj_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            decay_rate = 0.05  # Constant decay rate for trajectory fading
            n = len(self.trajectory)
            for i in range(n - 1):
                d = n - 1 - i  # Frames elapsed since this segment
                alpha = int(255 * math.exp(-decay_rate * d))
                min_alpha = 30  # Minimum alpha for visibility
                alpha = max(alpha, min_alpha)
                color_with_alpha = self.color + (alpha,)
                start_pos = (int(self.trajectory[i].x), int(self.trajectory[i].y))
                end_pos = (int(self.trajectory[i+1].x), int(self.trajectory[i+1].y))
                pygame.draw.line(traj_surf, color_with_alpha, start_pos, end_pos, 2)
            surface.blit(traj_surf, (0, 0))
        
        # Draw the drone image at its current position.
        image_rect = self.drone_image.get_rect(center=(int(self.pos.x), int(self.pos.y)))
        surface.blit(self.drone_image, image_rect)
        
        # Optionally draw detection range as a dashed circle.
        if show_detection:
            draw_dashed_circle(surface, (self.color[0], self.color[1], self.color[2], 64), (int(self.pos.x), int(self.pos.y)),
                               ENEMY_DETECTION_RANGE, dash_length=5, space_length=5, width=1)
        
        # Render the drone's ID with transparency.
        font = pygame.font.SysFont(FONT_FAMILY, 10)
        label = font.render(f"ID: E{self.drone_id}", True, (255, 255, 255))
        label.set_alpha(128)
        surface.blit(label, (int(self.pos.x) + 20, int(self.pos.y) - 20))
        
        if show_debug:
            len_info = len(self.info)
            debug_label = font.render(self.info, True, (255, 215, 0))
            surface.blit(debug_label, (int(self.pos.x) - 3.5 * len_info, int(self.pos.y) + 25))