from pathlib import Path

# # Agora pode importar normalmente
import DroneSwarm2D
from DroneSwarm2D.core import settings

# Ensure HOLD_SPREAD is set to True for compatibility 
settings.HOLD_SPREAD = True 

import json
from datetime import datetime
import numpy as np
import pygame
from DroneSwarm2D.core.utils import pos_to_cell, intercept_direction, can_intercept, load_best_model
from DroneSwarm2D.behaviorsDefault import BaseBehavior, BehaviorType

class FriendCommonBehaviorAI(BaseBehavior):
    def __init__(self, friend_activation_threshold_position: float = 0.7,
                 enemy_activation_threshold_position: float = 0.4,
                 save_joystick_data: bool = False,
                 buffer_size: int = 1000,
                 joystick_data_path: str = "./src/imitation_learning/data/joystick_data"):
        
        super().__init__(behavior_type=BehaviorType.AI)
        self.friend_activation_threshold_position = friend_activation_threshold_position
        self.enemy_activation_threshold_position = enemy_activation_threshold_position
        self.model = load_best_model("./src/imitation_learning/models/", r"val_loss=(\d+\.\d+)")
        
        # Configuração para salvar dados do joystick
        self.save_joystick_data = save_joystick_data
        self.joystick_data_path = Path(joystick_data_path)
        self.joystick_buffer = []
        self.buffer_size = buffer_size  # Salva a cada 1000 amostras
        
        if settings.JOYSTICK == "Friend":
            self._init_joystick()
        
        if self.save_joystick_data:
            self.joystick_data_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_session_file = self.joystick_data_path / f"joystick_session_{timestamp}.jsonl"
            print(f"💾 Salvando dados de joystick em: {self.current_session_file}")
            
    def apply(self, state, joystick_controlled: bool = False) -> tuple:          
        # Extração e preparação do estado
        pos = np.squeeze(state['pos'])
        pos = pygame.math.Vector2(pos[0], pos[1])
        friend_intensity = np.squeeze(state['friend_intensity'])
        enemy_intensity = np.squeeze(state['enemy_intensity'])
        friend_direction = np.squeeze(state['friend_direction'])
        enemy_direction = np.squeeze(state['enemy_direction'])
                
        enemy_targets = []
        # Identifica células da matriz de inimigos com intensidade acima do limiar
        for cell, intensity in np.ndenumerate(enemy_intensity):
            if intensity < self.enemy_activation_threshold_position:
                continue
            
            target_pos = pygame.math.Vector2((cell[0] + 0.5) * settings.CELL_SIZE,
                                            (cell[1] + 0.5) * settings.CELL_SIZE)
            distance_to_interest = target_pos.distance_to(settings.INTEREST_POINT_CENTER)
            
            enemy_targets.append((cell, target_pos, distance_to_interest))
            
            
        # Obtém a célula correspondente à posição do drone self
        my_cell = pos_to_cell(pos)
        my_cell_center = pygame.math.Vector2((my_cell[0] + 0.5) * settings.CELL_SIZE,
                                            (my_cell[1] + 0.5) * settings.CELL_SIZE)
        
        # Obtém os candidatos amigos a partir da matriz friend_intensity.
        # Cada candidato é identificado pela célula em que há uma detecção ativa.
        friend_candidates = []
        for cell, intensity in np.ndenumerate(friend_intensity):
            if intensity >= self.friend_activation_threshold_position:
                candidate_pos = pygame.math.Vector2((cell[0] + 0.5) * settings.CELL_SIZE,
                                                    (cell[1] + 0.5) * settings.CELL_SIZE)
                friend_candidates.append((cell, candidate_pos))
                
        # Inclui a própria célula self se não houver detecção ativa (para diferenciá-lo)
        if not any(cell == my_cell for cell, pos_candidate in friend_candidates):
            friend_candidates.append((my_cell, my_cell_center))
                
        if not enemy_targets:
            # Se nenhum alvo foi atribuído a self, mantém a posição.
            friends_hold = [(cell, candidate_pos) for (cell, candidate_pos) in friend_candidates]# if 1.1 * settings.INITIAL_DISTANCE - candidate_pos.distance_to(settings.INTEREST_POINT_CENTER) > 0]
            
            return self._hold_position(pos, friend_intensity, enemy_intensity, enemy_direction, friends_hold, joystick_controlled)
        
        # Ordena os alvos pelo quão próximos estão do ponto de interesse
        enemy_targets.sort(key=lambda t: t[2])
        
        engagement_assignment = {}
        assigned_friend_cells = set()
        
        # Para cada alvo inimigo, em ordem, atribui o candidato amigo mais próximo que ainda não foi designado.
        for cell, target_pos, _ in enemy_targets:
            sorted_candidates = sorted(friend_candidates, key=lambda x: x[1].distance_to(target_pos))
            enemy_dir_vec = pygame.math.Vector2(enemy_direction[cell][0], enemy_direction[cell][1])  * settings.ENEMY_SPEED
            
            for candidate in sorted_candidates:
                candidate_cell, candidate_pos = candidate
                
                # if candidate_cell not in assigned_friend_cells and \
                # can_intercept(candidate_pos, settings.FRIEND_SPEED, target_pos, enemy_dir_vec, settings.INTEREST_POINT_CENTER):
                
                if candidate_cell not in assigned_friend_cells: ### DEBUG ###
                    
                    engagement_assignment[tuple(target_pos)] = candidate_cell
                    assigned_friend_cells.add(candidate_cell)
                    break  # Avança para o próximo alvo
                
        # friend_candidates é uma lista de tuplas (cell, candidate_pos) derivada da matriz friend_intensity
                
        # Verifica se algum dos alvos designados possui self como candidato
        engaged_enemies = []
        for cell, target_pos, _ in enemy_targets:
            if engagement_assignment.get(tuple(target_pos)) == my_cell:
                distance = my_cell_center.distance_to(target_pos)
                engaged_enemies.append((distance, cell, target_pos))
        
        if engaged_enemies:
            engaged_enemies.sort(key=lambda t: t[0])
            _, chosen_cell, chosen_target_pos = engaged_enemies[0]
            vel = intercept_direction(pos, settings.FRIEND_SPEED, chosen_target_pos, enemy_direction[chosen_cell])
            
            if vel.length() > 0:
                info = ("PURSUING", None, None, None)
                return info, vel
            else:
                info = ("ERROR PURSUING", None, None, None)
                return info, pygame.math.Vector2(0, 0)
        
        # Se nenhum alvo foi atribuído a self, mantém a posição.
        friends_hold = [(cell, candidate_pos) for (cell, candidate_pos) in friend_candidates
            if cell not in assigned_friend_cells] # and 1.1 * settings.INITIAL_DISTANCE - candidate_pos.distance_to(settings.INTEREST_POINT_CENTER) > 0]

        return self._hold_position(pos, friend_intensity, enemy_intensity, enemy_direction, friends_hold, joystick_controlled)

    def _hold_position(self, pos, friend_intensity, enemy_intensity, enemy_direction, friends_hold=None,
                    joystick_controlled = False, activation_threshold_position: float = 1, enemy_threshold: float = 0.4) -> tuple:

        # Caso 1: Se o drone estiver muito próximo do ponto de interesse, permanece parado.
        if pos.distance_to(settings.INTEREST_POINT_CENTER) < settings.EPSILON:
            info = ("HOLD - WAIT", None, None, friends_hold)
            return info, pygame.math.Vector2(0, 0)
        
        # Constrói a grade dos centros das células com base em friend_intensity.
        grid_x = np.linspace(settings.CELL_SIZE/2, settings.SIM_WIDTH - settings.CELL_SIZE/2, settings.GRID_WIDTH)
        grid_y = np.linspace(settings.CELL_SIZE/2, settings.SIM_HEIGHT - settings.CELL_SIZE/2, settings.GRID_HEIGHT)
        X, Y = np.meshgrid(grid_x, grid_y, indexing='ij')
        distance_matrix = np.sqrt((X - pos.x)**2 + (Y - pos.y)**2)
        comm_mask = distance_matrix < settings.COMMUNICATION_RANGE
        active_friend_count = np.sum((friend_intensity >= activation_threshold_position) & comm_mask)
        
        # Caso 2: Se o drone não se comunica, retorna para a distância INITIAL_DISTANCE do PI.
        if settings.MIN_COMMUNICATION_HOLD == 0 and pos.distance_to(settings.INTEREST_POINT_CENTER) > settings.INITIAL_DISTANCE:
            info = ("HOLD - RETURN", None, None, friends_hold)
            direction = (settings.INTEREST_POINT_CENTER - pos).normalize()
            return info, direction
            
        # Caso 3: Se conectado a pelo menos MIN_COMMUNICATION_HOLD amigos, verifica oportunidade defensiva.
        elif active_friend_count >= settings.MIN_COMMUNICATION_HOLD:
            candidate_intercepts = []
            for cell, intensity in np.ndenumerate(enemy_intensity):
                if intensity < enemy_threshold:
                    continue
                # Centro da célula correspondente
                cell_center = pygame.math.Vector2((cell[0] + 0.5) * settings.CELL_SIZE,
                                                    (cell[1] + 0.5) * settings.CELL_SIZE)
                # Vetor que vai do centro da célula até o ponto de interesse
                vec_to_interest = settings.INTEREST_POINT_CENTER - cell_center
                if vec_to_interest.length() == 0:
                    continue
                vec_to_interest = vec_to_interest.normalize()
                # Vetor de direção detectado para o inimigo nesta célula
                enemy_dir = pygame.math.Vector2(*enemy_direction[cell])
                if enemy_dir.length() == 0:
                    continue
                enemy_dir = enemy_dir.normalize()
                if enemy_dir.dot(vec_to_interest) >= 0.9:
                    candidate_intercepts.append((cell_center.distance_to(settings.INTEREST_POINT_CENTER),
                                                cell_center, enemy_dir))
                    
            if candidate_intercepts:
                candidate_intercepts.sort(key=lambda t: t[0])
                _, chosen_cell_center, enemy_dir = candidate_intercepts[0]
                # Calcula a projeção da posição do drone sobre a reta que passa por chosen_cell_center com direção enemy_dir.
                s = (pos - chosen_cell_center).dot(enemy_dir)
                projection_point = chosen_cell_center + s * enemy_dir
                                
                feasible = projection_point.distance_to(settings.INTEREST_POINT_CENTER) < cell_center.distance_to(settings.INTEREST_POINT_CENTER)
                    
                if pos.distance_to(projection_point) <= settings.EPSILON and feasible:
                    info = ("HOLD - INTCPT", None, None, friends_hold)
                    direction = pygame.math.Vector2(0, 0)
                    return info, direction
                
                if chosen_cell_center.distance_to(projection_point) > chosen_cell_center.distance_to(settings.INTEREST_POINT_CENTER):
                    defensive_point = settings.INTEREST_POINT_CENTER
                else:
                    defensive_point = projection_point
                
                # Se o drone estiver muito distante dessa projeção, permanece em hold.
                if (pos - defensive_point).length() < settings.THRESHOLD_PROJECTION and feasible:
                    info = ("GO HOLD INTCPT", defensive_point, chosen_cell_center, friends_hold)
                    direction = (defensive_point - pos).normalize()
                    
                    return info, direction
            
            if settings.HOLD_SPREAD and len(friends_hold) > 2:
            # if len(friends_hold) > 2 and pos.distance_to(settings.INTEREST_POINT_CENTER) <= 1.1 * settings.INITIAL_DISTANCE:    
                # Se o drone estiver em hold e houver pelo menos dois amigos ativos, verifica a possibilidade de espalhamento.
                matrix_friends_hold = np.zeros((settings.GRID_WIDTH, settings.GRID_HEIGHT), dtype=np.float32)
                _pos = np.array([[pos.x, pos.y]], dtype=np.float32)
                
                for cell, _ in friends_hold:
                    matrix_friends_hold[cell] = 1.0
                    
                if joystick_controlled:
                    return self._joystick(_pos, matrix_friends_hold)
                
                if settings.JOYSTICK != "Friend":
                    matrix_friends_with_batch = np.expand_dims(matrix_friends_hold, axis=0)
                    
                    repulsion = np.squeeze(self.model.predict([_pos, matrix_friends_with_batch], verbose=0))
                    repulsion = pygame.math.Vector2(repulsion[0], repulsion[1])

                    radial = pos - settings.INTEREST_POINT_CENTER
                    if radial.length() == 0:
                        radial = pygame.math.Vector2(1, 0)
                    else:
                        radial = radial.normalize()
                    # Projeção da força repulsiva na direção radial.
                    radial_component = repulsion.dot(radial) * radial
                    # Componente tangencial: subtrai a parte radial.
                    direction = repulsion - radial_component
                    
                    if direction.length() > 1:
                        direction = direction.normalize()
                        
                    vel = direction * settings.FRIEND_SPEED
                    info = ("HOLD - AI", None, None, friends_hold)
                        
                    return info, vel

            info = ("HOLD - WAIT", None, None, friends_hold)
            return info, pygame.math.Vector2(0, 0)
        
        # Caso 3: Se não estiver conectado a pelo menos dois amigos, retorna a direção para o PI.
        info = ("HOLD - NO ENOUGH COMM", None, None, friends_hold)
        direction = (settings.INTEREST_POINT_CENTER - pos).normalize()
        return info, direction

    def _joystick(self, pos, matrix_friends_hold) -> tuple:
        if self.joystick is not None:
            x_axis = self.joystick.get_axis(0) 
            y_axis = self.joystick.get_axis(1)
            
            direction = pygame.math.Vector2(x_axis, y_axis)
            
            if direction.length() > 0.1:
                direction = direction.normalize()
            else:
                direction = pygame.math.Vector2(0, 0)

            vel = direction * settings.FRIEND_SPEED
            info = ("HOLD - JOYSTICK", None, None, None)
            
            # Salvar dados se ativado
            if self.save_joystick_data:
                self._save_joystick_sample(pos, matrix_friends_hold, vel)
                
        else:
            vel = pygame.math.Vector2(0, 0)
            info = ("HOLD - NO JOYSTICK", None, None, None)
            
        return info, vel
    
    def _save_joystick_sample(self, pos, matrix_friends_hold, velocity):
        """Salva amostra de friends_hold -> velocity"""
        
        # Cria sample em formato serializável
        sample = {
            'pos': pos.tolist(),
            'friends_hold': matrix_friends_hold.tolist(),
            'velocity': [velocity.x, velocity.y]
        }
        
        self.joystick_buffer.append(sample)
        
        # Flush para disco quando buffer enche
        if len(self.joystick_buffer) >= self.buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Salva buffer em disco (formato JSONL - uma linha por sample)"""
        if not self.joystick_buffer:
            return
        
        with open(self.current_session_file, 'a') as f:
            for sample in self.joystick_buffer:
                f.write(json.dumps(sample) + '\n')
        
        print(f"💾 Salvos {len(self.joystick_buffer)} samples de joystick")
        self.joystick_buffer.clear()
    
    def __del__(self):
        """Garante que dados restantes sejam salvos ao destruir objeto"""
        if hasattr(self, 'save_joystick_data') and self.save_joystick_data:
            self._flush_buffer()
            
class FriendBenchmarkBehavior(BaseBehavior):
    """
    Comportamento similar ao FriendCommonBehavior, mas com movimento circular
    aleatório em torno do ponto de interesse durante o estado HOLD.
    
    No estado HOLD - BENCHMARK, os drones circulam alternadamente no sentido
    horário e anti-horário por um número aleatório de steps.
    """
    
    def __init__(self, 
                 friend_activation_threshold_position: float = 0.7,
                 enemy_activation_threshold_position: float = 0.4,
                 n_clockwise: int = 100):
        """
        Args:
            friend_activation_threshold_position: Limiar de intensidade para detecção de amigos.
            enemy_activation_threshold_position: Limiar de intensidade para detecção de inimigos.
            n_clockwise: Número máximo de steps para movimento circular em cada direção.
        """
        super().__init__(behavior_type=BehaviorType.COMMON)
        self.friend_activation_threshold_position = friend_activation_threshold_position
        self.enemy_activation_threshold_position = enemy_activation_threshold_position
        self.n_clockwise = n_clockwise
        
        # Dicionário para armazenar o estado circular de cada drone individualmente
        self._drone_states = {}
        self._rng = np.random.default_rng()

    def _get_drone_state(self, drone_id):
        """Obtém ou inicializa o estado circular de um drone específico."""
        if drone_id not in self._drone_states:
            self._drone_states[drone_id] = {
                'steps_remaining': 0,
                'direction': 1
            }
        return self._drone_states[drone_id]
    
    def _reset_drone_state(self, drone_id):
        """Reseta o estado circular de um drone específico."""
        if drone_id in self._drone_states:
            self._drone_states[drone_id]['steps_remaining'] = 0

    def apply(self, state, joystick_controlled: bool = False) -> tuple:
        
        # Extração e preparação do estado
        pos = np.squeeze(state['pos'])
        pos = pygame.math.Vector2(pos[0], pos[1])
        drone_id = state['drone_id']
        friend_intensity = np.squeeze(state['friend_intensity'])
        enemy_intensity = np.squeeze(state['enemy_intensity'])
        friend_direction = np.squeeze(state['friend_direction'])
        enemy_direction = np.squeeze(state['enemy_direction'])
        
        enemy_targets = []
        # Identifica células da matriz de inimigos com intensidade acima do limiar
        for cell, intensity in np.ndenumerate(enemy_intensity):
            if intensity < self.enemy_activation_threshold_position:
                continue
            
            target_pos = pygame.math.Vector2((cell[0] + 0.5) * settings.CELL_SIZE,
                                            (cell[1] + 0.5) * settings.CELL_SIZE)
            distance_to_interest = target_pos.distance_to(settings.INTEREST_POINT_CENTER)
            
            enemy_targets.append((cell, target_pos, distance_to_interest))
            
        # Obtém a célula correspondente à posição do drone self
        my_cell = pos_to_cell(pos)
        my_cell_center = pygame.math.Vector2((my_cell[0] + 0.5) * settings.CELL_SIZE,
                                            (my_cell[1] + 0.5) * settings.CELL_SIZE)
        
        # Obtém os candidatos amigos a partir da matriz friend_intensity.
        friend_candidates = []
        for cell, intensity in np.ndenumerate(friend_intensity):
            if intensity >= self.friend_activation_threshold_position:
                candidate_pos = pygame.math.Vector2((cell[0] + 0.5) * settings.CELL_SIZE,
                                                    (cell[1] + 0.5) * settings.CELL_SIZE)
                friend_candidates.append((cell, candidate_pos))
                
        # Inclui a própria célula self se não houver detecção ativa
        if not any(cell == my_cell for cell, pos_candidate in friend_candidates):
            friend_candidates.append((my_cell, my_cell_center))
                
        if not enemy_targets:
            # Se nenhum alvo foi detectado, mantém a posição com comportamento circular
            friends_hold = [(cell, candidate_pos) for (cell, candidate_pos) in friend_candidates]
            return self._hold_position(pos, friend_intensity, enemy_intensity, enemy_direction, 
                                      friends_hold, drone_id)
        
        # Ordena os alvos pelo quão próximos estão do ponto de interesse
        enemy_targets.sort(key=lambda t: t[2])
        
        engagement_assignment = {}
        assigned_friend_cells = set()
        
        # Para cada alvo inimigo, em ordem, atribui o candidato amigo mais próximo
        for cell, target_pos, _ in enemy_targets:
            sorted_candidates = sorted(friend_candidates, key=lambda x: x[1].distance_to(target_pos))
            
            for candidate in sorted_candidates:
                candidate_cell, candidate_pos = candidate
                
                if candidate_cell not in assigned_friend_cells:
                    engagement_assignment[tuple(target_pos)] = candidate_cell
                    assigned_friend_cells.add(candidate_cell)
                    break
                
        # Verifica se algum dos alvos designados possui self como candidato
        engaged_enemies = []
        for cell, target_pos, _ in enemy_targets:
            if engagement_assignment.get(tuple(target_pos)) == my_cell:
                distance = my_cell_center.distance_to(target_pos)
                engaged_enemies.append((distance, cell, target_pos))
        
        if engaged_enemies:
            # Reset do estado circular quando sair do HOLD
            self._reset_drone_state(drone_id)
            
            engaged_enemies.sort(key=lambda t: t[0])
            _, chosen_cell, chosen_target_pos = engaged_enemies[0]
            vel = intercept_direction(pos, settings.FRIEND_SPEED, chosen_target_pos, enemy_direction[chosen_cell])
            
            if vel.length() > 0:
                info = ("PURSUING", None, None, None)
                return info, vel
            else:
                info = ("ERROR PURSUING", None, None, None)
                return info, pygame.math.Vector2(0, 0)
        
        # Se nenhum alvo foi atribuído a self, mantém a posição com comportamento circular
        friends_hold = [(cell, candidate_pos) for (cell, candidate_pos) in friend_candidates
            if cell not in assigned_friend_cells]
        
        return self._hold_position(pos, friend_intensity, enemy_intensity, enemy_direction, 
                                   friends_hold, drone_id)

    def _hold_position(self, pos, friend_intensity, enemy_intensity, enemy_direction, 
                      friends_hold=None, drone_id=None,
                      activation_threshold_position: float = 1, enemy_threshold: float = 0.4) -> tuple:
        
        # Obtém o estado específico deste drone
        drone_state = self._get_drone_state(drone_id)
        
        # Caso 1: Se o drone estiver muito próximo do ponto de interesse, permanece parado.
        if pos.distance_to(settings.INTEREST_POINT_CENTER) < settings.EPSILON:
            self._reset_drone_state(drone_id)
            info = ("HOLD - WAIT", None, None, friends_hold)
            return info, pygame.math.Vector2(0, 0)
        
        # Constrói a grade dos centros das células com base em friend_intensity.
        grid_x = np.linspace(settings.CELL_SIZE/2, settings.SIM_WIDTH - settings.CELL_SIZE/2, settings.GRID_WIDTH)
        grid_y = np.linspace(settings.CELL_SIZE/2, settings.SIM_HEIGHT - settings.CELL_SIZE/2, settings.GRID_HEIGHT)
        X, Y = np.meshgrid(grid_x, grid_y, indexing='ij')
        distance_matrix = np.sqrt((X - pos.x)**2 + (Y - pos.y)**2)
        comm_mask = distance_matrix < settings.COMMUNICATION_RANGE
        active_friend_count = np.sum((friend_intensity >= activation_threshold_position) & comm_mask)
        
        # Caso 2: Se o drone não se comunica, retorna para a distância settings.INITIAL_DISTANCE do PI.
        if settings.MIN_COMMUNICATION_HOLD == 0 and pos.distance_to(settings.INTEREST_POINT_CENTER) > settings.INITIAL_DISTANCE:
            self._reset_drone_state(drone_id)
            info = ("HOLD - RETURN", None, None, friends_hold)
            direction = (settings.INTEREST_POINT_CENTER - pos).normalize()
            vel = direction * settings.FRIEND_SPEED
            return info, vel
            
        # Caso 3: Se conectado a pelo menos settings.MIN_COMMUNICATION_HOLD amigos, verifica oportunidade defensiva.
        elif active_friend_count >= settings.MIN_COMMUNICATION_HOLD:
            candidate_intercepts = []
            for cell, intensity in np.ndenumerate(enemy_intensity):
                if intensity < enemy_threshold:
                    continue
                cell_center = pygame.math.Vector2((cell[0] + 0.5) * settings.CELL_SIZE,
                                                    (cell[1] + 0.5) * settings.CELL_SIZE)
                vec_to_interest = settings.INTEREST_POINT_CENTER - cell_center
                if vec_to_interest.length() == 0:
                    continue
                vec_to_interest = vec_to_interest.normalize()
                enemy_dir = pygame.math.Vector2(*enemy_direction[cell])
                if enemy_dir.length() == 0:
                    continue
                enemy_dir = enemy_dir.normalize()
                if enemy_dir.dot(vec_to_interest) >= 0.9:
                    candidate_intercepts.append((cell_center.distance_to(settings.INTEREST_POINT_CENTER),
                                                cell_center, enemy_dir))
                    
            if candidate_intercepts:
                self._reset_drone_state(drone_id)
                
                candidate_intercepts.sort(key=lambda t: t[0])
                _, chosen_cell_center, enemy_dir = candidate_intercepts[0]
                s = (pos - chosen_cell_center).dot(enemy_dir)
                projection_point = chosen_cell_center + s * enemy_dir
                                
                feasible = projection_point.distance_to(settings.INTEREST_POINT_CENTER) < cell_center.distance_to(settings.INTEREST_POINT_CENTER)
                    
                if pos.distance_to(projection_point) <= settings.EPSILON and feasible:
                    info = ("HOLD - INTCPT", None, None, friends_hold)
                    vel = pygame.math.Vector2(0, 0)
                    return info, vel
                
                if chosen_cell_center.distance_to(projection_point) > chosen_cell_center.distance_to(settings.INTEREST_POINT_CENTER):
                    defensive_point = settings.INTEREST_POINT_CENTER
                else:
                    defensive_point = projection_point
                
                if (pos - defensive_point).length() < settings.THRESHOLD_PROJECTION and feasible:
                    info = ("GO HOLD INTCPT", defensive_point, chosen_cell_center, friends_hold)
                    vel = (defensive_point - pos).normalize() * settings.FRIEND_SPEED
                    return info, vel
            
            # Movimento circular em torno do ponto de interesse
            if settings.HOLD_SPREAD and len(friends_hold) > 2:    
                # Verifica se precisa iniciar novo ciclo de movimento circular
                if drone_state['steps_remaining'] <= 0:
                    # Sorteia nova quantidade de steps e direção
                    drone_state['steps_remaining'] = self._rng.integers(0, self.n_clockwise + 1)
                    drone_state['direction'] = self._rng.choice([1, -1])  # 1: horário, -1: anti-horário
                
                # Calcula o vetor tangencial para movimento circular
                radial = pos - settings.INTEREST_POINT_CENTER
                if radial.length() == 0:
                    radial = pygame.math.Vector2(1, 0)
                else:
                    radial = radial.normalize()
                
                # Vetor tangencial perpendicular ao radial
                # Para movimento horário: rotaciona 90° no sentido horário (-90° matemático)
                # Para movimento anti-horário: rotaciona 90° no sentido anti-horário (+90° matemático)
                tangential = pygame.math.Vector2(-radial.y, radial.x) * drone_state['direction']
                
                vel = tangential * settings.FRIEND_SPEED
                
                # Decrementa o contador de steps
                drone_state['steps_remaining'] -= 1
                    
                info = ("HOLD - BENCHMARK", None, None, friends_hold)
                return info, vel

            # Se não houver movimento circular, espera parado
            self._reset_drone_state(drone_id)
            info = ("HOLD - WAIT", None, None, friends_hold)
            return info, pygame.math.Vector2(0, 0)
        
        # Caso 4: Se não estiver conectado a pelo menos MIN_COMMUNICATION_HOLD amigos, retorna ao PI.
        self._reset_drone_state(drone_id)
        info = ("HOLD - NO ENOUGH COMM", None, None, friends_hold)
        direction = (settings.INTEREST_POINT_CENTER - pos).normalize()
        vel = direction * settings.FRIEND_SPEED
        
        return info, vel