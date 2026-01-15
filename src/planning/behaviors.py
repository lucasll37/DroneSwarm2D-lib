import DroneSwarm2D
from DroneSwarm2D.core import settings

import numpy as np
import pygame
from DroneSwarm2D.core.utils import pos_to_cell, intercept_direction, can_intercept
from DroneSwarm2D.behaviorsDefault import BaseBehavior, BehaviorType

# -------------------------------------------------------------------------
# Planning Policy (Class Method)
# -------------------------------------------------------------------------   

class FriendCommonBehavior(BaseBehavior):
    def __init__(self, friend_activation_threshold_position: float = 0.7,
                 enemy_activation_threshold_position: float = 0.4):

        super().__init__(behavior_type=BehaviorType.COMMON)
        self.friend_activation_threshold_position = friend_activation_threshold_position
        self.enemy_activation_threshold_position = enemy_activation_threshold_position

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
            
            return self._hold_position(pos, friend_intensity, enemy_intensity, enemy_direction, friends_hold)
        
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
        
        return self._hold_position(pos, friend_intensity, enemy_intensity, enemy_direction, friends_hold)

    def _hold_position(self, pos, friend_intensity, enemy_intensity, enemy_direction, friends_hold=None,
                    activation_threshold_position: float = 1, enemy_threshold: float = 0.4) -> tuple:
        
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
        
        # Caso 2: Se o drone não se comunica, retorna para a distância settings.INITIAL_DISTANCE do PI.
        if settings.MIN_COMMUNICATION_HOLD == 0 and pos.distance_to(settings.INTEREST_POINT_CENTER) > settings.INITIAL_DISTANCE:
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
                    vel = pygame.math.Vector2(0, 0)
                    return info, vel
                
                if chosen_cell_center.distance_to(projection_point) > chosen_cell_center.distance_to(settings.INTEREST_POINT_CENTER):
                    defensive_point = settings.INTEREST_POINT_CENTER
                else:
                    defensive_point = projection_point
                
                # Se o drone estiver muito distante dessa projeção, permanece em hold.
                if (pos - defensive_point).length() < settings.THRESHOLD_PROJECTION and feasible:
                    info = ("GO HOLD INTCPT", defensive_point, chosen_cell_center, friends_hold)
                    vel = (defensive_point - pos).normalize() * settings.FRIEND_SPEED
                    
                    return info, vel
            
            if settings.HOLD_SPREAD and len(friends_hold) > 2:    
            # if len(friends_hold) > 2 and pos.distance_to(settings.INTEREST_POINT_CENTER) <= 1.1 * settings.INITIAL_DISTANCE:    
                # Se o drone estiver em hold e houver pelo menos dois amigos ativos, verifica a possibilidade de espalhamento.
                # Calcula a força de repulsão entre os amigos.
                repulsion = pygame.math.Vector2(0, 0)
                
                my_cell = pos_to_cell(pos)
                for cell, candidate_pos in friends_hold:
                    # Exclui self (assumindo que sua célula é única)
                    if cell == my_cell:
                        continue
                    
                    delta = pos - candidate_pos
                    repulsion += delta.normalize()
                
                # Para manter a formação em torno do ponto de interesse, extraímos somente a componente
                # tangencial do vetor repulsivo em relação ao vetor radial (do ponto de interesse até pos).
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
                    
                info = ("HOLD - BC", None, None, friends_hold) 
                # info = ("HOLD - SPREAD", None, None, friends_hold) # debug
                    
                return info, vel

            info = ("HOLD - WAIT", None, None, friends_hold)
            return info, pygame.math.Vector2(0, 0)
        
        # Caso 3: Se não estiver conectado a pelo menos dois amigos, retorna a direção para o PI.
        info = ("HOLD - NO ENOUGH COMM", None, None, friends_hold)
        direction = (settings.INTEREST_POINT_CENTER - pos).normalize()
        vel = direction * settings.FRIEND_SPEED
        
        return info, vel

    
class FriendAEWBehavior(BaseBehavior):
    def __init__(self, friend_activation_threshold_position: float = 0.7,
                 enemy_activation_threshold_position: float = 0.4):

        super().__init__(behavior_type=BehaviorType.AEW)
        self.friend_activation_threshold_position = friend_activation_threshold_position
        self.enemy_activation_threshold_position = enemy_activation_threshold_position

        
    def apply(self, state, joystick_controlled: bool = False) -> tuple:
        
        pos = np.squeeze(state['pos'])
        pos = pygame.math.Vector2(pos[0], pos[1])
        friend_intensity = np.squeeze(state['friend_intensity'])
        enemy_intensity = np.squeeze(state['enemy_intensity'])
        friend_direction = np.squeeze(state['friend_direction'])
        enemy_direction = np.squeeze(state['enemy_direction'])
        

        # Compute radial vector from interest point
        r_vec = pos - settings.INTEREST_POINT_CENTER
        current_distance = r_vec.length()
        
        if current_distance == 0:
            r_vec = pygame.math.Vector2(settings.AEW_RANGE, 0)
            current_distance = settings.AEW_RANGE
            
        # Calculate orbit correction
        radial_error = settings.AEW_RANGE - current_distance
        k_radial = 0.05  # Radial correction factor
        radial_correction = k_radial * radial_error * r_vec.normalize()
        
        # Calculate tangential velocity (perpendicular to radial)
        tangent = pygame.math.Vector2(-r_vec.y, r_vec.x).normalize()
        vel = tangent * settings.AEW_SPEED
        info = ("AEW", None, None, None)
        
        return info, vel


class FriendRadarBehavior(BaseBehavior):
    def __init__(self, friend_activation_threshold_position: float = 0.7,
                 enemy_activation_threshold_position: float = 0.4):

        super().__init__(behavior_type=BehaviorType.RADAR)
        self.friend_activation_threshold_position = friend_activation_threshold_position
        self.enemy_activation_threshold_position = enemy_activation_threshold_position
        self.type = BehaviorType.RADAR
        
    def apply(self, state, joystick_controlled: bool = False) -> tuple:
        """
        Apply RADAR behavior - remain stationary.
        """
        info = ("RADAR", None, None, None)
        vel = pygame.math.Vector2(0, 0)
        
        return info, vel