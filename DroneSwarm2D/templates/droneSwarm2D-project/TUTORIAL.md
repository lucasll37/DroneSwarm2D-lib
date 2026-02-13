# DroneSwarm2D - Guia de Uso da Biblioteca


## ⚙️ Configuração do Cenário (`config.json`)

O arquivo `config.json` define todos os parâmetros da simulação. Aqui estão os principais grupos de configuração:

### Configurações de Drones

```json
{
  "FRIEND_COUNT": 8,           // Número de drones amigos (defensivos)
  "ENEMY_COUNT": 8,            // Número de drones inimigos
  "RADAR_COUNT": 0,            // Drones radar estacionários
  "AEW_COUNT": 0,              // Drones AEW (Alerta Aéreo Antecipado)
  "BROKEN_COUNT": 0            // Drones com detecção defeituosa
}
```

### Alcances de Detecção e Comunicação

```json
{
  "FRIEND_DETECTION_RANGE": 250,   // Alcance de detecção dos drones amigos (px)
  "ENEMY_DETECTION_RANGE": 100,    // Alcance de detecção dos drones inimigos (px)
  "COMMUNICATION_RANGE": 250,       // Alcance de comunicação entre drones (px)
  "RADAR_DETECTION_RANGE": 350,    // Alcance de detecção dos radares (px)
  "AEW_DETECTION_RANGE": 200       // Alcance de detecção dos drones AEW (px)
}
```

### Parâmetros do Ambiente

```json
{
  "SIM_WIDTH": "int(FULL_WIDTH * 0.7)",  // Largura da área de simulação
  "SIM_HEIGHT": "FULL_HEIGHT",            // Altura da área de simulação
  "CELL_SIZE": 20,                        // Tamanho das células da grade (px)
  "GRID_WIDTH": "SIM_WIDTH // CELL_SIZE", // Largura da grade em células
  "GRID_HEIGHT": "SIM_HEIGHT // CELL_SIZE" // Altura da grade em células
}
```

### Comportamento e Física

```json
{
  "BASE_SPEED": 2.0,                    // Velocidade base dos drones (px/frame)
  "ENEMY_SPEED": "BASE_SPEED",          // Velocidade dos inimigos
  "FRIEND_SPEED": "BASE_SPEED",         // Velocidade dos amigos
  "DT_STEP": 0.6,                       // Delta de tempo por step (segundos)
  "DECAY_FACTOR": 0.99,                 // Fator de decaimento das matrizes de detecção
  "MESSAGE_LOSS_PROBABILITY": 0.1       // Probabilidade de perda de mensagens
}
```

### Comunicação e Rede

```json
{
  "N_CONNECTIONS": 3,              // Número máximo de conexões simultâneas por drone
  "CICLE_COMM_BY_STEP": 3,        // Ciclos de comunicação por step
  "MIN_COMMUNICATION_HOLD": 3      // Mínimo de conexões para comportamento de hold
}
```

### Ponto de Interesse (Alvo a Defender)

```json
{
  "INTEREST_POINT_CENTER": ["SIM_WIDTH / 2", "SIM_HEIGHT / 2"],
  "INTERNAL_RADIUS": "min(SIM_WIDTH, SIM_HEIGHT) / 10",
  "EXTERNAL_RADIUS": "INTERNAL_RADIUS * 4",
  "INTEREST_POINT_INITIAL_HEALTH": 100,
  "INTEREST_POINT_DAMAGE": "INTEREST_POINT_INITIAL_HEALTH // ENEMY_COUNT"
}
```

### Neutralização e Combate

```json
{
  "NEUTRALIZATION_RANGE": 20,                  // Distância para neutralização
  "NEUTRALIZATION_PROB_FRIEND_ALIVE": 0.5,     // Prob. de apenas amigo sobreviver
  "NEUTRALIZATION_PROB_ENEMY_ALIVE": 0.2,      // Prob. de apenas inimigo sobreviver
  "NEUTRALIZATION_PROB_BOTH_DEAD": "...",      // Prob. de ambos serem destruídos
  "INITIAL_AGGRESSIVENESS": 0.5,               // Agressividade inicial dos inimigos
  "ESCAPE_STEPS": 40                           // Steps de fuga após detecção
}
```

### Zonas Desmilitarizadas (DMZ)

```json
{
  "DMZ": [
    ["SIM_WIDTH * 0.35", "SIM_HEIGHT * 0.30", 60],  // [x, y, raio]
    ["SIM_WIDTH * 0.65", "SIM_HEIGHT * 0.35", 40],
    ["SIM_WIDTH * 0.55", "SIM_HEIGHT * 0.75", 80]
  ]
}
```

### Detecção Passiva e Triangulação

```json
{
  "PASSIVE_DETECTION": true,         // Ativar detecção passiva por triangulação
  "N_LINE_SIGHT_CROSSING": 3,       // Número de linhas de visada para confirmação
  "TRIANGULATION_GRANULARITY": 32    // Granularidade da grade de triangulação
}
```

### Controle por Joystick

```json
{
  "JOYSTICK": "Enemy"  // "Friend", "Enemy" ou "None" para desabilitar
}
```

---

## 🎯 Arquitetura da Simulação

### Sistema de Coordenadas

A simulação utiliza um **sistema de grade 2D** onde:
- Coordenadas diretas (`pos`): posição real dos drones em pixels
- Coordenadas celular (`cell`): posição em termos de célula da grade

```python
# Conversão de posição para célula
from DroneSwarm2D.core.utils import pos_to_cell

pos = pygame.math.Vector2(250.5, 180.7)
cell = pos_to_cell(pos)  # Retorna (12, 9) se CELL_SIZE = 20
```

### Matrizes de Estado

Cada drone mantém **matrizes de detecção** que representam sua percepção do ambiente:

```python
# Estrutura das matrizes (dimensões: GRID_WIDTH x GRID_HEIGHT)
drone.enemy_intensity     # Intensidade de detecção de inimigos (0-1)
drone.enemy_direction     # Vetores de direção dos inimigos (GRID_WIDTH x GRID_HEIGHT x 2)
drone.enemy_timestamp     # Timestamp de última atualização de cada célula

drone.friend_intensity    # Intensidade de detecção de amigos (0-1)
drone.friend_direction    # Vetores de direção dos amigos
drone.friend_timestamp    # Timestamp de última atualização
```

#### Como Funcionam as Matrizes

1. **Detecção Local**: Quando um drone detecta um inimigo, ele atualiza a célula correspondente:
   ```python
   cell = pos_to_cell(enemy.pos)
   drone.enemy_intensity[cell] = 1.0
   drone.enemy_direction[cell] = velocity_vector.normalize()
   drone.enemy_timestamp[cell] = current_time
   ```

2. **Decaimento Temporal**: A cada step, as intensidades decaem exponencialmente:
   ```python
   drone.enemy_intensity *= DECAY_FACTOR  # Ex: 0.99
   ```

3. **Fusão de Informações**: Durante a comunicação, drones mesclam suas matrizes:
   ```python
   # Atualiza apenas células com timestamp mais recente
   update_mask = neighbor.enemy_timestamp > self.enemy_timestamp
   np.putmask(self.enemy_intensity, update_mask, neighbor.enemy_intensity)
   ```

### Detecção Passiva e Triangulação

Quando `PASSIVE_DETECTION` está ativo, os drones podem detectar inimigos mesmo fora de seu alcance direto, através da triangulação de múltiplas linhas de visada:

```python
# Cada drone registra linhas de visada para alvos
drone.passive_detection_matrix  # Grade de alta resolução (GRID_WIDTH*GRANULARITY)

# Quando N_LINE_SIGHT_CROSSING linhas se cruzam em uma região,
# o alvo é considerado detectado
```

### Rede de Comunicação Ad-hoc

Os drones formam uma **rede descentralizada** baseada em proximidade:

```python
# Cada drone mantém conexões com seus N_CONNECTIONS vizinhos mais próximos
for _ in range(CICLE_COMM_BY_STEP):
    for neighbor in drone.neighbors:
        if random() > MESSAGE_LOSS_PROBABILITY:
            drone.merge_enemy_matrix(neighbor)
            drone.merge_friend_matrix(neighbor)
```

---

## 🤖 Sistema de Comportamentos

### Estrutura Base

Todos os comportamentos herdam de `BaseBehavior` e devem implementar o método `apply()`:

```python
from DroneSwarm2D.behaviorsDefault import BaseBehavior, BehaviorType
import pygame
import numpy as np

class MyCustomBehavior(BaseBehavior):
    def __init__(self):
        super().__init__(behavior_type=BehaviorType.COMMON)
    
    def apply(self, state, joystick_controlled: bool = False) -> tuple:
        """
        Args:
            state: Dicionário com informações do drone
            joystick_controlled: Se o drone está sob controle manual
        
        Returns:
            (info, velocity): Tupla com informações de debug e vetor velocidade
        """
        # Extração e preparação do estado
        drone_id = state['drone_id']
        pos = np.squeeze(state['pos'])
        pos = pygame.math.Vector2(pos[0], pos[1])
        friend_intensity = np.squeeze(state['friend_intensity'])
        enemy_intensity = np.squeeze(state['enemy_intensity'])
        friend_direction = np.squeeze(state['friend_direction'])
        enemy_direction = np.squeeze(state['enemy_direction'])
        
        # Suas decisões aqui...
        
        info = ("ESTADO", target_pos, projection, friends_hold)
        direction = pygame.math.Vector2(dx, dy).normalize()  # Exemplo de direção
        velocity = direction * settings.FRIEND_SPEED
        
        return info, velocity
```

### Estado Fornecido ao Comportamento

O dicionário `state` contém:

```python
state = {
    'drone_id': self.drone_id,
    'pos': np.array([[x, y]]),                    # Posição do drone (1, 2)
    'friend_intensity': np.ndarray,               # Matriz (1, GRID_W, GRID_H)
    'enemy_intensity': np.ndarray,                # Matriz (1, GRID_W, GRID_H)
    'friend_direction': np.ndarray,               # Matriz (1, GRID_W, GRID_H, 2)
    'enemy_direction': np.ndarray                 # Matriz (1, GRID_W, GRID_H, 2)
}
```

### Tipos de Comportamento

```python
class BehaviorType(Enum):
    RADAR = "RADAR"      # Radar estacionário
    AEW = "AEW"         # Alerta Aéreo Antecipado (órbita)
    COMMON = "COMMON"   # Comportamento padrão/tático
    AI = "AI"          # Baseado em Inteligência Artificial
```

---

## 🛠️ Implementando Comportamentos Customizados

### Exemplo 1: Comportamento de Perseguição Simples

```python
class SimplePursuitBehavior(BaseBehavior):
    def __init__(self, activation_threshold: float = 0.4):
        super().__init__(behavior_type=BehaviorType.COMMON)
        self.activation_threshold = activation_threshold
    
    def apply(self, state, joystick_controlled: bool = False) -> tuple:
        from DroneSwarm2D.core.settings import CELL_SIZE, FRIEND_SPEED
        
        pos = np.squeeze(state['pos'])
        pos = pygame.math.Vector2(pos[0], pos[1])
        enemy_intensity = np.squeeze(state['enemy_intensity'])
        
        # Encontrar inimigo mais próximo
        enemy_targets = []
        for cell, intensity in np.ndenumerate(enemy_intensity):
            if intensity < self.activation_threshold:
                continue
            
            target_pos = pygame.math.Vector2(
                (cell[0] + 0.5) * CELL_SIZE,
                (cell[1] + 0.5) * CELL_SIZE
            )
            distance = pos.distance_to(target_pos)
            enemy_targets.append((distance, target_pos))
        
        # Perseguir o mais próximo
        if enemy_targets:
            enemy_targets.sort()
            _, target = enemy_targets[0]
            direction = (target - pos).normalize()
            vel = direction * FRIEND_SPEED
            info = ("PURSUING", target, None, None)
        else:
            vel = pygame.math.Vector2(0, 0)
            info = ("IDLE", None, None, None)
        
        return info, vel
```

### Exemplo 2: Comportamento com Interceptação

```python
from DroneSwarm2D.core.utils import intercept_direction, can_intercept

class InterceptBehavior(BaseBehavior):
    def __init__(self):
        super().__init__(behavior_type=BehaviorType.COMMON)
    
    def apply(self, state, joystick_controlled: bool = False) -> tuple:
        from DroneSwarm2D.core.settings import (
            CELL_SIZE, FRIEND_SPEED, ENEMY_SPEED, INTEREST_POINT_CENTER
        )
        
        pos = np.squeeze(state['pos'])
        pos = pygame.math.Vector2(pos[0], pos[1])
        enemy_intensity = np.squeeze(state['enemy_intensity'])
        enemy_direction = np.squeeze(state['enemy_direction'])
        
        # Encontrar alvos interceptáveis
        for cell, intensity in np.ndenumerate(enemy_intensity):
            if intensity < 0.4:
                continue
            
            target_pos = pygame.math.Vector2(
                (cell[0] + 0.5) * CELL_SIZE,
                (cell[1] + 0.5) * CELL_SIZE
            )
            
            # Vetor de velocidade do inimigo
            enemy_vel = pygame.math.Vector2(
                enemy_direction[cell][0],
                enemy_direction[cell][1]
            ) * ENEMY_SPEED
            
            # Verificar se é possível interceptar
            if can_intercept(pos, FRIEND_SPEED, target_pos, 
                           enemy_vel, INTEREST_POINT_CENTER):
                
                # Calcular direção de interceptação
                vel = intercept_direction(pos, FRIEND_SPEED, 
                                        target_pos, enemy_vel)
                info = ("INTERCEPT", target_pos, None, None)
                return info, vel
        
        # Patrulhar se não houver alvos
        vel = pygame.math.Vector2(0, 0)
        info = ("PATROL", None, None, None)
        return info, vel
```

### Exemplo 3: Comportamento de Formação Defensiva

```python
class FormationBehavior(BaseBehavior):
    def __init__(self, patrol_radius: float = 150):
        super().__init__(behavior_type=BehaviorType.COMMON)
        self.patrol_radius = patrol_radius
    
    def apply(self, state, joystick_controlled: bool = False) -> tuple:
        from DroneSwarm2D.core.settings import FRIEND_SPEED, INTEREST_POINT_CENTER
        
        pos = np.squeeze(state['pos'])
        pos = pygame.math.Vector2(pos[0], pos[1])
        enemy_intensity = np.squeeze(state['enemy_intensity'])
        
        # Se houver inimigos, perseguir o mais próximo do centro
        if np.max(enemy_intensity) > 0.4:
            # [Código de perseguição...]
            pass
        else:
            # Manter formação circular
            r_vec = pos - INTEREST_POINT_CENTER
            current_distance = r_vec.length()
            
            if current_distance == 0:
                r_vec = pygame.math.Vector2(self.patrol_radius, 0)
                current_distance = self.patrol_radius
            
            # Correção radial
            radial_error = self.patrol_radius - current_distance
            k_radial = 0.05
            radial_correction = k_radial * radial_error * r_vec.normalize()
            
            # Velocidade tangencial (órbita)
            tangent = pygame.math.Vector2(-r_vec.y, r_vec.x).normalize()
            vel = tangent * FRIEND_SPEED + radial_correction
            
            info = ("FORMATION", None, None, None)
        
        return info, vel
```

### Exemplo 4: Comportamento AEW (Alerta Aéreo)

```python
class CustomAEWBehavior(BaseBehavior):
    def __init__(self):
        super().__init__(behavior_type=BehaviorType.AEW)
    
    def apply(self, state, joystick_controlled: bool = False) -> tuple:
        from DroneSwarm2D.core.settings import AEW_RANGE, AEW_SPEED, INTEREST_POINT_CENTER
        
        pos = np.squeeze(state['pos'])
        pos = pygame.math.Vector2(pos[0], pos[1])
        
        # Órbita ao redor do ponto de interesse
        r_vec = pos - INTEREST_POINT_CENTER
        current_distance = r_vec.length()
        
        if current_distance == 0:
            r_vec = pygame.math.Vector2(AEW_RANGE, 0)
            current_distance = AEW_RANGE
        
        # Correção de órbita
        radial_error = AEW_RANGE - current_distance
        k_radial = 0.05
        radial_correction = k_radial * radial_error * r_vec.normalize()
        
        # Velocidade tangencial
        tangent = pygame.math.Vector2(-r_vec.y, r_vec.x).normalize()
        vel = tangent * AEW_SPEED + radial_correction
        
        info = ("AEW", None, None, None)
        return info, vel
```

---

## 🎮 Arquivo Principal (`main.py`)

### Estrutura Básica

```python
import DroneSwarm2D

# 1. Inicializar configurações
settings = DroneSwarm2D.init(
    config_path="./config.json",
    fullscreen=True
)

# 2. Importar comportamentos customizados
from behaviors import (
    FriendCommonBehavior, 
    FriendRadarBehavior, 
    FriendAEWBehavior
)

# 3. Criar ambiente
env = DroneSwarm2D.AirTrafficEnv(
    mode='human',
    friend_behavior=FriendCommonBehavior(),
    friend_aew_behavior=FriendAEWBehavior(),
    friend_radar_behavior=FriendRadarBehavior(),
    enemy_behavior=settings.ENEMY_BEHAVIOR,
    demilitarized_zones=settings.DMZ,
    seed=42
)

# 4. Loop de simulação
obs, done = env.reset()
while not done:
    obs, reward, done, info = env.step(None)

# 5. Fechar ambiente
env.close()
```

### Executando Múltiplos Episódios

```python
def main():
    NUM_EPISODES = 5
    results = []
    
    for episode in range(NUM_EPISODES):
        obs, done = env.reset()
        episode_reward = 0.0
        
        print(f"Episódio {episode + 1}/{NUM_EPISODES}...")
        
        while not done:
            obs, reward, done, info = env.step(None)
            episode_reward += reward
        
        # Estatísticas do episódio
        print(f"  Steps: {info['current_step']}")
        print(f"  Reward: {info['accum_reward']:.2f}")
        print(f"  Inimigos abatidos: {info['enemies_shotdown']}")
        print(f"  Ataques bem-sucedidos: {info['sucessful_attacks']}")
        print(f"  Saúde do PI: {info['interest_point_health']}")
        
        results.append(info)
    
    # Estatísticas finais
    avg_reward = sum(r['accum_reward'] for r in results) / len(results)
    print(f"\nReward médio: {avg_reward:.2f}")
    
    env.close()
```

---

## 📊 Informações Retornadas

### Objeto `info` Retornado por `step()`

```python
info = {
    'current_step': int,              # Step atual da simulação
    'accum_reward': float,            # Reward acumulado
    'enemies_shotdown': int,          # Inimigos neutralizados
    'friends_shotdown': int,          # Amigos perdidos
    'sucessful_attacks': int,         # Ataques bem-sucedidos ao PI
    'interest_point_health': int,     # Saúde restante do ponto de interesse
    'state_percentages': dict,        # Porcentagem de tempo em cada estado
    'total_distance_traveled': float  # Distância total percorrida
}
```

### Porcentagens de Estado

```python
# Exemplo de state_percentages
{
    'PURSUING': 45.2,      # 45.2% do tempo perseguindo
    'HOLD - WAIT': 30.8,   # 30.8% em posição de espera
    'HOLD - INTCPT': 15.0, # 15.0% interceptando
    'PATROLLING': 9.0      # 9.0% patrulhando
}
```

---

## 🔧 Utilitários Disponíveis

### Funções de Conversão

```python
from DroneSwarm2D.core.utils import pos_to_cell

# Converter posição para célula da grade
pos = pygame.math.Vector2(250.5, 180.7)
cell = pos_to_cell(pos)  # (12, 9) se CELL_SIZE = 20

# Com parâmetros customizados
cell = pos_to_cell(pos, cell_size=40, grid_width=50, grid_height=50)
```

### Cálculo de Interceptação

```python
from DroneSwarm2D.core.utils import intercept_direction, can_intercept

# Verificar se é possível interceptar
chaser_pos = pygame.math.Vector2(100, 100)
target_pos = pygame.math.Vector2(300, 300)
target_vel = pygame.math.Vector2(1, 1)

can_catch = can_intercept(
    chaser_pos, 
    chaser_speed=5.0,
    target_pos, 
    target_vel,
    point_of_interest=INTEREST_POINT_CENTER
)

# Calcular direção de interceptação
if can_catch:
    velocity = intercept_direction(
        chaser_pos,
        chaser_speed=5.0,
        target_pos,
        target_vel
    )
```

### Desenhar Elementos

```python
from DroneSwarm2D.core.utils import draw_dashed_circle, draw_dashed_line

# Desenhar círculo tracejado
draw_dashed_circle(
    surface=screen,
    color=(255, 255, 255, 128),
    center=(400, 300),
    radius=150,
    dash_length=5,
    space_length=5,
    width=2
)

# Desenhar linha tracejada
start = pygame.math.Vector2(100, 100)
end = pygame.math.Vector2(400, 400)
draw_dashed_line(
    surface=screen,
    color=(255, 0, 0, 128),
    start_pos=start,
    end_pos=end,
    width=2,
    dash_length=10,
    space_length=5
)
```

---

## 🎨 Controles da Interface

Durante a simulação, você pode usar os seguintes botões:

- **Tog. Graph**: Ativa/desativa visualização 3D das matrizes de detecção
- **Pause**: Pausa/retoma a simulação
- **Reset**: Reinicia o episódio atual
- **Exit**: Encerra a simulação
- **Tog. Friend Range**: Mostra/oculta alcance de detecção dos amigos
- **Tog. Enemy Range**: Mostra/oculta alcance de detecção dos inimigos
- **Tog. Friend Comm.**: Mostra/oculta links de comunicação
- **Tog. DMZ**: Mostra/oculta zonas desmilitarizadas
- **Tog. Comm Range**: Mostra/oculta alcance de comunicação
- **Export Tacview**: Exporta trajetórias em formato Tacview
- **Tog. Save Frames**: Salva frames da simulação
- **Tog. Target Lines**: Mostra linhas dos inimigos ao PI
- **Tog. Trajetory**: Mostra/oculta trajetórias dos drones
- **Tog. Debug**: Mostra informações de debug
- **Tog. D. Passive**: Ativa/desativa detecção passiva
- **Tog. Return**: Ordena retorno à base

### Seleção de Drones

Clique em qualquer drone amigo para selecioná-lo e visualizar suas matrizes de detecção no gráfico 3D.

---

## 🧪 Exemplo Completo: Sistema de Defesa em Camadas

```python
# behaviors.py
import DroneSwarm2D
settings = DroneSwarm2D.init("./src/config.json", fullscreen=True)

import numpy as np
import pygame
from DroneSwarm2D.core.utils import intercept_direction, pos_to_cell
from DroneSwarm2D.behaviorsDefault import BaseBehavior, BehaviorType

class LayeredDefenseBehavior(BaseBehavior):
    """Sistema de defesa em três camadas:
    1. Camada externa: Interceptação precoce
    2. Camada média: Contenção e bloqueio
    3. Camada interna: Defesa de último recurso
    """
    
    def __init__(self):
        super().__init__(behavior_type=BehaviorType.COMMON)
        self.outer_radius = 300
        self.middle_radius = 200
        self.inner_radius = 100
    
    def apply(self, state, joystick_controlled: bool = False) -> tuple:
        from DroneSwarm2D.core.settings import (
            CELL_SIZE, FRIEND_SPEED, ENEMY_SPEED, INTEREST_POINT_CENTER
        )
        
        pos = np.squeeze(state['pos'])
        pos = pygame.math.Vector2(pos[0], pos[1])
        enemy_intensity = np.squeeze(state['enemy_intensity'])
        enemy_direction = np.squeeze(state['enemy_direction'])
        
        # Determinar camada do drone
        distance_to_center = pos.distance_to(INTEREST_POINT_CENTER)
        
        if distance_to_center > self.middle_radius:
            layer = "OUTER"
        elif distance_to_center > self.inner_radius:
            layer = "MIDDLE"
        else:
            layer = "INNER"
        
        # Buscar alvos prioritários
        enemy_targets = []
        for cell, intensity in np.ndenumerate(enemy_intensity):
            if intensity < 0.4:
                continue
            
            target_pos = pygame.math.Vector2(
                (cell[0] + 0.5) * CELL_SIZE,
                (cell[1] + 0.5) * CELL_SIZE
            )
            target_dist = target_pos.distance_to(INTEREST_POINT_CENTER)
            priority = 1000 - target_dist  # Mais próximo = maior prioridade
            
            enemy_targets.append((priority, target_pos, cell))
        
        if enemy_targets:
            enemy_targets.sort(reverse=True)
            _, target_pos, cell = enemy_targets[0]
            
            enemy_vel = pygame.math.Vector2(
                enemy_direction[cell][0],
                enemy_direction[cell][1]
            ) * ENEMY_SPEED
            
            # Estratégia por camada
            if layer == "OUTER":
                # Interceptação agressiva
                vel = intercept_direction(pos, FRIEND_SPEED * 1.2, 
                                        target_pos, enemy_vel)
                info = ("OUTER-INTERCEPT", target_pos, None, None)
                
            elif layer == "MIDDLE":
                # Bloqueio tático
                # Posicionar-se entre o inimigo e o centro
                to_center = INTEREST_POINT_CENTER - target_pos
                block_point = target_pos + to_center.normalize() * 50
                direction = (block_point - pos).normalize()
                vel = direction * FRIEND_SPEED
                info = ("MIDDLE-BLOCK", block_point, None, None)
                
            else:  # INNER
                # Defesa desesperada - ir direto ao inimigo
                direction = (target_pos - pos).normalize()
                vel = direction * FRIEND_SPEED * 1.5
                info = ("INNER-DESPERATE", target_pos, None, None)
        else:
            # Sem inimigos - manter posição na camada
            target_radius = {
                "OUTER": self.outer_radius,
                "MIDDLE": self.middle_radius,
                "INNER": self.inner_radius
            }[layer]
            
            r_vec = pos - INTEREST_POINT_CENTER
            if r_vec.length() == 0:
                r_vec = pygame.math.Vector2(target_radius, 0)
            
            radial_error = target_radius - r_vec.length()
            radial_correction = 0.1 * radial_error * r_vec.normalize()
            
            tangent = pygame.math.Vector2(-r_vec.y, r_vec.x).normalize()
            vel = tangent * FRIEND_SPEED * 0.5 + radial_correction
            
            info = (f"{layer}-PATROL", None, None, None)
        
        return info, vel


# main.py
import DroneSwarm2D
settings = DroneSwarm2D.init("./src/config.json", fullscreen=True)

from behaviors import LayeredDefenseBehavior

env = DroneSwarm2D.AirTrafficEnv(
    mode='human',
    friend_behavior=LayeredDefenseBehavior(),
    enemy_behavior=settings.ENEMY_BEHAVIOR,
    demilitarized_zones=settings.DMZ,
    seed=42
)

obs, done = env.reset()
while not done:
    obs, reward, done, info = env.step(None)

print(f"Defesa concluída! PI Health: {info['interest_point_health']}")
env.close()
```

---

## 📝 Notas Importantes

### Sementes Aleatórias

Tanto drones amigos quanto inimigos possuem geradores de números aleatórios independentes:

```python
# Definir seed da classe (afeta todos os drones daquele tipo)
FriendDrone.set_class_seed(seed=42)
EnemyDrone.set_class_seed(seed=123)

# Ou passar seed ao criar ambiente
env = DroneSwarm2D.AirTrafficEnv(seed=42)
```

### Debugging

Ative informações de debug para ver:
- Estado atual de cada drone
- Alvos sendo perseguidos
- Linhas de comunicação
- Matrizes de detecção

---

## 🤝 Contribuindo

Para reportar bugs ou sugerir melhorias:
1. Crie uma issue no repositório
2. Descreva o comportamento esperado vs. observado
3. Inclua código mínimo para reproduzir o problema

---

## 📄 Licença

MIT License - veja o arquivo LICENSE para detalhes.

---

## 🆘 Suporte

Em caso de dúvidas:
1. Consulte a documentação inline dos métodos
2. Verifique os exemplos em `behaviors.py`
3. Abra uma issue no GitHub
