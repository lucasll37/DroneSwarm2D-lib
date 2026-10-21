# Tutorial do DroneSwarm2D

🇺🇸 [Read in English](TUTORIAL.md)

Este tutorial cobre o que você precisa para alterar o cenário e escrever as suas próprias políticas de drones. A documentação completa (arquitetura, todas as chaves de configuração, referência da API) está em <https://github.com/lucasll37/DroneSwarm2D-lib/tree/main/docs/pt-BR>.

## Sumário

1. [O laço principal](#1-o-laço-principal)
2. [O cenário: `config.json`](#2-o-cenário-configjson)
3. [Como a simulação funciona](#3-como-a-simulação-funciona)
4. [Escrevendo comportamentos](#4-escrevendo-comportamentos)
5. [Métricas](#5-métricas)
6. [A interface](#6-a-interface)
7. [Receitas](#7-receitas)

## 1. O laço principal

`src/main.py` faz quatro coisas:

```python
import DroneSwarm2D

# 1. Carregue o cenário. Sempre o primeiro passo: a biblioteca o lê no momento da importação.
settings = DroneSwarm2D.init("src/config.json", fullscreen=False)   # headless=True: sem janela

# 2. Só agora importe os seus comportamentos.
from behaviors import FriendCommonBehavior, FriendAEWBehavior, FriendRadarBehavior

# 3. Crie o ambiente.
env = DroneSwarm2D.AirTrafficEnv(
    mode="human",                        # None = sem janela (headless) e rápido
    friend_behavior=FriendCommonBehavior(),
    friend_aew_behavior=FriendAEWBehavior(),
    friend_radar_behavior=FriendRadarBehavior(),
    enemy_behavior=settings.ENEMY_BEHAVIOR,
    demilitarized_zones=settings.DMZ,
    seed=42,                             # mesma semente -> mesmo episódio
)

# 4. Execute os episódios.
obs, done = env.reset()
while not done:
    obs, reward, done, info = env.step(None)
env.close()
```

## 2. O cenário: `config.json`

Valores do tipo string são expressões Python sobre as chaves definidas acima deles (`"EXTERNAL_RADIUS": "INTERNAL_RADIUS * 4"`), e `"None"` significa `None`. As chaves mais úteis:

| Chave | Significado |
|---|---|
| `FRIEND_COUNT`, `ENEMY_COUNT` | Tamanho de cada lado (`FRIEND_COUNT` inclui radares e drones AEW) |
| `RADAR_COUNT`, `AEW_COUNT` | Radares fixos / drones de alerta aéreo antecipado (posicionados a `RADAR_RANGE` / `AEW_RANGE` do centro) |
| `BROKEN_COUNT` | Interceptadores com sensores defeituosos que espalham dados aleatórios |
| `FRIEND_DETECTION_RANGE`, `COMMUNICATION_RANGE` | Alcance do sensor e do rádio, em pixels |
| `N_CONNECTIONS`, `CICLE_COMM_BY_STEP`, `MESSAGE_LOSS_PROBABILITY` | Rede: enlaces por drone, trocas por passo, taxa de perda |
| `DECAY_FACTOR` | Com que rapidez a informação envelhece (intensidade × fator a cada passo) |
| `PASSIVE_DETECTION`, `N_LINE_SIGHT_CROSSING` | Só confirma inimigos vistos por N sensores |
| `ENEMY_BEHAVIOR` | `"None"` (aleatório), `"direct"`, `"zigzag"`, `"spiral"`, `"formation"`, ... |
| `INITIAL_AGGRESSIVENESS`, `ESCAPE_STEPS` | Como os inimigos reagem ao serem detectados |
| `NEUTRALIZATION_RANGE`, `NEUTRALIZATION_PROB_*` | Combate |
| `DMZ` | Zonas desmilitarizadas `[x, y, radius]`, onde ninguém pode combater |
| `JOYSTICK` | `"Friend"` / `"Enemy"`: pilote um drone com um controle (joystick); `"None"` para desativar |

O tamanho da janela é definido pela sua tela. Passe `screen_size=(1920, 1080)` para `DroneSwarm2D.init` para que os resultados não dependam do monitor.

## 3. Como a simulação funciona

A cada passo:

1. **Cada drone amigo:**
   - *esquece*: as suas matrizes decaem;
   - *conecta-se*: escolhe até `N_CONNECTIONS` vizinhos;
   - *percebe*: detecta inimigos e amigos dentro do alcance;
   - *conversa*: mescla as matrizes com as dos vizinhos, mantendo a informação mais recente em cada célula;
   - *age*: o seu comportamento retorna uma velocidade.
2. **Os inimigos** se movem. Se um amigo estiver dentro de `ENEMY_DETECTION_RANGE`, o inimigo ataca em linha reta ou foge; quanto mais perto ele estiver do alvo, maior a probabilidade de atacar.
3. **Ataques.** Um inimigo que alcança o ponto de interesse causa dano a ele (e destrói um radar).
4. **Engajamentos.** Um amigo e um inimigo a uma distância menor que `NEUTRALIZATION_RANGE` combatem: o inimigo, o amigo ou ambos são destruídos.

O conhecimento de um drone é um conjunto de matrizes de grade (células de `CELL_SIZE` pixels):

- `enemy_intensity[i, j]` vale 1 quando um inimigo acabou de ser visto na célula `(i, j)` e cai em direção a 0 à medida que a informação envelhece;
- `enemy_direction[i, j]` é a direção (vetor unitário) desse inimigo;
- as matrizes de amigos guardam a mesma informação sobre os drones amigos.

## 4. Escrevendo comportamentos

Um comportamento é uma classe com um método `apply(state)` que retorna `(info, velocity)`:

```python
import numpy as np
import pygame

from DroneSwarm2D.behaviorsDefault import BaseBehavior, BehaviorType
from DroneSwarm2D.core import settings
from DroneSwarm2D.core.utils import active_cells, cell_center, intercept_direction


class MyBehavior(BaseBehavior):
    def __init__(self):
        super().__init__(BehaviorType.COMMON)   # COMMON, AEW ou RADAR

    def apply(self, state, joystick_controlled=False):
        pos = pygame.math.Vector2(*np.squeeze(state["pos"]).tolist())
        enemies = active_cells(np.squeeze(state["enemy_intensity"]), 0.4)
        if not enemies:
            return ("WAIT", None, None, None), pygame.math.Vector2(0, 0)

        cell = min(enemies, key=lambda c: cell_center(c).distance_to(pos))
        heading = pygame.math.Vector2(*np.squeeze(state["enemy_direction"])[cell])
        vel = intercept_direction(pos, settings.FRIEND_SPEED, cell_center(cell), heading * settings.ENEMY_SPEED)
        return ("PURSUING", cell_center(cell), None, None), vel
```

**O dicionário `state`.** Todo array tem um primeiro eixo de tamanho 1; use `np.squeeze`.

| Chave | Formato | Conteúdo |
|---|---|---|
| `drone_id` | `int` | Id do drone |
| `pos` | `(1, 2)` | Posição em pixels |
| `enemy_intensity` / `friend_intensity` | `(1, GRID_W, GRID_H)` | Recência das detecções (0..1) |
| `enemy_direction` / `friend_direction` | `(1, GRID_W, GRID_H, 2)` | Direções unitárias |

**Regras práticas:**

- `info[0]` é um rótulo exibido na tela e contabilizado em `info["state_percentages"]`; `info[1]` é um ponto-alvo opcional desenhado no modo de depuração.
- Retorne uma velocidade cujo módulo não ultrapasse `settings.FRIEND_SPEED`.
- Uma única instância é compartilhada por todos os drones de uma mesma função, então guarde a memória de cada drone em um dicionário indexado por `state["drone_id"]`.
- Use `active_cells` em vez de laços Python sobre a grade inteira: `apply` é executado para cada drone a cada passo.
- As direções são vetores unitários; multiplique por `settings.ENEMY_SPEED` para obter velocidades.
- Não modifique os arrays de `state`: eles são visões (views) da memória do drone.

A política da própria biblioteca, `FriendCommonBehaviorDefault`, é uma boa referência. Crie uma subclasse dela para reaproveitar a atribuição de alvos e mudar apenas o que os drones ociosos fazem:

```python
from DroneSwarm2D.behaviorsDefault import FriendCommonBehaviorDefault

class MyDefault(FriendCommonBehaviorDefault):
    def _hold_position(self, pos, friend_intensity, enemy_intensity, enemy_direction, friends_hold=None, **kw):
        return ("PATROL", None, None, friends_hold), pygame.math.Vector2(0, 0)
```

## 5. Métricas

`env.step()` retorna `info` com:

- `enemies_shotdown`, `friends_shotdown`;
- `sucessful_attacks`, `interest_point_health`;
- `accum_reward`, `current_step`;
- `friends_returned`, `enemies_remaining`, `friends_remaining`;
- `state_percentages`, `total_distance_traveled`.

## 6. A interface

| Botão | Efeito |
|---|---|
| Tog. Graph | Gráfico 3-D da percepção do drone selecionado (clique em um drone para selecioná-lo) |
| Pause / Reset / Exit | Controle da simulação |
| Tog. Friend Range / Enemy Range / Comm Range | Círculos de alcance dos sensores e do rádio |
| Tog. Friend Comm. | Enlaces de comunicação |
| Tog. DMZ | Mostra **e ativa** as zonas desmilitarizadas |
| Export Tacview | Grava trilhas em CSV em `./output/` quando o episódio termina |
| Tog. Save Frames | Salva quadros PNG em `./tmp/`; gere um vídeo com `python -m DroneSwarm2D create_video ./tmp/frames_...` |
| Tog. Target Lines / Trajetory / Debug | Auxílios visuais |
| Tog. D. Passive. | Detecção direta ↔ passiva (triangulação) |
| Tog. Return | Todos os drones retornam à base |

## 7. Receitas

**Comparar duas políticas com as mesmas sementes:**

```python
results = []
for seed in range(20):
    env = DroneSwarm2D.AirTrafficEnv(mode=None, seed=seed, friend_behavior=MyBehavior(),
                                     enemy_behavior=settings.ENEMY_BEHAVIOR, demilitarized_zones=settings.DMZ)
    _, done = env.reset()
    while not done:
        _, _, done, info = env.step(None)
    results.append(info)
print(sum(r["sucessful_attacks"] for r in results) / len(results))
```

**Estressar a rede:**
- aumente `MESSAGE_LOSS_PROBABILITY`;
- reduza `N_CONNECTIONS` ou `COMMUNICATION_RANGE`;
- adicione drones defeituosos com `BROKEN_COUNT`.

**Inimigos mais difíceis:**
- `"ENEMY_BEHAVIOR": "formation"`;
- aumente `INITIAL_AGGRESSIVENESS`;
- aumente `ENEMY_SPEED`.
