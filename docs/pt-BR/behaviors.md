# Escrevendo comportamentos

🇺🇸 [English version](../en/behaviors.md) · [Voltar ao README](../../README.pt-BR.md)

Um **comportamento** é a política de um drone amigo. A cada passo, cada drone entrega sua percepção *local* ao seu comportamento e voa com a velocidade que ele retorna. Escrever uma nova tática defensiva significa escrever uma pequena classe.

## Sumário

- [Anatomia de um comportamento](#anatomia-de-um-comportamento)
- [O dicionário `state`](#o-dicionário-state)
- [Funções auxiliares](#funções-auxiliares)
- [Conectando comportamentos ao ambiente](#conectando-comportamentos-ao-ambiente)
- [Exemplos](#exemplos)
- [Dicas](#dicas)
- [Comparando políticas](#comparando-políticas)

## Anatomia de um comportamento

```python
import numpy as np
import pygame

from DroneSwarm2D.behaviorsDefault import BaseBehavior, BehaviorType
from DroneSwarm2D.core import settings


class MyBehavior(BaseBehavior):
    def __init__(self):
        super().__init__(BehaviorType.COMMON)     # o papel do drone

    def apply(self, state, joystick_controlled=False):
        pos = pygame.math.Vector2(*np.squeeze(state["pos"]).tolist())
        # ... decidir ...
        to_centre = settings.INTEREST_POINT_CENTER - pos
        velocity = to_centre.normalize() * settings.FRIEND_SPEED if to_centre.length() > 0 else to_centre
        info = ("GOING HOME", None, None, None)  # (rótulo, ponto-alvo, ponto de referência, extra)
        return info, velocity
```

**O papel.** `BehaviorType` define o papel do drone:

- `COMMON`: interceptador;
- `AEW`: drone de alerta antecipado, com um alcance de sensor diferente;
- `RADAR`: sensor fixo;
- `AI`: reservado para políticas aprendidas; comporta-se como `COMMON`.

Os membros de `BehaviorType` são considerados iguais aos seus nomes em comparações (`BehaviorType.RADAR == "RADAR"`), e `BaseBehavior` também aceita o nome como string.

**O valor de retorno.** `apply` retorna `(info, velocity)`:

- `velocity` é um `pygame.math.Vector2`, em pixels por passo. Seu módulo não é verificado: mantenha-o no máximo igual à velocidade do drone.
- `info[0]` é um rótulo curto. Ele é exibido abaixo do drone e agregado em `info["state_percentages"]`.
- `info[1]` e `info[2]` são pontos `Vector2` opcionais. No modo de depuração, é desenhada uma linha do drone até `info[1]` e outra do ponto de interesse até `info[2]`.
- `info[3]` fica livre para os seus próprios dados.

**Ordem de importação.** Importe seus comportamentos **depois** de `DroneSwarm2D.init(...)`. Além disso, leia as configurações no momento da chamada, como `settings.FRIEND_SPEED` com `from DroneSwarm2D.core import settings`, a menos que você as importe depois de `init()`.

## O dicionário `state`

Todos os arrays mantêm um eixo inicial de lote (batch) de tamanho 1, o que é conveniente para redes neurais. Use `np.squeeze` para removê-lo.

| Chave | Formato | Conteúdo |
|---|---|---|
| `drone_id` | `int` | Id do drone (único dentro do episódio) |
| `pos` | `(1, 2)` float32 | Posição em pixels |
| `enemy_intensity` | `(1, GRID_W, GRID_H)` | Recência das detecções de inimigos (1 = visto agora; decai por `DECAY_FACTOR` a cada passo) |
| `enemy_direction` | `(1, GRID_W, GRID_H, 2)` | Direção unitária do inimigo visto em cada célula |
| `friend_intensity` | `(1, GRID_W, GRID_H)` | Recência das detecções de amigos |
| `friend_direction` | `(1, GRID_W, GRID_H, 2)` | Direção unitária do amigo visto em cada célula |

As matrizes combinam o que o próprio drone detectou com o que ele recebeu pela rede. Elas **não** são a verdade de referência (ground truth): as posições são quantizadas em células, a informação pode estar desatualizada e drones defeituosos podem ter injetado lixo.

> Os arrays são *views* da memória do próprio drone. Não os modifique diretamente (in place).

## Funções auxiliares

Tudo o que está abaixo fica em `DroneSwarm2D.core.utils`:

| Função | Finalidade |
|---|---|
| `active_cells(matrix, threshold)` | Lista das células `(i, j)` com valor ≥ limiar, em ordem de linhas (row-major) (rápida, vetorizada) |
| `cell_center((i, j))` | Centro de uma célula, em pixels |
| `pos_to_cell(pos)` | Célula que contém uma posição |
| `intercept_direction(pos, speed, target_pos, target_vel)` | Velocidade que intercepta um alvo que se move com velocidade constante (recorre à perseguição pura quando isso não é possível) |
| `can_intercept(pos, speed, target_pos, target_vel, poi)` | Se a interceptação acontece antes de o alvo alcançar `poi` |
| `get_friends_hold(state, ...)` | Células de amigos que ficaram sem atribuição em uma alocação gulosa de interceptações |

As matrizes de direção armazenam direções **unitárias**. Multiplique-as por `settings.ENEMY_SPEED` para obter a velocidade do inimigo que `intercept_direction` espera.

## Conectando comportamentos ao ambiente

```python
env = DroneSwarm2D.AirTrafficEnv(
    friend_behavior=MyBehavior(),            # interceptadores
    friend_aew_behavior=MyAEWBehavior(),     # drones AEW (AEW_COUNT)
    friend_radar_behavior=MyRadarBehavior(), # radares (RADAR_COUNT)
)
```

Qualquer argumento omitido usa a política padrão. **Uma única instância é compartilhada por todos os drones de um mesmo papel**, então a memória de cada drone deve ser indexada por `state["drone_id"]` (veja o [exemplo 5](#5-memória-comprometer-se-com-um-alvo)).

## Exemplos

Todos os exemplos pressupõem estas importações (feitas depois de `DroneSwarm2D.init(...)`):

```python
import numpy as np
import pygame

from DroneSwarm2D.behaviorsDefault import BaseBehavior, BehaviorType, FriendCommonBehaviorDefault
from DroneSwarm2D.core import settings
from DroneSwarm2D.core.utils import active_cells, can_intercept, cell_center, intercept_direction

V = pygame.math.Vector2


def position(state):
    return V(*np.squeeze(state["pos"]).tolist())
```

### 1. Perseguição pura do inimigo mais próximo

```python
class SimplePursuit(BaseBehavior):
    def __init__(self, threshold=0.4):
        super().__init__(BehaviorType.COMMON)
        self.threshold = threshold

    def apply(self, state, joystick_controlled=False):
        pos = position(state)
        cells = active_cells(np.squeeze(state["enemy_intensity"]), self.threshold)
        if not cells:
            return ("IDLE", None, None, None), V(0, 0)
        target = min((cell_center(c) for c in cells), key=pos.distance_to)
        if target == pos:
            return ("ON TARGET", target, None, None), V(0, 0)
        return ("PURSUING", target, None, None), (target - pos).normalize() * settings.FRIEND_SPEED
```

### 2. Interceptação, só quando ela pode dar certo

```python
class SmartIntercept(BaseBehavior):
    def __init__(self):
        super().__init__(BehaviorType.COMMON)

    def apply(self, state, joystick_controlled=False):
        pos = position(state)
        intensity = np.squeeze(state["enemy_intensity"])
        heading = np.squeeze(state["enemy_direction"])
        center = settings.INTEREST_POINT_CENTER

        # Ameaças mais urgentes primeiro.
        for cell in sorted(active_cells(intensity, 0.4), key=lambda c: cell_center(c).distance_to(center)):
            target, enemy_vel = cell_center(cell), V(*heading[cell]) * settings.ENEMY_SPEED
            if can_intercept(pos, settings.FRIEND_SPEED, target, enemy_vel, center):
                vel = intercept_direction(pos, settings.FRIEND_SPEED, target, enemy_vel)
                return ("INTERCEPT", target, None, None), vel
        return ("WAIT", None, None, None), V(0, 0)
```

### 3. Órbita de patrulha quando não há nada a perseguir

```python
def orbit(pos, radius, speed, k_radial=0.05):
    """Orbita o ponto de interesse, sendo puxado de volta para `radius`."""
    r_vec = pos - settings.INTEREST_POINT_CENTER
    if r_vec.length() == 0:
        r_vec = V(radius, 0)
    tangent = V(-r_vec.y, r_vec.x).normalize()
    return tangent * speed + k_radial * (radius - r_vec.length()) * r_vec.normalize()


class PatrolOrIntercept(SmartIntercept):
    def __init__(self, patrol_radius=150):
        super().__init__()
        self.patrol_radius = patrol_radius

    def apply(self, state, joystick_controlled=False):
        info, vel = super().apply(state, joystick_controlled)
        if info[0] == "WAIT":
            return ("PATROL", None, None, None), orbit(position(state), self.patrol_radius, settings.FRIEND_SPEED)
        return info, vel
```

### 4. Uma órbita AEW personalizada

```python
class TightAEW(BaseBehavior):
    def __init__(self, radius_factor=0.8):
        super().__init__(BehaviorType.AEW)
        self.radius_factor = radius_factor

    def apply(self, state, joystick_controlled=False):
        radius = settings.AEW_RANGE * self.radius_factor
        return ("AEW", None, None, None), orbit(position(state), radius, settings.AEW_SPEED)
```

### 5. Memória: comprometer-se com um alvo

A instância do comportamento é compartilhada, então guarde os dados de cada drone em um dicionário indexado por `drone_id`:

```python
class CommittedPursuit(SimplePursuit):
    def __init__(self, commit_steps=30):
        super().__init__()
        self.commit_steps = commit_steps
        self.memory = {}  # drone_id -> (alvo, passos restantes)

    def apply(self, state, joystick_controlled=False):
        drone_id, pos = state["drone_id"], position(state)
        target, left = self.memory.get(drone_id, (None, 0))
        if left > 0 and target.distance_to(pos) > settings.EPSILON:
            self.memory[drone_id] = (target, left - 1)
            return ("COMMITTED", target, None, None), (target - pos).normalize() * settings.FRIEND_SPEED

        info, vel = super().apply(state, joystick_controlled)
        if info[0] == "PURSUING":
            self.memory[drone_id] = (info[1], self.commit_steps)
        return info, vel
```

### 6. Estendendo a política padrão

`FriendCommonBehaviorDefault` cuida da atribuição gulosa e da perseguição. Sobrescreva `_hold_position` para mudar apenas o que os drones ociosos fazem:

```python
class DefaultButPatrol(FriendCommonBehaviorDefault):
    def _hold_position(self, pos, friend_intensity, enemy_intensity, enemy_direction, friends_hold=None, **kwargs):
        return ("PATROL", None, None, friends_hold), orbit(pos, settings.INITIAL_DISTANCE, settings.FRIEND_SPEED)
```

## Dicas

- **Evite laços em Python sobre a grade.** Uma tela de 1920×1080 tem cerca de 3 600 células, e `apply` é executado para cada drone a cada passo. Use `active_cells` ou NumPy (`np.argwhere`, máscaras) em vez de `np.ndenumerate`.
- **Mantenha as execuções reprodutíveis.** Evite o estado global de `random` / `numpy.random`. Use seu próprio gerador com semente (`random.Random(seed)`) se a política for estocástica.
- **Rotule os estados.** Os rótulos alimentam `state_percentages`, que é a forma mais fácil de ver em que uma política passa o tempo.
- **Retorne uma velocidade no máximo igual à velocidade do drone.** O motor não a limita.
- **Depure visualmente.** Execute com `mode="human"`, mantenha *Tog. Debug* ativado e retorne um ponto-alvo em `info[1]`.

## Comparando políticas

Execute cada política sem janela (headless) com as mesmas sementes e compare as métricas:

```python
import DroneSwarm2D

settings = DroneSwarm2D.init("config.json", headless=True, screen_size=(1920, 1080))
from behaviors import SmartIntercept  # seu módulo, importado depois de init()


def evaluate(behavior, seeds):
    results = []
    for seed in seeds:
        env = DroneSwarm2D.AirTrafficEnv(mode=None, seed=seed, friend_behavior=behavior,
                                         enemy_behavior=settings.ENEMY_BEHAVIOR,
                                         demilitarized_zones=settings.DMZ)
        _, done = env.reset()
        while not done:
            _, _, done, info = env.step(None)
        results.append(info)
    return results


default = evaluate(DroneSwarm2D.behaviorsDefault.FriendCommonBehaviorDefault(), range(20))
custom = evaluate(SmartIntercept(), range(20))
print(sum(r["sucessful_attacks"] for r in default), sum(r["sucessful_attacks"] for r in custom))
```

Com o extra `analysis` instalado, converta os resultados em um `pandas.DataFrame` e use `mannwhitneyu` / `ks_2samp` do SciPy para testar se a diferença é significativa. Esses são os testes usados na tese.
