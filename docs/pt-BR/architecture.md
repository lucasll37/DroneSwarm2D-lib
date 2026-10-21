# Arquitetura e modelo de simulação

🇺🇸 [English version](../en/architecture.md) · [Voltar ao README](../../README.pt-BR.md)

Esta página descreve o que acontece dentro de um episódio do DroneSwarm2D: o mundo, como os drones percebem o ambiente e se comunicam, como os inimigos se comportam, as regras de combate, a recompensa e as garantias de reprodutibilidade.

## Sumário

- [O mundo](#o-mundo)
- [O laço de passos](#o-laço-de-passos)
- [Percepção: matrizes da grade](#percepção-matrizes-da-grade)
- [Detecção](#detecção)
- [Comunicação](#comunicação)
- [Comportamentos e movimento](#comportamentos-e-movimento)
- [A política padrão dos interceptadores](#a-política-padrão-dos-interceptadores)
- [Inimigos](#inimigos)
- [Ataques, engajamentos e zonas desmilitarizadas](#ataques-engajamentos-e-zonas-desmilitarizadas)
- [Recompensa, término e métricas](#recompensa-término-e-métricas)
- [Drones defeituosos](#drones-defeituosos)
- [Aleatoriedade e reprodutibilidade](#aleatoriedade-e-reprodutibilidade)
- [Mapa do código](#mapa-do-código)

## O mundo

- A tela é dividida entre a **área de simulação** (`SIM_WIDTH x SIM_HEIGHT` pixels, os 70% da esquerda por padrão) e o **painel de gráficos** à direita.
- As posições são dadas em pixels, com a origem no canto superior esquerdo e o *y* crescendo para baixo. As velocidades são dadas em pixels por passo, e um passo representa `DT_STEP` segundos simulados (usados no relógio e nos timestamps (carimbos de tempo) do Tacview).
- O **ponto de interesse** fica no centro da área de simulação. Ele tem três raios:
  - `INTERNAL_RADIUS`: a área protegida;
  - `INTEREST_POINT_ATTACK_RANGE`: a distância na qual um inimigo atinge o ativo protegido;
  - `EXTERNAL_RADIUS`: os drones amigos nunca podem se afastar do centro mais do que isso.
- A percepção usa uma **grade** grosseira de células de `CELL_SIZE` pixels (`GRID_WIDTH x GRID_HEIGHT`). `pos_to_cell(pos)` e `cell_center(cell)` fazem a conversão entre pixels e células.

No início de um episódio (`env.reset()`):

- os **radares terrestres** são posicionados em um círculo de raio `RADAR_RANGE`;
- os **drones AEW** são posicionados em um círculo de raio `AEW_RANGE`;
- os **interceptadores** restantes são posicionados em `INITIAL_N_LAYERS` anéis a partir de `INITIAL_DISTANCE` (separados por 40 px);
- os **inimigos** surgem em pontos aleatórios das quatro bordas.

## O laço de passos

`AirTrafficEnv.step()` executa, nesta ordem:

1. Cada drone amigo chama `FriendDrone.update()`, na ordem da lista. Assim, um drone já enxerga os movimentos e as mensagens dos drones atualizados antes dele no mesmo passo.
2. Cada inimigo chama `EnemyDrone.update(detector)`. Aqui, `detector` é a posição do primeiro amigo dentro de `ENEMY_DETECTION_RANGE`, ou `None`.
3. Os ataques são resolvidos e, em seguida, os engajamentos.
4. A recompensa é calculada (`compute_reward()`) e as trajetórias são registradas para o Tacview.
5. O quadro é desenhado se `mode == "human"`.

`FriendDrone.update()`, por sua vez, faz:

```
decay_matrices()                 esquecer: intensity *= DECAY_FACTOR
update_neighbors(friends)        escolher os enlaces de comunicação
update_passive_detection(...)    linhas de visada (apenas no modo passivo)
update_local_enemy_detection()   detectar inimigos, limpar células vazias
update_local_friend_detection()  detectar amigos, limpar células vazias
communication(friends)           mesclar as matrizes com os vizinhos
take_action()                    comportamento -> velocidade -> mover -> limites
```

## Percepção: matrizes da grade

Cada drone amigo armazena seis arrays:

| Array | Forma | Conteúdo |
|---|---|---|
| `enemy_intensity` | `(GRID_W, GRID_H)` | Recência de uma detecção de inimigo: 1 quando acabou de ocorrer, multiplicada por `DECAY_FACTOR` a cada passo |
| `enemy_direction` | `(GRID_W, GRID_H, 2)` | Vetor unitário da direção do inimigo visto na célula |
| `enemy_timestamp` | `(GRID_W, GRID_H)` | Tempo lógico da última atualização da célula |
| `friend_intensity` / `friend_direction` / `friend_timestamp` | idem | As mesmas informações sobre os drones amigos |

A intensidade é, portanto, uma *idade*. Com `DECAY_FACTOR = 0.99`, um valor de 0.4 significa que a célula foi confirmada pela última vez há cerca de 90 passos (0.99⁹⁰ ≈ 0.4).

**Limpeza.** Após o sensoriamento, toda célula dentro de 80% do alcance do sensor que não contém uma detecção recente (intensidade < 1) é zerada e marcada com o tempo atual. "Olhei e não vi nada aqui" também é informação, e se propaga como qualquer outra.

**Timestamps.** A partir desta versão, os timestamps vêm de um *relógio lógico* determinístico, que avança a cada atualização de drone, em vez do relógio do sistema. A regra de mesclagem continua a mesma ("o mais recente vence"), mas os resultados não dependem mais da velocidade do computador.

## Detecção

**Modo direto** (`PASSIVE_DETECTION = false`):

- Todo inimigo mais próximo do que o alcance do sensor do drone é detectado. O alcance é `FRIEND_DETECTION_RANGE` para os interceptadores, `AEW_DETECTION_RANGE` para os drones AEW e `RADAR_DETECTION_RANGE` para os radares.
- O primeiro avistamento armazena apenas a posição do inimigo, pois determinar a direção exige duas observações.
- A partir do segundo avistamento, a célula do inimigo recebe intensidade 1, a direção observada e o timestamp atual.
- Quando o inimigo passa para outra célula, a célula que ele deixou é limpa.

**Modo passivo** (`PASSIVE_DETECTION = true`, ou o botão *Tog. D. Passive.*):

- Um inimigo só é detectado quando pelo menos `N_LINE_SIGHT_CROSSING` sensores amigos o cobrem. Isso modela sensores que medem apenas o azimute (bearing-only) e que precisam do cruzamento de várias linhas de visada para localizar um alvo.
- Cada drone também traça suas linhas de visada em uma grade fina (`TRIANGULATION_GRANULARITY` subcélulas por célula). A grade fina é exposta como `passive_detection_matrix` e só é construída quando esse atributo é lido.
- A interface desenha as linhas de visada no modo de depuração.

Os **amigos** são detectados da mesma forma, dentro de `COMMUNICATION_RANGE`. Os drones AEW não são rastreados.

## Comunicação

**Enlaces.** A cada passo, um drone se conecta aos seus `N_CONNECTIONS` drones mais próximos dentro de `COMMUNICATION_RANGE`, além de todos os drones que o escolheram. Portanto, todos os enlaces são simétricos. A lista de vizinhos tem uma ordem determinística: primeiro as escolhas do próprio drone, por distância, e depois os drones que o escolheram.

**Trocas.** Durante `CICLE_COMM_BY_STEP` ciclos, o drone troca matrizes com cada vizinho. Cada troca é perdida com probabilidade `MESSAGE_LOSS_PROBABILITY`, de modo que os ciclos extras funcionam como retransmissões.

**Mesclagem.** Quando o drone A recebe as matrizes do drone B, cada célula em que o timestamp de B é mais recente é copiada para A: intensidade, direção e timestamp. Assim, a informação percorre vários saltos por passo, e o que os drones lembram se espalha pelo enxame junto com a sua idade.

A interface mostra os enlaces como linhas tracejadas, além da taxa de mensagens e do número de conexões.

## Comportamentos e movimento

A cada passo, um drone amigo monta um dicionário `state` a partir das próprias matrizes e chama `behavior.apply(state)`. A chamada retorna `(info, velocity)`:

- `info[0]` é um rótulo (por exemplo, `"PURSUING"`). Ele é desenhado abaixo do drone e contabilizado em `state_percentages`.
- `info[1]` e `info[2]` são pontos opcionais, desenhados como linhas de depuração.

Depois de se mover, o drone é mantido dentro da área de simulação (ricocheteando nas bordas) e a no máximo `EXTERNAL_RADIUS` do centro. Se fosse sair desse raio, ele é recolocado na fronteira e para.

O papel de um drone vem de `behavior.type`:

| `BehaviorType` | Alcance do sensor | Observações |
|---|---|---|
| `COMMON` | `FRIEND_DETECTION_RANGE` | Interceptador (armado) |
| `AEW` | `AEW_DETECTION_RANGE` | Não é rastreado pelos outros amigos; combate apenas por colisão |
| `RADAR` | `RADAR_DETECTION_RANGE` | Criado com `fixed=True` (nunca se move); destruído quando o ativo é atingido; combate apenas por colisão |
| `AI` | `FRIEND_DETECTION_RANGE` | Reservado para políticas aprendidas; comporta-se como `COMMON` |

O *retorno à base* (botão ou `env.return_to_base = True`) se sobrepõe aos comportamentos: todos os drones voam para o centro e são removidos ao chegar.

## A política padrão dos interceptadores

`FriendCommonBehaviorDefault` é determinística. Por isso, drones que compartilham a mesma visão da situação chegam ao mesmo plano sem precisar negociar:

1. **Alvos.** As células de inimigos com intensidade ≥ 0.4 são ordenadas pela distância ao ponto de interesse, das mais urgentes para as menos urgentes.
2. **Candidatos.** As células de amigos com intensidade ≥ 0.7 são os candidatos. A célula do próprio drone é sempre incluída.
3. **Atribuição gulosa.** Cada alvo, em ordem, recebe o candidato mais próximo que ainda não foi usado.
4. **Perseguição.** Se a célula do próprio drone foi atribuída, o drone segue o curso de interceptação `intercept_direction(pos, FRIEND_SPEED, target, heading * ENEMY_SPEED)` e seu rótulo é `PURSUING`.
5. **Espera.** Caso contrário, ele fica em espera:
   - `HOLD - WAIT`: o drone está no centro ou não tem nada melhor a fazer.
   - `HOLD - NO ENOUGH COMM`: há menos de `MIN_COMMUNICATION_HOLD` amigos detectados recentemente ao alcance, então o drone se move em direção ao centro para se reagrupar.
   - `HOLD - RETURN`: apenas quando `MIN_COMMUNICATION_HOLD == 0`; o drone volta para `INITIAL_DISTANCE`.
   - `HOLD - INTCPT` / `GO HOLD INTCPT`: um inimigo está vindo diretamente para o centro (cosseno ≥ 0.9), então o drone espera no ponto da trajetória do inimigo mais próximo de si, ou se desloca até esse ponto (se estiver a menos de `THRESHOLD_PROJECTION`).
   - `HOLD - SPREAD` (`HOLD_SPREAD = true`): o drone desliza ao redor do centro, afastando-se dos outros drones ociosos. Apenas a componente tangencial da repulsão é usada.

## Inimigos

**Surgimento.** Os inimigos aparecem em um ponto aleatório da borda após um atraso aleatório de 0–100 passos (os padrões de formação seguem um cronograma próprio).

**Padrões.** Com `ENEMY_BEHAVIOR = "None"`, cada inimigo sorteia um dos padrões aleatorizados abaixo. Uma string define o mesmo padrão para todos os inimigos.

| Padrão | Movimento |
|---|---|
| `direct` | Linha reta até o alvo |
| `zigzag`, `zigzag_damped`, `zigzag_unstable`, `zigzag_variable_period` | Oscilação senoidal lateral (constante, amortecida, aleatória, com período variável) |
| `spiral`, `spiral_bounce`, `spiral_oscillatory` | Espiral em direção ao centro; recuos periódicos para fora; oscilação adicional |
| `alternating` | 50 passos para dentro, 50 passos para o lado |
| `bounce_approach` | 60 passos para dentro, 30 passos para trás com um desvio aleatório |
| `circle_wait_advance` | Circula no lugar e depois avança |
| `formation`, `biformation` | Fileiras de 5 entrando pela direita (ou pelos dois lados); depois, `direct` |
| `focal-direct` | Todos os inimigos partem do canto superior direito, um a cada 20 passos |
| `debug`, `u-debug` | Padrões de teste (órbita; para dentro, para o lado, para fora) |
| `joystick` | Pilotado com um gamepad (`JOYSTICK = "Enemy"`) |

**Reação à detecção.**

- A **agressividade** vale `INITIAL_AGGRESSIVENESS` em `EXTERNAL_RADIUS` e cresce linearmente até 1 em `INITIAL_DISTANCE`.
- Quando um amigo está dentro de `ENEMY_DETECTION_RANGE`, o inimigo se compromete, com probabilidade igual à sua agressividade, com um **ataque desesperado**: uma linha reta até o alvo pelo resto do episódio.
- Caso contrário, ele entra em **fuga**, afastando-se do detector durante `ESCAPE_STEPS` passos, e depois retoma o seu padrão.

## Ataques, engajamentos e zonas desmilitarizadas

**Ataques.** Um inimigo a menos de `INTEREST_POINT_ATTACK_RANGE + EPSILON` do centro:

- causa `INTEREST_POINT_DAMAGE` de dano ao ativo (a vida nunca fica abaixo de 0);
- conta como um ataque bem-sucedido;
- é removido do episódio;
- destrói um radar terrestre, se ainda restar algum.

**Engajamentos.** Cada par amigo/inimigo a menos de `NEUTRALIZATION_RANGE` de distância sorteia um número aleatório *u*:

| Papel do amigo | Resultado |
|---|---|
| Interceptador | *u* ≤ P(amigo vivo): inimigo destruído · *u* ≤ P(amigo vivo) + P(inimigo vivo): amigo destruído · caso contrário: ambos destruídos |
| AEW ou radar | *u* ≤ P(ambos mortos): ambos destruídos (colisão) · caso contrário, nada acontece |

Aqui, P(amigo vivo) é `NEUTRALIZATION_PROB_FRIEND_ALIVE`, P(inimigo vivo) é `NEUTRALIZATION_PROB_ENEMY_ALIVE` e P(ambos mortos) é `NEUTRALIZATION_PROB_BOTH_DEAD`. Por padrão, esta última vale `1 - (FRIEND_ALIVE + ENEMY_ALIVE)`.

**Zonas desmilitarizadas.** Enquanto a camada DMZ estiver ativada (padrão; botão *Tog. DMZ*), os inimigos dentro de uma zona não podem ser engajados. As zonas que se sobrepõem à área protegida são descartadas.

## Recompensa, término e métricas

A recompensa de um passo é calculada depois dos ataques e dos engajamentos:

```
reward = Σ distance(enemy, centre) × 1e-4
       + 10 × enemies_shot_down × |centre| × 1e-4
       − 2000 × attacks_in_this_step
```

- `|centre|` é a distância da origem até o centro.
- Os fatores são os argumentos de `compute_reward()`.
- `accum_reward` é a soma das recompensas de todos os passos do episódio.

Um episódio termina quando não resta nenhum inimigo ou após `max_steps` passos (5000 por padrão).

As métricas retornadas em `info` estão listadas no [README](../../README.pt-BR.md#executando-experimentos) e na [referência da API](api.md#o-dicionário-info).

## Drones defeituosos

Os primeiros `BROKEN_COUNT` interceptadores são *defeituosos*:

- Após o sensoriamento, eles sobrescrevem a área de cobertura do sensor com um padrão esparso e aleatório de detecções falsas.
- Eles sorteiam um novo padrão a cada `UPDATE_STATE_BROKEN` chamadas.
- Eles compartilham os dados falsos como qualquer outra informação, o que permite estudar o efeito de nós defeituosos sobre a rede inteira.
- Cada drone defeituoso tem o seu próprio gerador de padrões, com semente própria.

## Aleatoriedade e reprodutibilidade

Toda fonte de aleatoriedade é inicializada a partir da semente (`seed`) do ambiente:

- os geradores de classe de `FriendDrone` (perda de mensagens) e de `EnemyDrone` (surgimento, padrões, reações), que recebem nova semente a cada `reset()`;
- o gerador próprio do ambiente (resultados dos engajamentos);
- os geradores individuais dos drones defeituosos.

Os timestamps de detecção usam um relógio lógico, e as listas de vizinhos têm uma ordem determinística. Consequentemente, a mesma configuração, a mesma semente e o mesmo tamanho de tela sempre produzem exatamente o mesmo episódio, independentemente da velocidade da máquina ou do modo de renderização. `tests/test_environment.py::test_same_seed_same_episode` verifica isso a cada alteração.

## Mapa do código

| Conceito | Onde |
|---|---|
| Laço de passos, ataques, engajamentos, recompensa, interface | `core/AirTrafficEnv.py` |
| Percepção, comunicação, movimento | `core/FriendDrone.py` |
| Padrões e reações dos inimigos | `core/EnemyDrone.py` |
| Políticas padrão, `BaseBehavior`, `BehaviorType` | `behaviorsDefault/behaviors.py` |
| Funções auxiliares da grade, matemática da interceptação | `core/utils.py` |
| Desenho | `core/rendering.py` |
| Carregamento da configuração | `core/settings.py` |
