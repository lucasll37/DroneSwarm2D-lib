# Referência de configuração

🇺🇸 [English version](../en/configuration.md) · [Voltar ao README](../../README.pt-BR.md)

Um cenário é um arquivo JSON passado para `DroneSwarm2D.init(path)`. Cada chave se torna um atributo do módulo de configurações:

```python
settings = DroneSwarm2D.init("config.json")
settings.SIM_WIDTH
```

Também é possível importar cada chave, depois de `init()`:

```python
from DroneSwarm2D.core.settings import SIM_WIDTH
```

## Como os valores são lidos

- **Números, booleanos e listas** são usados como estão.
- **Strings** são avaliadas como expressões Python sobre as chaves definidas **acima** delas no arquivo; por isso, a ordem importa: `"GRID_WIDTH": "SIM_WIDTH // CELL_SIZE"` deve vir depois de `SIM_WIDTH` e `CELL_SIZE`.
- Uma string que não pode ser avaliada continua sendo uma string (`"Courier New"`), e `"None"` se torna `None`.
- **Os itens de listas** são avaliados individualmente, como em `"INTEREST_POINT_CENTER": ["SIM_WIDTH / 2", "SIM_HEIGHT / 2"]`.
- `FULL_WIDTH` / `FULL_HEIGHT` sempre vêm do tamanho da tela, ou de `init(..., screen_size=(w, h))`; no modo sem janela (headless), o padrão é 1920×1080. Qualquer valor definido no arquivo para essas duas chaves é ignorado.
- `INTEREST_POINT_CENTER`, `CENTER` e `TYPE_OF_SCENARIO` (o nome do arquivo) são recalculados após o carregamento.

> ⚠️ Os valores são avaliados com `eval`. Carregue apenas arquivos de configuração em que você confia.

A configuração é carregada **uma vez por processo**. Chamar `init()` novamente com outro arquivo emite um aviso e mantém a primeira configuração.

## Chaves

A coluna *Exemplo* mostra o valor usado em `example/src/config.json`. As distâncias são dadas em pixels e as velocidades, em pixels por passo.

### Forças

| Chave | Exemplo | Descrição |
|---|---|---|
| `FRIEND_COUNT` | `8` | Número total de drones amigos (radares + AEW + interceptadores) |
| `RADAR_COUNT` | `0` | Radares terrestres fixos, incluídos em `FRIEND_COUNT` |
| `AEW_COUNT` | `0` | Drones de alerta aéreo antecipado (AEW), incluídos em `FRIEND_COUNT` |
| `BROKEN_COUNT` | `0` | Quantos interceptadores têm sensores defeituosos que transmitem dados aleatórios |
| `ENEMY_COUNT` | `8` | Número de drones atacantes (incluindo o controlado por joystick, se houver) |

### Sensores

| Chave | Exemplo | Descrição |
|---|---|---|
| `FRIEND_DETECTION_RANGE` | `250` | Alcance do sensor dos interceptadores |
| `AEW_DETECTION_RANGE` | `200` | Alcance do sensor dos drones AEW |
| `RADAR_DETECTION_RANGE` | `350` | Alcance do sensor dos radares |
| `ENEMY_DETECTION_RANGE` | `100` | Distância na qual um inimigo percebe um drone amigo (e reage) |
| `PASSIVE_DETECTION` | `true` | Inicia no modo de detecção passiva (triangulação); pode ser alternado na interface |
| `N_LINE_SIGHT_CROSSING` | `3` | Número de sensores que precisam cobrir um inimigo para uma detecção passiva |
| `TRIANGULATION_GRANULARITY` | `32` | Subcélulas por célula da grade de linhas de visada |

### Comunicação

| Chave | Exemplo | Descrição |
|---|---|---|
| `COMMUNICATION_RANGE` | `250` | Distância máxima de um enlace; os amigos também são detectados dentro desse alcance |
| `N_CONNECTIONS` | `3` | Cada drone se conecta aos seus N drones mais próximos (os enlaces são simétricos) |
| `CICLE_COMM_BY_STEP` | `3` | Ciclos de comunicação por passo (cada um é uma oportunidade de retransmissão) |
| `MESSAGE_LOSS_PROBABILITY` | `0.1` | Probabilidade de uma troca ser perdida |
| `MIN_COMMUNICATION_HOLD` | `3` | Amigos detectados recentemente ao alcance necessários para manter a posição; abaixo disso, o drone se reagrupa. `0` faz os drones ociosos voltarem para `INITIAL_DISTANCE` |

### Grade de percepção

| Chave | Exemplo | Descrição |
|---|---|---|
| `CELL_SIZE` | `20` | Tamanho da célula em pixels |
| `GRID_WIDTH` / `GRID_HEIGHT` | `"SIM_WIDTH // CELL_SIZE"` / `"SIM_HEIGHT // CELL_SIZE"` | Tamanho da grade em células |
| `DECAY_FACTOR` | `0.99` | Multiplicador aplicado às matrizes de intensidade a cada passo |
| `PLOT_THRESHOLD` | `0.05` | Células com intensidade abaixo deste valor aparecem em branco nos gráficos 3-D |

### Tela e geometria

| Chave | Exemplo | Descrição |
|---|---|---|
| `FULL_WIDTH` / `FULL_HEIGHT` | tamanho da tela | Tamanho da janela (veja acima) |
| `SIM_WIDTH` / `SIM_HEIGHT` | `"int(FULL_WIDTH * 0.7)"` / `"FULL_HEIGHT"` | Área de simulação |
| `GRAPH_WIDTH` / `GRAPH_HEIGHT` | `"FULL_WIDTH - SIM_WIDTH"` / `"FULL_HEIGHT"` | Painel de gráficos e de controle |
| `INTEREST_POINT_CENTER` | `["SIM_WIDTH / 2", "SIM_HEIGHT / 2"]` | Recalculado como o centro da área de simulação |
| `INTERNAL_RADIUS` | `"min(SIM_WIDTH, SIM_HEIGHT) / 10"` | Área protegida |
| `EXTERNAL_RADIUS` | `"INTERNAL_RADIUS * 4"` | Os drones amigos nunca se afastam do centro mais do que isso |
| `INITIAL_DISTANCE` | `"INTERNAL_RADIUS * 1.4"` | Raio do primeiro anel de interceptadores; dentro dele, os inimigos ficam totalmente agressivos |
| `INITIAL_N_LAYERS` | `1` | Número de anéis em que os interceptadores começam (separados por 40 px) |
| `RADAR_RANGE` | `0` | Distância do centro em que os radares são posicionados |
| `AEW_RANGE` | `350` | Raio da órbita dos drones AEW (e onde eles começam) |
| `THRESHOLD_PROJECTION` | `"INTERNAL_RADIUS * 0.5"` | Distância máxima que um drone ocioso percorre para se posicionar na linha de aproximação de um inimigo |
| `EPSILON` | `2` | Tolerância de distância (chegada, ponto de interceptação) |
| `DMZ` | `[["SIM_WIDTH * 0.35", "SIM_HEIGHT * 0.30", 60], ...]` | Zonas desmilitarizadas (DMZ) no formato `[x, y, radius]` |
| `GEO_TOP_LEFT` / `GEO_BOTTOM_RIGHT` | `[-74.0, 40.8]` / `[-73.9, 40.7]` | Longitude/latitude dos cantos da área (exportação para o Tacview) |

### Movimento

| Chave | Exemplo | Descrição |
|---|---|---|
| `BASE_SPEED` | `2.0` | Valor de conveniência referenciado pelas velocidades abaixo |
| `FRIEND_SPEED` | `"BASE_SPEED"` | Velocidade dos interceptadores |
| `AEW_SPEED` | `"FRIEND_SPEED"` | Velocidade dos drones AEW |
| `ENEMY_SPEED` | `"BASE_SPEED"` | Velocidade dos inimigos |
| `DT_STEP` | `0.6` | Segundos simulados por passo (relógio e timestamps (carimbos de tempo) do Tacview) |

### Ponto de interesse e combate

| Chave | Exemplo | Descrição |
|---|---|---|
| `INTEREST_POINT_INITIAL_HEALTH` | `100` | Vida inicial do ativo protegido |
| `INTEREST_POINT_DAMAGE` | `"INTEREST_POINT_INITIAL_HEALTH // ENEMY_COUNT"` | Dano por ataque bem-sucedido |
| `INTEREST_POINT_ATTACK_RANGE` | `"INTERNAL_RADIUS"` | Um inimigo a esta distância do centro atinge o ativo |
| `NEUTRALIZATION_RANGE` | `20` | Um amigo e um inimigo a esta distância entram em combate |
| `NEUTRALIZATION_PROB_FRIEND_ALIVE` | `0.5` | O interceptador vence (inimigo destruído) |
| `NEUTRALIZATION_PROB_ENEMY_ALIVE` | `0.2` | O inimigo vence (interceptador destruído) |
| `NEUTRALIZATION_PROB_BOTH_DEAD` | `"1 - (... + ...)"` | Ambos destruídos; também é a probabilidade de colisão para AEW e radares |

### Inimigos

| Chave | Exemplo | Descrição |
|---|---|---|
| `ENEMY_BEHAVIOR` | `"None"` | Padrão para todos os inimigos (veja a [arquitetura](architecture.md#inimigos)); `"None"` sorteia um por inimigo |
| `INITIAL_AGGRESSIVENESS` | `0.5` | Probabilidade de ataque quando detectado em `EXTERNAL_RADIUS` (cresce até 1 em `INITIAL_DISTANCE`) |
| `ESCAPE_STEPS` | `40` | Duração de uma fuga |

### Falhas, interação e exibição

| Chave | Exemplo | Descrição |
|---|---|---|
| `UPDATE_STATE_BROKEN` | `100` | Um drone defeituoso sorteia um novo padrão falso após este número de chamadas |
| `HOLD_SPREAD` | `true` | Os interceptadores ociosos se espalham ao redor do centro |
| `JOYSTICK` | `"None"` | `"Friend"`: um gamepad pilota o interceptador líder; `"Enemy"`: ele pilota um inimigo, e os amigos só ficam visíveis perto dos inimigos |
| `FONT_FAMILY` | `"Courier New"` | Fonte da interface (usa a fonte padrão do pygame se ela não estiver disponível) |

### Chaves livres

`EPISODES`, `ANALYSIS_EPISODES` e `TARGET_INFLUENCE` não são usadas pela biblioteca. Seus scripts podem lê-las, por exemplo como `settings.EPISODES`, e você pode adicionar qualquer outra chave de que precisar.
