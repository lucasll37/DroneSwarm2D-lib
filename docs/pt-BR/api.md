# Referência da API

🇺🇸 [English version](../en/api.md) · [Voltar ao README](../../README.pt-BR.md)

Apenas os nomes listados aqui são considerados públicos. Todo o resto (nomes que começam com `_`) pode mudar sem aviso.

## Sumário

- [Pacote: `DroneSwarm2D`](#pacote-droneswarm2d)
- [Ambiente: `AirTrafficEnv`](#ambiente-airtrafficenv)
- [O dicionário `info`](#o-dicionário-info)
- [Comportamentos: `DroneSwarm2D.behaviorsDefault`](#comportamentos-droneswarm2dbehaviorsdefault)
- [Drones e cenário](#drones-e-cenário)
- [Utilitários: `DroneSwarm2D.core.utils`](#utilitários-droneswarm2dcoreutils)
- [Desenho: `DroneSwarm2D.core.rendering`](#desenho-droneswarm2dcorerendering)
- [Configurações: `DroneSwarm2D.core.settings`](#configurações-droneswarm2dcoresettings)
- [Ferramenta de vídeo e linha de comando](#ferramenta-de-vídeo-e-linha-de-comando)
- [Nomes obsoletos](#nomes-obsoletos)

## Pacote: `DroneSwarm2D`

### `init(config_path=None, fullscreen=True, headless=False, screen_size=None)`

Carrega o cenário e torna as classes abaixo disponíveis como atributos do pacote. Deve ser chamada antes de usar o resto da biblioteca; chamadas posteriores retornam as configurações que já estão carregadas.

| Argumento | Significado |
|---|---|
| `config_path` | Arquivo JSON; se `None`, o primeiro `*.json` encontrado em um diretório `config/` |
| `fullscreen` | Janela em tela cheia (ignorado quando `headless`) |
| `headless` | Sem janela (driver de vídeo dummy do SDL), para servidores, CI e execuções em lote |
| `screen_size` | `(width, height)` que substitui o tamanho da tela; padrão `(1920, 1080)` quando `headless` |

Retorna o módulo `DroneSwarm2D.core.settings` e lança `FileNotFoundError` se o arquivo não existir.

**Outros nomes do pacote:**

- `is_initialized()` informa se `init` já foi chamada.
- `__version__` é a string da versão.
- Disponíveis depois de `init`:
  - `AirTrafficEnv`, `FriendDrone`, `EnemyDrone`;
  - `CircleInterestPoint` (alias `InterestPoint`), `CircleDMZ` (alias `DemilitarizedZone`);
  - `generate_sparse_matrix`, `get_friends_hold`, `create_video_from_frames`.

## Ambiente: `AirTrafficEnv`

```python
AirTrafficEnv(max_steps=5000, mode="human", friend_behavior=None, friend_aew_behavior=None,
              friend_radar_behavior=None, enemy_behavior=None, demilitarized_zones=None, seed=None)
```

| Argumento | Significado |
|---|---|
| `max_steps` | Duração máxima do episódio |
| `mode` | `"human"` desenha todos os passos (limitado a `render_fps`, 30 por padrão); qualquer outro valor, como `None`, executa sem janela (headless) |
| `friend_behavior` / `friend_aew_behavior` / `friend_radar_behavior` | Instâncias de comportamento para interceptadores, drones AEW e radares (padrões: `FriendCommonBehaviorDefault`, `FriendAEWBehaviorDefault`, `FriendRadarBehaviorDefault`) |
| `enemy_behavior` | Nome do padrão dos inimigos (veja `EnemyDrone.ENEMY_BEHAVIORS`) ou `None` para um padrão aleatório por inimigo; normalmente `settings.ENEMY_BEHAVIOR` |
| `demilitarized_zones` | `[(x, y, radius), ...]`, normalmente `settings.DMZ`; zonas que se sobrepõem à área protegida são ignoradas |
| `seed` | Semente (seed) aleatória (aleatória se `None`); a mesma semente sempre gera o mesmo episódio |

**Métodos:**

| Método | Descrição |
|---|---|
| `reset() -> (obs, done)` | Inicia um novo episódio |
| `step(action=None) -> (obs, reward, done, info)` | Avança um passo; `action` é ignorada (os drones agem por meio de seus comportamentos) |
| `render(mode="human")` | Desenha um quadro; chamado por `step` |
| `compute_reward(normilizer_factor=1e-4, bonus_factor=10, penalty_factor=2000)` | Recompensa do passo atual |
| `close()` | Encerra o pygame |
| `handle_tacview()` | Grava um CSV do Tacview por drone em `./output/tacview_export_<date>/` |
| `plot_individual_states() -> Surface` | Gráfico 3D da percepção do drone selecionado |
| `is_in_demilitarized_zone(pos) -> bool` | Se um ponto está dentro de uma DMZ |

`obs` é um vetor `float32` com `[x, y, vx, vy]` de cada drone amigo, seguido de cada inimigo. Seu comprimento muda à medida que drones são destruídos.

**Alternadores da interface.** Cada método `toggle_*` inverte o atributo booleano indicado ao lado:

| Método | Atributo |
|---|---|
| `toggle_graph` | `show_graph` |
| `toggle_pause` | `paused` |
| `toggle_friend_range` | `show_friend_detection_range` |
| `toggle_enemy_range` | `show_enemy_detection_range` |
| `toggle_dashed` | `show_dashed_lines` |
| `toggle_dmz` | `show_dmz`, que também ativa a regra da DMZ |
| `toggle_comm_range` | `show_friend_comm_range` |
| `toggle_tacview_export` | `export_to_tacview` |
| `toggle_save_frames` | `save_frames` |
| `toggle_target_lines` | `show_target_lines` |
| `toggle_trajectory` | `show_trajectory` |
| `toggle_debug` | `show_debug` |
| `toggle_triangulation` | `use_triangulation` |
| `toggle_return` | `return_to_base` |

`reset_simulation()` e `exit_env()` implementam os botões *Reset* e *Exit*.

**Principais atributos:**

- `friend_drones`, `enemy_drones` (os drones vivos), `interest_point`, `demilitarized_zones`;
- `episode`, `current_step`, `accum_reward`;
- `enemies_neutralized`, `friends_lost`, `friends_returned`, `sucessful_attacks`;
- `selected_drone`, `leader`, `seed`, `render_fps`.

**Função do módulo.** `compute_global_matrices(drones, "enemy_intensity", "enemy_direction")` funde a percepção de vários drones: em cada célula, prevalece a detecção mais forte.

## O dicionário `info`

| Chave | Tipo | Significado |
|---|---|---|
| `current_step` | `int` | Passos decorridos no episódio |
| `accum_reward` | `float` | Soma das recompensas de cada passo |
| `enemies_shotdown` | `int` | Inimigos destruídos em engajamentos |
| `friends_shotdown` | `int` | Drones amigos destruídos (em engajamentos, mais os radares destruídos por ataques) |
| `sucessful_attacks` | `int` | Inimigos que alcançaram o ponto de interesse |
| `interest_point_health` | `int` | Vida restante do ativo |
| `state_percentages` | `dict[str, float]` | Fração média (%) dos passos que os drones vivos passaram em cada rótulo de comportamento |
| `total_distance_traveled` | `float` | Soma das distâncias percorridas pelos drones vivos |
| `friends_returned` | `int` | Drones que pousaram com *retorno à base* |
| `enemies_remaining` / `friends_remaining` | `int` | Drones ainda vivos |

## Comportamentos: `DroneSwarm2D.behaviorsDefault`

| Nome | Descrição |
|---|---|
| `BehaviorType` | `StrEnum` com `COMMON`, `AEW`, `RADAR` e `AI`; os membros são considerados iguais aos seus nomes em comparações |
| `BaseBehavior(behavior_type)` | Classe base abstrata; implemente `apply(state, joystick_controlled=False) -> (info, velocity)`. O atributo `type` guarda o papel como string |
| `FriendCommonBehaviorDefault(friend_activation_threshold_position=0.7, enemy_activation_threshold_position=0.4)` | Atribuição gulosa + interceptação + estados de espera (veja a [arquitetura](architecture.md#a-política-padrão-dos-interceptadores)) |
| `FriendAEWBehaviorDefault(...)` | Órbita com raio `AEW_RANGE` |
| `FriendRadarBehaviorDefault(...)` | Permanece parado |

Veja o [guia de comportamentos](behaviors.md) para o dicionário `state` e exemplos.

## Drones e cenário

### `FriendDrone(interest_point_center, position, behavior=None, fixed=False, broken=False)`

**Atributos:**

- movimento: `pos`, `vel`, `trajectory`, `distance_traveled`;
- identidade e papel: `drone_id`, `behavior`, `behavior_type` (propriedade), `fixed`, `broken`, `is_leader`, `selected`;
- comunicação: `neighbors`;
- percepção: `enemy_intensity`, `enemy_direction`, `enemy_timestamp`, `friend_intensity`, `friend_direction`, `friend_timestamp`, `passive_detection_matrix` (construída sob demanda), `direction_vectors`;
- saída do comportamento: `info`, `state_history`.

**Métodos:**

| Método | Descrição |
|---|---|
| `update(enemy_drones, friend_drones, use_triangulation, return_to_base=False)` | Um passo completo |
| `get_state() -> dict` | O `state` passado ao comportamento |
| `get_state_percentages() -> dict` | Fração dos passos por rótulo de comportamento |
| `update_neighbors(all_drones)` | Escolhe os enlaces de comunicação |
| `communication(all_drones)` | Mescla as matrizes com as dos vizinhos |
| `merge_enemy_matrix(other)` / `merge_friend_matrix(other)` | Copia as células em que `other` é mais recente |
| `decay_matrices()` | Aplica `DECAY_FACTOR` às intensidades |
| `update_local_enemy_detection(friends, enemies)` / `update_local_friend_detection(friends)` | Sensoriamento |
| `update_passive_detection(enemies)` | Traça as linhas de visada |
| `draw(surface, show_detection=True, show_comm_range=True, show_trajectory=False, show_debug=False)` | Desenha o drone |

**Métodos de classe.** `FriendDrone.set_class_seed(seed=None)` define a semente do gerador compartilhado e `FriendDrone.reset_clock()` reinicia o relógio lógico. O ambiente chama ambos a cada `reset()`.

### `EnemyDrone(interest_point_center, position=None, behavior_type=None, fixed=False)`

- **Constantes.** `ENEMY_BEHAVIORS` lista todos os nomes de padrões; `RANDOM_ENEMY_BEHAVIORS` lista os que são sorteados quando `behavior_type` é `None`. Ambas são definidas em `DroneSwarm2D.core.EnemyDrone`.
- **Atributos:** `pos`, `vel`, `drone_id`, `behavior_type`, `aggressiveness`, `detector`, `desperate_attack`, `escape_steps_count`, `start_delay`, `trajectory`, `info`.
- **Métodos:** `update(detector=None)`, `draw(surface, show_detection=True, show_trajectory=False, show_debug=False)` e `EnemyDrone.set_class_seed(seed=None)`.

### `CircleInterestPoint(center, internal_radius, external_radius, color=(0, 255, 0), seed=None)`

Atributos `center`, `internal_radius`, `external_radius`, `health` e `health_ratio` (propriedade); método `draw(surface)`.

### `CircleDMZ(center, radius, color=(0, 0, 255), seed=None)`

Atributos `center`, `radius`, `dmz_id`; métodos `contains(pos)` e `draw(surface)`.

## Utilitários: `DroneSwarm2D.core.utils`

| Função | Descrição |
|---|---|
| `pos_to_cell(pos, cell_size=CELL_SIZE, grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT)` | Posição em pixels → célula `(i, j)` (limitada à grade) |
| `cell_center(cell, cell_size=CELL_SIZE)` | Célula → centro em pixels |
| `active_cells(matrix, threshold)` | Células com valor ≥ limiar, em ordem de linhas (row-major) |
| `intercept_direction(chaser_pos, chaser_speed, target_pos, target_vel)` | Velocidade de interceptação (perseguição pura se a interceptação for impossível) |
| `can_intercept(chaser_pos, chaser_speed, target_pos, target_vel, point_of_interest=None)` | Se a interceptação ocorre antes de o alvo alcançar o ponto |
| `get_friends_hold(state, friend_activation_threshold_position=0.7, enemy_activation_threshold_position=0.4)` | Células de amigos que ficaram sem atribuição em uma alocação gulosa |
| `generate_sparse_matrix(shape, max_nonzero=20, seed=None)` | Intensidades esparsas aleatórias mais direções unitárias |
| `smooth_matrix_with_kernel_10x10(matrix, direction, sigma=1.0, flat_radius=1.0)` | Picos gaussianos ("bumps") em torno das células positivas (gráficos) |
| `symmetrical_flat_topped_gaussian_10x10(value, sigma, flat_radius)` | O kernel 10×10 |
| `sim_to_geo(x, y)` | Pixels → `(longitude, latitude)` usando `GEO_TOP_LEFT` / `GEO_BOTTOM_RIGHT` |
| `load_svg_as_surface(path)` | SVG → superfície do pygame |
| `plot_individual_states_matplotlib(state)` | Gráfico 3D interativo de um `state` (depuração) |

`draw_dashed_circle` e `draw_dashed_line` são reexportadas de `rendering`.

## Desenho: `DroneSwarm2D.core.rendering`

- `get_font(size, bold=False)` retorna uma fonte em cache; `clear_caches()` descarta os objetos em cache.
- Funções auxiliares de desenho:
  - `draw_dashed_circle(surface, color, center, radius, dash_length=5, space_length=5, width=1)`;
  - `draw_dashed_line(surface, color, start, end, width=1, dash_length=5, space_length=5)`;
  - `draw_fading_trajectory(surface, points, color, decay_rate, min_alpha=30, width=2)`;
  - `draw_grid(surface)`, `draw_heatmap(surface, intensity, base_color)`, `draw_direction(surface, intensity, direction, threshold)`, `draw_friend_communication(surface, drones, show_dashed=True)`.
- `Button(rect, text, callback, toggled=None, color=(70, 70, 70))` é o botão do painel de controle.

As funções auxiliares de desenho e `Button` também podem ser importadas de `DroneSwarm2D.core.AirTrafficEnv`, como antes.

## Configurações: `DroneSwarm2D.core.settings`

- `initialize(config_path=None, fullscreen=True, headless=False, screen_size=None)` carrega a configuração (chamada por `DroneSwarm2D.init`).
- `get_config()` retorna o `ConfigManager`, que oferece `get(key, default)`, `config[key]`, `key in config`, `keys()`, `to_dict()`, além dos atributos `config_path`, `config_name`, `screen`, `clock` e `headless`.
- `is_initialized()` informa se a configuração já foi carregada.
- Cada chave de configuração, além de `screen` e `clock`, é um atributo do módulo. Ler qualquer uma delas antes da inicialização lança um `RuntimeError` que explica o que fazer.

## Ferramenta de vídeo e linha de comando

- `create_video_from_frames(frames_dir, output_dir="./video", fps=30, codec="libx264", remove_frames=False)` codifica os arquivos `frame_<n>.png` em `<output_dir>/<scenario>.mp4` e retorna o caminho, ou `None` quando não há nada para codificar.
- `get_frame_number(filename)` extrai o número do quadro de um nome de arquivo.

| Comando | Descrição |
|---|---|
| `python -m DroneSwarm2D` | O mesmo que `init` |
| `python -m DroneSwarm2D init [--name NAME] [--template TEMPLATE]` | Cria um projeto |
| `python -m DroneSwarm2D create_video PATH [--output-dir DIR] [--fps N] [--codec C] [--remove-frames]` | Codifica os quadros salvos |
| `python -m DroneSwarm2D --version` | Exibe a versão |

O comando de console `droneswarm2d` é equivalente a `python -m DroneSwarm2D`.

## Nomes obsoletos

Estes nomes continuam funcionando e serão mantidos por retrocompatibilidade:

| Obsoleto | Use no lugar |
|---|---|
| `FriendDrone.update_neghbors` | `update_neighbors` |
| `AirTrafficEnv.toogle_save_frames` | `toggle_save_frames` |
| `AirTrafficEnv.toggle_trajetory` | `toggle_trajectory` |
| `AirTrafficEnv.toogle_return` | `toggle_return` |
| `AirTrafficEnv.toogle_triangulation` | `toggle_triangulation` |

A chave de `info` `sucessful_attacks` (com erro de grafia) e a chave de configuração `CICLE_COMM_BY_STEP` fazem parte da interface estável e mantêm seus nomes.
