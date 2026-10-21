# DroneSwarm2D

**Um simulador 2D para estudar táticas defensivas distribuídas com enxames de drones autônomos.**

[![tests](https://github.com/lucasll37/DroneSwarm2D-lib/actions/workflows/tests.yml/badge.svg)](https://github.com/lucasll37/DroneSwarm2D-lib/actions/workflows/tests.yml)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)

🇺🇸 [Read in English](README.md)

![Banner do enxame de drones](./images/banner.gif)

O DroneSwarm2D simula um enxame de drones defensivos que protege um ativo (o *ponto de interesse*) contra drones atacantes de baixo custo. Não há controlador central: cada drone defensivo monta sua própria visão do campo de batalha a partir dos seus sensores e do que os vizinhos lhe informam por uma rede ad hoc sujeita a perdas, e então decide sozinho o que fazer. A biblioteca foi desenvolvida como trabalho de graduação no ITA (Instituto Tecnológico de Aeronáutica).

## Sumário

- [Recursos](#recursos)
- [Instalação](#instalação)
- [Início rápido](#início-rápido)
- [Como funciona](#como-funciona)
- [Escrevendo seu próprio comportamento](#escrevendo-seu-próprio-comportamento)
- [Configuração](#configuração)
- [Executando experimentos](#executando-experimentos)
- [Interface interativa](#interface-interativa)
- [Desempenho](#desempenho)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Desenvolvimento](#desenvolvimento)
- [Motivação e resultados da pesquisa](#motivação-e-resultados-da-pesquisa)
- [Licença e contato](#licença-e-contato)

**Documentação:** [arquitetura](docs/pt-BR/architecture.md) · [referência de configuração](docs/pt-BR/configuration.md) · [guia de comportamentos](docs/pt-BR/behaviors.md) · [referência da API](docs/pt-BR/api.md) · [changelog](CHANGELOG.md)

## Recursos

- **Tomada de decisão distribuída.** Cada drone executa sua própria política sobre sua própria percepção; ninguém enxerga o campo de batalha inteiro.
- **Comunicação ad hoc.** Enlaces com os *N* drones mais próximos dentro do alcance, enlaces simétricos, perda probabilística de mensagens e retransmissões.
- **Percepção local.** Cada drone guarda o que sabe em matrizes de grade com a *recência* e a *direção* das detecções, que se esvaem com o tempo e são mescladas pelo timestamp.
- **Detecção direta ou passiva.** O modo passivo só confirma um alvo visto por vários sensores (triangulação). É possível injetar drones defeituosos que transmitem dados espúrios.
- **Vários papéis de drone.** Interceptadores móveis, drones de alerta aéreo antecipado (AEW) e radares terrestres fixos. Os inimigos usam 11 padrões de aproximação sorteados (linhas retas, zigue-zagues, espirais, ricochetes...), além de ataques em formação, e reagem quando são detectados.
- **Zonas desmilitarizadas (DMZ)**, onde nenhum engajamento pode ocorrer.
- **Comportamentos plugáveis.** Uma nova política defensiva é uma pequena classe Python.
- **Pensado para experimentos.** API `reset` / `step` no estilo do Gym, execuções totalmente reprodutíveis a partir de uma semente (seed) e um modo sem janela (headless) para experimentos em lote e CI.
- **Visualização rica.** Interface interativa em pygame, gráficos 3D da percepção de um drone, gravação de quadros em vídeo e exportação para o Tacview.

## Instalação

O DroneSwarm2D requer **Python 3.12+** e a biblioteca gráfica **Cairo**, usada para rasterizar os sprites SVG.

| Sistema | Cairo |
|---|---|
| Debian / Ubuntu | `sudo apt install libcairo2` |
| macOS | `brew install cairo` |
| conda (qualquer SO) | `conda install -c conda-forge cairo` |

Instale a biblioteca a partir do GitHub:

```bash
pip install "DroneSwarm2D @ git+https://github.com/lucasll37/DroneSwarm2D-lib.git"
```

Extras opcionais:

- `pip install "DroneSwarm2D[analysis] @ git+..."` adiciona pandas, seaborn, SciPy, scikit-learn, OpenCV e Jupyter para análise de dados.
- `pip install "DroneSwarm2D[dev] @ git+..."` adiciona as ferramentas de teste e de lint.

Para trabalhar na própria biblioteca, instale-a a partir de um clone:

```bash
git clone https://github.com/lucasll37/DroneSwarm2D-lib.git
cd DroneSwarm2D-lib
pip install -e ".[dev]"      # ou: make create_env  (ambiente conda em ./.venv com versões fixadas)
```

## Início rápido

Crie um projeto inicial e execute-o:

```bash
python -m DroneSwarm2D init          # ou simplesmente: droneswarm2d init
cd droneSwarm2D-project
python src/main.py --windowed        # janela interativa (omita --windowed para tela cheia)
python src/main.py --headless --episodes 20 --seed 7   # sem janela, o mais rápido possível
```

O projeto contém `src/config.json` (o cenário), `src/behaviors.py` (as políticas dos seus drones), `src/main.py` (o laço de episódios) e um tutorial.

Um script mínimo fica assim:

```python
import DroneSwarm2D

# 1. Carregue o cenário primeiro: o resto da biblioteca o lê no momento da importação.
settings = DroneSwarm2D.init("config.json", headless=True)

# 2. Crie o ambiente. mode="human" abre a interface, mode=None executa em modo headless.
env = DroneSwarm2D.AirTrafficEnv(
    mode=None,
    seed=42,
    enemy_behavior=settings.ENEMY_BEHAVIOR,
    demilitarized_zones=settings.DMZ,
)

# 3. Execute um episódio.
obs, done = env.reset()
while not done:
    obs, reward, done, info = env.step(None)

print(info["enemies_shotdown"], info["sucessful_attacks"], info["interest_point_health"])
env.close()
```

> **Importante:** chame `DroneSwarm2D.init()` antes de importar `DroneSwarm2D.behaviorsDefault`, `DroneSwarm2D.core.utils` ou os seus próprios comportamentos. Esses módulos leem a configuração no momento em que são importados.

## Como funciona

Cada chamada a `env.step()` avança o mundo em um passo de simulação:

1. **Drones amigos** (nesta ordem):
   - *esquecer*: suas matrizes de intensidade decaem pelo fator `DECAY_FACTOR`;
   - *conectar*: escolhem seus vizinhos de comunicação;
   - *perceber*: detectam inimigos (diretamente, ou apenas quando `N_LINE_SIGHT_CROSSING` sensores os cobrem) e amigos;
   - *conversar*: mesclam suas matrizes com as dos vizinhos `CICLE_COMM_BY_STEP` vezes, e cada mensagem pode ser perdida;
   - *agir*: pedem uma velocidade ao seu **comportamento** e se movem.
2. **Inimigos** se movem segundo o seu padrão. Quando um amigo chega a menos de `ENEMY_DETECTION_RANGE`, o inimigo ou parte para um *ataque desesperado* em linha reta, ou *foge* por algum tempo; quanto mais agressivo ele é, maior a chance de atacar, e a agressividade aumenta à medida que ele se aproxima do alvo.
3. **Ataques.** Um inimigo que alcança o ponto de interesse causa dano a ele e é removido. Cada acerto também destrói um radar terrestre.
4. **Engajamentos.** Todo par amigo/inimigo a uma distância menor que `NEUTRALIZATION_RANGE` entra em combate. As configurações `NEUTRALIZATION_PROB_*` decidem se o inimigo, o amigo ou ambos são destruídos. Inimigos dentro de uma zona desmilitarizada ativa não podem ser engajados.
5. **Recompensa.** A recompensa cresce com a distância dos inimigos ao alvo e com os inimigos abatidos, e é penalizada a cada ataque bem-sucedido.

A percepção dos drones defensivos fica em uma grade de células de `CELL_SIZE` pixels. Uma detecção ajusta a *intensidade* da sua célula para 1, e essa intensidade então decai por `DECAY_FACTOR` a cada passo. Quando dois drones se comunicam, cada célula é copiada de quem a viu mais recentemente. O modelo completo está descrito em [docs/pt-BR/architecture.md](docs/pt-BR/architecture.md).

## Escrevendo seu próprio comportamento

Um comportamento recebe apenas a percepção **local** do drone e retorna um rótulo e uma velocidade:

```python
# Executado após DroneSwarm2D.init(...)
import numpy as np
import pygame

import DroneSwarm2D
from DroneSwarm2D.behaviorsDefault import BaseBehavior, BehaviorType
from DroneSwarm2D.core import settings
from DroneSwarm2D.core.utils import active_cells, cell_center, intercept_direction


class ChaseClosestThreat(BaseBehavior):
    def __init__(self):
        super().__init__(BehaviorType.COMMON)

    def apply(self, state, joystick_controlled=False):
        pos = pygame.math.Vector2(*np.squeeze(state["pos"]).tolist())
        enemies = active_cells(np.squeeze(state["enemy_intensity"]), threshold=0.4)
        if not enemies:
            return ("IDLE", None, None, None), pygame.math.Vector2(0, 0)

        # Mira o inimigo mais próximo do ponto de interesse.
        cell = min(enemies, key=lambda c: cell_center(c).distance_to(settings.INTEREST_POINT_CENTER))
        heading = pygame.math.Vector2(*np.squeeze(state["enemy_direction"])[cell])
        velocity = intercept_direction(pos, settings.FRIEND_SPEED, cell_center(cell), heading * settings.ENEMY_SPEED)
        return ("PURSUING", cell_center(cell), None, None), velocity


env = DroneSwarm2D.AirTrafficEnv(friend_behavior=ChaseClosestThreat(), ...)
```

O [guia de comportamentos](docs/pt-BR/behaviors.md) explica o dicionário `state` e as funções auxiliares, percorre a política padrão (atribuição gulosa, interceptação e estados de "hold") e traz mais exemplos.

## Configuração

Um cenário é um arquivo JSON. Um valor do tipo string é avaliado como uma expressão Python sobre as chaves definidas acima dele:

```json
{
  "FRIEND_COUNT": 8,
  "ENEMY_COUNT": 8,
  "SIM_WIDTH": "int(FULL_WIDTH * 0.7)",
  "INTERNAL_RADIUS": "min(SIM_WIDTH, SIM_HEIGHT) / 10",
  "EXTERNAL_RADIUS": "INTERNAL_RADIUS * 4",
  "ENEMY_BEHAVIOR": "None",
  "DMZ": [["SIM_WIDTH * 0.35", "SIM_HEIGHT * 0.30", 60]]
}
```

| Grupo | Chaves principais |
|---|---|
| Forças | `FRIEND_COUNT`, `ENEMY_COUNT`, `RADAR_COUNT`, `AEW_COUNT`, `BROKEN_COUNT` |
| Sensores e enlaces | `FRIEND_DETECTION_RANGE`, `AEW_DETECTION_RANGE`, `RADAR_DETECTION_RANGE`, `ENEMY_DETECTION_RANGE`, `COMMUNICATION_RANGE`, `N_CONNECTIONS`, `CICLE_COMM_BY_STEP`, `MESSAGE_LOSS_PROBABILITY` |
| Percepção | `CELL_SIZE`, `DECAY_FACTOR`, `PASSIVE_DETECTION`, `N_LINE_SIGHT_CROSSING` |
| Geometria | `INTERNAL_RADIUS`, `EXTERNAL_RADIUS`, `INITIAL_DISTANCE`, `RADAR_RANGE`, `AEW_RANGE`, `DMZ` |
| Combate | `NEUTRALIZATION_RANGE`, `NEUTRALIZATION_PROB_*`, `INTEREST_POINT_DAMAGE`, `INITIAL_AGGRESSIVENESS`, `ESCAPE_STEPS` |
| Inimigos e controle | `ENEMY_BEHAVIOR` (padrão de ataque, ou `"None"` para aleatório), `JOYSTICK` (`"Friend"`, `"Enemy"` ou `"None"`) |

Todas as chaves estão documentadas na [referência de configuração](docs/pt-BR/configuration.md). A configuração é avaliada com `eval`, portanto carregue apenas arquivos em que você confia.

## Executando experimentos

- **Reprodutibilidade.** A mesma configuração, a mesma semente e o mesmo tamanho de tela sempre produzem o mesmo episódio. Como os tamanhos derivam da tela, passe `screen_size=(1920, 1080)` para `init` para tornar os resultados independentes do monitor.
- **Execuções headless.** `DroneSwarm2D.init(config, headless=True)` junto com `AirTrafficEnv(mode=None)` executa sem janela, tão rápido quanto a CPU permitir. Use esse modo em servidores e em experimentos em lote.
- **Métricas.** `step()` retorna um dicionário `info`:

| Chave | Significado |
|---|---|
| `current_step`, `accum_reward` | Progresso do episódio |
| `enemies_shotdown` | Inimigos destruídos em combate |
| `friends_shotdown` | Drones amigos destruídos (em combate, ou radares atingidos por um ataque) |
| `sucessful_attacks`, `interest_point_health` | Dano sofrido pelo ativo defendido |
| `friends_returned` | Drones que pousaram após o "retorno à base" |
| `enemies_remaining`, `friends_remaining` | Drones ainda vivos |
| `state_percentages` | Fração do tempo que o enxame passou em cada rótulo de comportamento |
| `total_distance_traveled` | Distância percorrida pelo enxame, em pixels |

Os nomes originais das chaves (com erros de grafia), como `sucessful_attacks`, são mantidos por retrocompatibilidade.

## Interface interativa

Com `mode="human"`, a janela mostra:

- **Área de simulação (à esquerda).** Drones amigos (brancos), inimigos (vermelhos), o ponto de interesse (verde, ficando vermelho à medida que perde vida), alcances dos sensores, enlaces de comunicação, zonas desmilitarizadas e o mapa de calor consolidado dos inimigos.
- **Painel de percepção (à direita).** Um gráfico 3D das matrizes de inimigos e de amigos do drone selecionado. A altura representa a recência e a cor, a direção. Clique em um drone para selecioná-lo.
- **Painel de controle.** Botões que ligam e desligam cada camada, além de pausar, reiniciar, sair, *retorno à base* (todos os drones voltam para casa), detecção passiva, exportação para o Tacview (`./output/`) e gravação de quadros (`./tmp/`). Transforme os quadros gravados em vídeo com `python -m DroneSwarm2D create_video ./tmp/frames_...`.

Com `JOYSTICK` definido como `"Enemy"` (ou `"Friend"`), um gamepad pilota um inimigo (ou o drone amigo líder).

## Desempenho

O motor de simulação foi otimizado nesta versão. As varreduras de percepção, a seleção de vizinhos, a cobertura dos sensores, o traçado de linha de visada e a mesclagem de matrizes agora são vetorizados; a grade de triangulação de 30 MB por drone deixou de ser alocada a cada passo; as fontes ficam em cache; e o desenho das trajetórias é recortado. Medido no mesmo cenário (24 contra 24 drones, 1920×1080, `make benchmark`) e verificado passo a passo para garantir resultados idênticos:

| | Antes | Depois | Ganho |
|---|---:|---:|---:|
| Passo headless, detecção direta | 122 ms | 15,7 ms | 7,8× |
| Passo headless, detecção passiva | 123 ms | 18,9 ms | 6,5× |
| Quadro renderizado | 220 ms | 30 ms | 7,4× |

Execute `make benchmark ARGS="--friends 50 --enemies 50 --passive"` para medir os seus próprios cenários.

## Estrutura do projeto

```
DroneSwarm2D/
├── __init__.py            # init() e as classes públicas
├── cli.py                 # python -m DroneSwarm2D init | create_video
├── core/
│   ├── settings.py        # carregador da configuração
│   ├── AirTrafficEnv.py   # ambiente: step, regras de combate, recompensa, interface
│   ├── FriendDrone.py     # percepção, comunicação, movimento
│   ├── EnemyDrone.py      # padrões de ataque
│   ├── InterestPoint.py   # ativo defendido
│   ├── DemilitarizedZone.py
│   ├── rendering.py       # funções auxiliares de desenho e widgets
│   └── utils.py           # geometria da grade, interceptação, matrizes
├── behaviorsDefault/      # BaseBehavior, BehaviorType, políticas padrão
├── tools/create_video.py
├── templates/             # projeto criado por `init`
└── assets/                # sprites
example/                   # exemplo pronto para executar (mesmo código do template)
tests/                     # suíte de testes pytest
benchmarks/                # medição de desempenho
docs/en, docs/pt-BR/       # documentação
```

## Desenvolvimento

```bash
make install-dev    # instalação editável com pytest, pytest-cov e ruff
make test           # executa a suíte de testes (headless)
make coverage       # testes com relatório de cobertura
make lint           # ruff
make benchmark      # velocidade da simulação
make demo           # executa o exemplo (ARGS="--windowed --episodes 5")
```

A integração contínua executa o linter e a suíte de testes a cada push. Veja o [CONTRIBUTING.pt-BR.md](CONTRIBUTING.pt-BR.md) para as diretrizes.

## Motivação e resultados da pesquisa

Drones baratos, muitas vezes improvisados com materiais simples, mudaram os conflitos modernos. As defesas aéreas convencionais são caras e têm dificuldade para enfrentar ataques de enxames. O DroneSwarm2D explora enxames defensivos descentralizados que:

1. eliminam pontos únicos de falha, comuns em sistemas centralizados;
2. têm boa relação custo-benefício em comparação com as contramedidas tradicionais;
3. respondem com flexibilidade e escalam para ameaças diversas;
4. otimizam a coordenação tática por meio de princípios de computação distribuída.

O simulador compara três estratégias defensivas:

1. **Benchmark (não cooperativa).** Os drones operam de forma independente, sem comunicação.
2. **Proposta (distribuída).** Os drones compartilham informações pela rede ad hoc.
3. **Centralizada.** Os drones têm informação global completa.

Nos experimentos do trabalho de graduação:

- a abordagem distribuída superou claramente a não cooperativa;
- a diferença entre as abordagens distribuída e centralizada foi pequena em muitos cenários;
- a estratégia distribuída foi mais resiliente a falhas de comunicação e à adaptação do inimigo.

Os testes de Kolmogorov–Smirnov e de Mann–Whitney confirmaram que as diferenças nas métricas de desempenho são estatisticamente significativas.

## Licença e contato

Veja [LICENSE](LICENSE). Para dúvidas ou colaborações, entre em contato pelo e-mail [lucas.silva1037@gmail.com](mailto:lucas.silva1037@gmail.com).
