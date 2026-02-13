# DroneSwarm2D: Um Simulador para Estudo de Táticas Defensivas Distribuídas com Enxames de Drones Autônomos

![Banner do Enxame de Drones](./images/centralized.gif)

---


## 📚 Visão Geral

DroneSwarm2D é um ambiente de simulação 2D projetado para estudar táticas defensivas distribuídas para enxames de drones autônomos. O simulador aborda o crescente desafio representado por drones ofensivos de baixo custo em conflitos modernos, possibilitando a pesquisa de estratégias de defesa descentralizadas e resilientes. Ele fornece uma plataforma para modelagem de comportamentos de enxames de drones, implementação de redes de comunicação ad-hoc e avaliação da eficácia de vários algoritmos defensivos.

### Características Principais

- **Tomada de Decisão Distribuída**: Modela drones autônomos que operam sem controle central
- **Comunicação em Rede Ad-hoc**: Simula comunicações realistas entre drones
- **Percepção de Estado**: Cada drone mantém sua própria percepção do ambiente através de detecção local
- **Compartilhamento de Informações**: Os drones trocam e mesclam matrizes de detecção para melhorar a consciência colaborativa
- **Múltiplos Comportamentos**: Implementa vários padrões de aproximação inimiga e estratégias de resposta defensiva
- **Visualizações**: Ferramentas ricas de visualização para matrizes de detecção, estados dos drones e estatísticas
- **Métricas de Desempenho**: Métricas abrangentes para avaliar a eficácia da defesa

## 🎯 Motivação do Projeto

A crescente acessibilidade de drones de baixo custo transformou os cenários de conflito modernos. Esses dispositivos de baixo custo, frequentemente construídos com materiais simples e adaptados para fins ofensivos, representam desafios significativos para os sistemas de defesa convencionais, que normalmente exigem investimentos substanciais e frequentemente têm dificuldades para enfrentar ataques em enxame.

O DroneSwarm2D aborda essa assimetria explorando redes descentralizadas de defesa de drones que:

1. Eliminam pontos únicos de falha (comuns em sistemas centralizados)
2. Fornecem alternativas econômicas às contramedidas tradicionais caras
3. Permitem respostas flexíveis e escaláveis a diversas ameaças
4. Otimizam a coordenação tática através de princípios de computação distribuída

Esta abordagem se inspira em princípios operacionais de pesquisa de longa data, visando criar sistemas de defesa resilientes que possam operar efetivamente mesmo com recursos limitados.

## 🔧 Arquitetura Técnica

### Ambiente de Simulação

A simulação é construída em um sistema de grade 2D com os seguintes componentes:

- **Ponto de Interesse**: Área central a ser defendida, com saúde que diminui quando atacada com sucesso
- **Drones Amigos**: Drones defensivos autônomos implementando vários comportamentos
- **Drones Inimigos**: Drones ofensivos com padrões de ataque configuráveis
- **Zonas Desmilitarizadas**: Áreas onde o engajamento é proibido

### Representação de Estado

Cada drone mantém sua própria percepção local do ambiente através de:

1. **Matrizes de Detecção**:
   - `enemy_intensity`: Registra detecções recentes de inimigos (valores 0-1)
   - `enemy_direction`: Armazena vetores de direção de inimigos detectados
   - `friend_intensity`: Registra detecções recentes de drones amigos
   - `friend_direction`: Armazena vetores de direção de drones amigos detectados

2. **Sistema de Triangulação**:
   Quando múltiplos drones detectam o mesmo alvo de diferentes ângulos, suas informações de detecção são combinadas para melhorar a precisão.

### Sistema de Comunicação

A simulação implementa comunicação realista entre drones com:
- Alcance de comunicação limitado entre drones
- Formação de rede ad-hoc baseada em proximidade
- Perda probabilística de mensagens
- Fusão de informações através da mesclagem de matrizes

### Comportamentos dos Drones

#### Comportamentos de Drones Defensivos
- **Planning**: Tomada de decisão estratégica baseada na percepção atual do estado
- **AEW (Alerta Aéreo Antecipado)**: Padrão de vigilância em órbita
- **RADAR**: Unidade de detecção estacionária
- **AI**: Tomada de decisão baseada em rede neural

#### Comportamentos de Drones Ofensivos
- **Direct**: Abordagem direta ao alvo
- **Zigzag**: Abordagem com oscilações laterais
- **Spiral**: Movimento em espiral em direção ao alvo
- **Alternating**: Alternância entre abordagem direta e movimento perpendicular
- **Bounce Approach**: Avançar e recuar com variação direcional
- **Formation**: Movimento coordenado em formações predefinidas


### Configuração da Simulação

O simulador utiliza arquivos JSON na pasta `config/preset/` para definir parâmetros de simulação. Você pode:

1. Modificar os arquivos existentes para ajustar parâmetros
2. Criar novos arquivos de configuração baseados nos existentes
3. Especificar um arquivo de configuração personalizado através da variável de ambiente `CONFIG_FILE`

## 🎮 Interface do Usuário

A interface da simulação consiste em:

1. **Área de Simulação**: Representação 2D do ambiente mostrando:
   - Drones amigos (brancos)
   - Drones inimigos (vermelhos)
   - Ponto de interesse (círculo verde que se torna vermelho conforme a saúde diminui)
   - Alcances de detecção (círculos tracejados)
   - Links de comunicação (linhas tracejadas)

2. **Painel de Visualização**: Visualização 3D da percepção de um drone selecionado:
   - Intensidade e direção de detecção de inimigos
   - Intensidade e direção de detecção de amigos
   - Ângulos de direção codificados por cores

3. **Painel de Controle**: Botões de interface para:
   - Alternar recursos de visualização
   - Pausar/retomar simulação
   - Exportar dados
   - Reiniciar a simulação

4. **Exibição de Estatísticas**: Métricas em tempo real mostrando:
   - Contagens de drones
   - Estatísticas de comunicação
   - Saúde do ponto de interesse
   - Tempo de simulação

## 📊 Descobertas da Pesquisa

O simulador permite a comparação entre diferentes estratégias defensivas:

1. **Benchmark (Não-cooperativo)**: Drones operam independentemente sem comunicação
2. **Proposta (Distribuída)**: Drones compartilham informações através de redes ad-hoc
3. **Centralizada**: Drones têm informação global completa

A análise dessas abordagens demonstra que:

- Abordagens distribuídas oferecem vantagens significativas sobre as não-cooperativas
- A diferença de desempenho entre abordagens distribuídas e centralizadas é mínima em muitos cenários
- Estratégias distribuídas mostram maior resiliência a falhas de comunicação e adaptações inimigas

Testes estatísticos (Kolmogorov-Smirnov e Mann-Whitney) confirmam diferenças significativas nas métricas de desempenho entre as abordagens.

## 🚀 Como Iniciar

### Pré-requisitos

- Anaconda

### 🚀 Instalação

#### 1. Criar Ambiente Python com o Anaconda

```bash
conda create --prefix .venv python=3.12 -y
```

#### 2. Ativar o Ambiente

```bash
conda activate ./.venv
```

#### 3. Instalar a Biblioteca

```bash
pip install DroneSwarm2D @ git+https://github.com/lucasll37/DroneSwarm2D-lib.git
```

#### 4. Criar Projeto Exemplo

```bash
python -m DroneSwarm2D
```

Este comando criará uma pasta `droneSwarm2D-project` com a estrutura completa do projeto exemplo.

#### 5. Entre no diretório do Projeto Exemplo

```bash
cd droneSwarm2D-project
```

#### 6. Execute o Projeto Exemplo

```bash
python -u ./src/main.py
```
---

### 📁 Estrutura do Projeto Exemplo

```
droneSwarm2D-project/
├── src/
|    ├── behaviors.py      # Comportamentos customizados dos drones
|    ├── config.json       # Configurações do cenário
|    └── main.py           # Ponto de entrada da simulação
|
├── README.md
└──TUTORIAL.md
```

###

## 🖥️ Interface de Linha de Comando (CLI)

O DroneSwarm2D fornece uma interface de linha de comando para facilitar a criação de projetos e geração de vídeos.

### Comandos Disponíveis

#### 1. Inicializar Projeto

Cria um novo projeto DroneSwarm2D com estrutura pré-configurada:
```bash
# Criar projeto com nome padrão
python -m DroneSwarm2D init

# Criar projeto com nome personalizado
python -m DroneSwarm2D init --name meu-projeto-defesa

# Criar projeto com template específico
python -m DroneSwarm2D init --name projeto-avancado --template advanced
```

**Opções:**
- `--name`: Nome do projeto/diretório (padrão: `droneSwarm2D-project`)

**Estrutura criada:**
```
droneSwarm2D-project/
├── src/
|    ├── behaviors.py      # Comportamentos customizados dos drones
|    ├── config.json       # Configurações do cenário
|    └── main.py           # Ponto de entrada da simulação
|
├── README.md
└──TUTORIAL.md
```

#### 2. Criar Vídeo

Converte frames PNG de simulações em vídeo MP4:
```bash
# Criar vídeo com configurações padrão
python -m DroneSwarm2D create_video ./frames

# Especificar diretório de saída
python -m DroneSwarm2D create_video ./frames --output-dir ./videos

# Ajustar FPS e codec
python -m DroneSwarm2D create_video ./frames --fps 60 --codec libx265

# Remover frames após criar vídeo
python -m DroneSwarm2D create_video ./frames --remove-frames

# Exemplo completo
python -m DroneSwarm2D create_video ./simulation_frames \
    --output-dir ./videos/resultado \
    --fps 60 \
    --codec libx264 \
    --remove-frames
```

**Argumentos:**
- `path`: Caminho para o diretório contendo os frames PNG (obrigatório)

**Opções:**
- `--output-dir`: Diretório de saída (padrão: `./video`)
- `--fps`: Frames por segundo (padrão: `30`)
- `--codec`: Codec de vídeo (padrão: `libx264`)
- `--remove-frames`: Remove frames após criar vídeo

**Codecs recomendados por caso de uso:**

| Caso de Uso | FPS | Codec | Motivo |
|-------------|-----|-------|--------|
| Visualização rápida | 30 | libx264 | Balanço entre qualidade e tamanho |
| Apresentação acadêmica | 60 | libx264 | Alta fluidez, boa compatibilidade |
| Análise frame-a-frame | 30 | libx264 | Qualidade suficiente, fácil navegação |
| Armazenamento longo prazo | 30 | libx265 | Menor tamanho, mesma qualidade |

### Workflow Completo
```bash
# 1. Criar novo projeto
python -m DroneSwarm2D init --name simulacao-defesa
cd simulacao-defesa

# 2. Executar simulação (configurar save_frames=True no código)
python main.py

# 3. Criar vídeo a partir dos frames
python -m DroneSwarm2D create_video ./frames --fps 30 --remove-frames
```

### Ajuda
```bash
# Ajuda geral
python -m DroneSwarm2D --help

# Ajuda do comando init
python -m DroneSwarm2D init --help

# Ajuda do comando create_video
python -m DroneSwarm2D create_video --help
```


## 🤝 Contribuindo

Contribuições para o DroneSwarm2D são bem-vindas! Sinta-se à vontade para enviar pull requests ou abrir issues para discutir possíveis melhorias.

## 📄 Licença

Este projeto é lançado sob a MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 📧 Contato

Para perguntas ou oportunidades de colaboração, entre em contato pelo email [lucas.silva1037@gmail.com](mailto:lucas.silva1037@gmail.com).