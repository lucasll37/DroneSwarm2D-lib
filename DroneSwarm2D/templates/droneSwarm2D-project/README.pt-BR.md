# Meu projeto DroneSwarm2D

🇺🇸 [Read in English](README.md)

![Enxame de drones](https://raw.githubusercontent.com/lucasll37/DroneSwarm2D-lib/main/images/banner.gif)

Este projeto foi criado com `python -m DroneSwarm2D init`. Ele é um ponto de partida pronto para rodar, feito para experimentar com enxames defensivos de drones.

## Estrutura

```
droneSwarm2D-project/
├── src/
│   ├── main.py        # executa os episódios e imprime as estatísticas
│   ├── behaviors.py   # as políticas dos seus drones: edite este arquivo!
│   └── config.json    # o cenário: forças, sensores, comunicação, combate...
├── README.md          # visão geral em inglês (README.pt-BR.md: este arquivo)
└── TUTORIAL.md        # como a simulação funciona e como escrever comportamentos (TUTORIAL.pt-BR.md: em português)
```

## Como executar

```bash
python src/main.py                         # janela em tela cheia, 3 episódios
python src/main.py --windowed --episodes 5
python src/main.py --headless --episodes 20 --seed 7   # sem janela, o mais rápido possível
python src/main.py --help
```

Na janela:

- clique em um drone para inspecionar a percepção dele no painel à direita (*Tog. Graph*);
- use os botões para mostrar ou ocultar camadas, pausar, reiniciar, gravar quadros ou exportar para o Tacview.

## Próximos passos

1. Altere o cenário em `src/config.json` (por exemplo `ENEMY_COUNT`, `MESSAGE_LOSS_PROBABILITY` ou `ENEMY_BEHAVIOR`).
2. Edite `FriendCommonBehavior` em `src/behaviors.py`.
3. Execute sem janela (headless) com várias sementes (seeds) e compare as médias impressas por `main.py`.

Leia o [TUTORIAL.pt-BR.md](TUTORIAL.pt-BR.md) para os detalhes. A documentação completa está em <https://github.com/lucasll37/DroneSwarm2D-lib>.
