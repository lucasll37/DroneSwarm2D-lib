# Exemplo do DroneSwarm2D

🇺🇸 [Read in English](README.md)

Um cenário pronto para executar: 8 drones defensivos contra 8 atacantes com padrões de aproximação aleatórios, detecção passiva e três zonas desmilitarizadas (DMZ). O código em `src/` é idêntico ao do projeto criado por `python -m DroneSwarm2D init`.

| Arquivo | Conteúdo |
|---|---|
| `src/main.py` | Executa os episódios e imprime as estatísticas |
| `src/behaviors.py` | Políticas de exemplo: um interceptador que persegue a ameaça mais urgente ou patrulha, uma órbita de AEW e um radar fixo |
| `src/config.json` | O cenário |

## Executando

A partir da raiz do repositório, depois de `pip install -e .`:

```bash
python example/src/main.py --windowed            # janela interativa
python example/src/main.py --headless --episodes 10 --seed 1
python example/src/main.py --help
```

Sem instalar, `make demo` / `make demo-headless` configuram o `PYTHONPATH` para você (opções extras vão em `ARGS="..."`).

## O que experimentar

- Mude `ENEMY_BEHAVIOR` em `config.json` para `"formation"` ou `"spiral"`.
- Defina `MESSAGE_LOSS_PROBABILITY` como `0.5` e veja como o enxame se sai.
- Substitua `FriendCommonBehavior()` por `FriendCommonBehaviorDefault()` (de `DroneSwarm2D.behaviorsDefault`) em `main.py` e compare as médias.

Veja o [tutorial](../DroneSwarm2D/templates/droneSwarm2D-project/TUTORIAL.pt-BR.md) e o [guia de comportamentos](../docs/pt-BR/behaviors.md).
