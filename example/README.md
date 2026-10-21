# DroneSwarm2D example

🇧🇷 [Leia em português](README.pt-BR.md)

A ready-to-run scenario: 8 defensive drones against 8 attackers with random approach patterns, passive detection and three demilitarised zones. The code in `src/` is identical to the project created by `python -m DroneSwarm2D init`.

| File | Content |
|---|---|
| `src/main.py` | Runs the episodes and prints the statistics |
| `src/behaviors.py` | Example policies: an interceptor that chases the most urgent threat or patrols, an AEW orbit and a fixed radar |
| `src/config.json` | The scenario |

## Running

From the repository root, after `pip install -e .`:

```bash
python example/src/main.py --windowed            # interactive window
python example/src/main.py --headless --episodes 10 --seed 1
python example/src/main.py --help
```

Without installing, `make demo` / `make demo-headless` set `PYTHONPATH` for you (extra options go in `ARGS="..."`).

## Things to try

- Change `ENEMY_BEHAVIOR` in `config.json` to `"formation"` or `"spiral"`.
- Set `MESSAGE_LOSS_PROBABILITY` to `0.5` and see how the swarm copes.
- Replace `FriendCommonBehavior()` with `FriendCommonBehaviorDefault()` (from `DroneSwarm2D.behaviorsDefault`) in `main.py` and compare the averages.

See the [tutorial](../DroneSwarm2D/templates/droneSwarm2D-project/TUTORIAL.md) and the [behaviour guide](../docs/en/behaviors.md).
