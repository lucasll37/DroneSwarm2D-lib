# My DroneSwarm2D project

🇧🇷 [Leia em português](README.pt-BR.md)

![Drone swarm](https://raw.githubusercontent.com/lucasll37/DroneSwarm2D-lib/main/images/banner.gif)

This project was created with `python -m DroneSwarm2D init`. It is a ready-to-run starting point for experimenting with defensive drone swarms.

## Structure

```
droneSwarm2D-project/
├── src/
│   ├── main.py        # runs the episodes and prints the statistics
│   ├── behaviors.py   # the policies of your drones: edit me!
│   └── config.json    # the scenario: forces, sensors, communication, combat...
├── README.md          # this file (README.pt-BR.md: Portuguese)
└── TUTORIAL.md        # how the simulation works and how to write behaviours (TUTORIAL.pt-BR.md: Portuguese)
```

## Running

```bash
python src/main.py                         # full-screen window, 3 episodes
python src/main.py --windowed --episodes 5
python src/main.py --headless --episodes 20 --seed 7   # no window, as fast as possible
python src/main.py --help
```

In the window:

- click a drone to inspect its perception in the right-hand panel (*Tog. Graph*);
- use the buttons to show or hide layers, pause, reset, record frames or export to Tacview.

## Next steps

1. Change the scenario in `src/config.json` (for example `ENEMY_COUNT`, `MESSAGE_LOSS_PROBABILITY` or `ENEMY_BEHAVIOR`).
2. Edit `FriendCommonBehavior` in `src/behaviors.py`.
3. Run headless over several seeds and compare the averages printed by `main.py`.

Read [TUTORIAL.md](TUTORIAL.md) for the details. The full documentation is at <https://github.com/lucasll37/DroneSwarm2D-lib>.
