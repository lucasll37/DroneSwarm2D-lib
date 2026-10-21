# DroneSwarm2D tutorial

🇧🇷 [Leia em português](TUTORIAL.pt-BR.md)

This tutorial covers what you need to change the scenario and write your own drone policies. The complete documentation (architecture, every configuration key, API reference) is at <https://github.com/lucasll37/DroneSwarm2D-lib/tree/main/docs/en>.

## Contents

1. [The main loop](#1-the-main-loop)
2. [The scenario: `config.json`](#2-the-scenario-configjson)
3. [How the simulation works](#3-how-the-simulation-works)
4. [Writing behaviours](#4-writing-behaviours)
5. [Metrics](#5-metrics)
6. [The interface](#6-the-interface)
7. [Recipes](#7-recipes)

## 1. The main loop

`src/main.py` does four things:

```python
import DroneSwarm2D

# 1. Load the scenario. Always the first step: the library reads it at import time.
settings = DroneSwarm2D.init("src/config.json", fullscreen=False)   # headless=True: no window

# 2. Import your behaviours only now.
from behaviors import FriendCommonBehavior, FriendAEWBehavior, FriendRadarBehavior

# 3. Create the environment.
env = DroneSwarm2D.AirTrafficEnv(
    mode="human",                        # None = headless and fast
    friend_behavior=FriendCommonBehavior(),
    friend_aew_behavior=FriendAEWBehavior(),
    friend_radar_behavior=FriendRadarBehavior(),
    enemy_behavior=settings.ENEMY_BEHAVIOR,
    demilitarized_zones=settings.DMZ,
    seed=42,                             # same seed -> same episode
)

# 4. Run episodes.
obs, done = env.reset()
while not done:
    obs, reward, done, info = env.step(None)
env.close()
```

## 2. The scenario: `config.json`

String values are Python expressions over the keys defined above them (`"EXTERNAL_RADIUS": "INTERNAL_RADIUS * 4"`), and `"None"` means `None`. The most useful keys:

| Key | Meaning |
|---|---|
| `FRIEND_COUNT`, `ENEMY_COUNT` | Size of each side (`FRIEND_COUNT` includes radars and AEW drones) |
| `RADAR_COUNT`, `AEW_COUNT` | Fixed radars / early-warning drones (placed at `RADAR_RANGE` / `AEW_RANGE` from the centre) |
| `BROKEN_COUNT` | Interceptors with faulty sensors that spread random data |
| `FRIEND_DETECTION_RANGE`, `COMMUNICATION_RANGE` | Sensor and radio range, in pixels |
| `N_CONNECTIONS`, `CICLE_COMM_BY_STEP`, `MESSAGE_LOSS_PROBABILITY` | Network: links per drone, exchanges per step, loss rate |
| `DECAY_FACTOR` | How fast information gets old (intensity × factor per step) |
| `PASSIVE_DETECTION`, `N_LINE_SIGHT_CROSSING` | Only confirm enemies seen by N sensors |
| `ENEMY_BEHAVIOR` | `"None"` (random), `"direct"`, `"zigzag"`, `"spiral"`, `"formation"`, ... |
| `INITIAL_AGGRESSIVENESS`, `ESCAPE_STEPS` | How enemies react when detected |
| `NEUTRALIZATION_RANGE`, `NEUTRALIZATION_PROB_*` | Combat |
| `DMZ` | Demilitarised zones `[x, y, radius]`, where nobody may fight |
| `JOYSTICK` | `"Friend"` / `"Enemy"`: fly a drone with a game pad; `"None"` to disable |

The window size comes from your display. Pass `screen_size=(1920, 1080)` to `DroneSwarm2D.init` to make results independent of the monitor.

## 3. How the simulation works

Each step:

1. **Every friendly drone:**
   - *forgets*: its matrices decay;
   - *links*: it picks up to `N_CONNECTIONS` neighbours;
   - *senses*: it detects enemies and friends in range;
   - *talks*: it merges matrices with its neighbours, keeping the newest information in each cell;
   - *acts*: its behaviour returns a velocity.
2. **Enemies** move. If a friend is within `ENEMY_DETECTION_RANGE`, an enemy either attacks in a straight line or escapes; the closer it is to the target, the more likely it attacks.
3. **Attacks.** An enemy that reaches the interest point damages it (and destroys a radar).
4. **Engagements.** A friend and an enemy closer than `NEUTRALIZATION_RANGE` fight: the enemy, the friend or both are destroyed.

A drone's knowledge is a set of grid matrices (`CELL_SIZE`-pixel cells):

- `enemy_intensity[i, j]` is 1 when an enemy was just seen in cell `(i, j)` and fades towards 0 as the information gets older;
- `enemy_direction[i, j]` is that enemy's unit heading;
- the friend matrices hold the same information about friendly drones.

## 4. Writing behaviours

A behaviour is a class with an `apply(state)` method that returns `(info, velocity)`:

```python
import numpy as np
import pygame

from DroneSwarm2D.behaviorsDefault import BaseBehavior, BehaviorType
from DroneSwarm2D.core import settings
from DroneSwarm2D.core.utils import active_cells, cell_center, intercept_direction


class MyBehavior(BaseBehavior):
    def __init__(self):
        super().__init__(BehaviorType.COMMON)   # COMMON, AEW or RADAR

    def apply(self, state, joystick_controlled=False):
        pos = pygame.math.Vector2(*np.squeeze(state["pos"]).tolist())
        enemies = active_cells(np.squeeze(state["enemy_intensity"]), 0.4)
        if not enemies:
            return ("WAIT", None, None, None), pygame.math.Vector2(0, 0)

        cell = min(enemies, key=lambda c: cell_center(c).distance_to(pos))
        heading = pygame.math.Vector2(*np.squeeze(state["enemy_direction"])[cell])
        vel = intercept_direction(pos, settings.FRIEND_SPEED, cell_center(cell), heading * settings.ENEMY_SPEED)
        return ("PURSUING", cell_center(cell), None, None), vel
```

**The `state` dictionary.** Every array has a leading axis of size 1; use `np.squeeze`.

| Key | Shape | Content |
|---|---|---|
| `drone_id` | `int` | Id of the drone |
| `pos` | `(1, 2)` | Position in pixels |
| `enemy_intensity` / `friend_intensity` | `(1, GRID_W, GRID_H)` | Recency of detections (0..1) |
| `enemy_direction` / `friend_direction` | `(1, GRID_W, GRID_H, 2)` | Unit headings |

**Rules of thumb:**

- `info[0]` is a label shown on screen and counted in `info["state_percentages"]`; `info[1]` is an optional target point drawn in debug mode.
- Return a velocity no longer than `settings.FRIEND_SPEED`.
- One instance is shared by all drones of a role, so keep per-drone memory in a dictionary keyed by `state["drone_id"]`.
- Use `active_cells` instead of Python loops over the whole grid: `apply` runs for every drone at every step.
- Headings are unit vectors; multiply by `settings.ENEMY_SPEED` to get velocities.
- Do not modify the `state` arrays: they are views of the drone's memory.

The library's own policy, `FriendCommonBehaviorDefault`, is a good reference. Subclass it to reuse the target assignment and change only what idle drones do:

```python
from DroneSwarm2D.behaviorsDefault import FriendCommonBehaviorDefault

class MyDefault(FriendCommonBehaviorDefault):
    def _hold_position(self, pos, friend_intensity, enemy_intensity, enemy_direction, friends_hold=None, **kw):
        return ("PATROL", None, None, friends_hold), pygame.math.Vector2(0, 0)
```

## 5. Metrics

`env.step()` returns `info` with:

- `enemies_shotdown`, `friends_shotdown`;
- `sucessful_attacks`, `interest_point_health`;
- `accum_reward`, `current_step`;
- `friends_returned`, `enemies_remaining`, `friends_remaining`;
- `state_percentages`, `total_distance_traveled`.

## 6. The interface

| Button | Effect |
|---|---|
| Tog. Graph | 3-D plot of the selected drone's perception (click a drone to select it) |
| Pause / Reset / Exit | Simulation control |
| Tog. Friend Range / Enemy Range / Comm Range | Sensor and radio circles |
| Tog. Friend Comm. | Communication links |
| Tog. DMZ | Shows **and enables** the demilitarised zones |
| Export Tacview | Writes CSV tracks to `./output/` when the episode ends |
| Tog. Save Frames | Saves PNG frames to `./tmp/`; make a video with `python -m DroneSwarm2D create_video ./tmp/frames_...` |
| Tog. Target Lines / Trajetory / Debug | Visual aids |
| Tog. D. Passive. | Direct ↔ passive (triangulation) detection |
| Tog. Return | All drones return to base |

## 7. Recipes

**Compare two policies over the same seeds:**

```python
results = []
for seed in range(20):
    env = DroneSwarm2D.AirTrafficEnv(mode=None, seed=seed, friend_behavior=MyBehavior(),
                                     enemy_behavior=settings.ENEMY_BEHAVIOR, demilitarized_zones=settings.DMZ)
    _, done = env.reset()
    while not done:
        _, _, done, info = env.step(None)
    results.append(info)
print(sum(r["sucessful_attacks"] for r in results) / len(results))
```

**Stress the network:**
- raise `MESSAGE_LOSS_PROBABILITY`;
- lower `N_CONNECTIONS` or `COMMUNICATION_RANGE`;
- add `BROKEN_COUNT` faulty drones.

**Harder enemies:**
- `"ENEMY_BEHAVIOR": "formation"`;
- raise `INITIAL_AGGRESSIVENESS`;
- raise `ENEMY_SPEED`.
