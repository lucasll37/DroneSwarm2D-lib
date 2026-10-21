# Configuration reference

🇧🇷 [Versão em português](../pt-BR/configuration.md) · [Back to the README](../../README.md)

A scenario is a JSON file passed to `DroneSwarm2D.init(path)`. Each key becomes an attribute of the settings module:

```python
settings = DroneSwarm2D.init("config.json")
settings.SIM_WIDTH
```

It can also be imported, after `init()`:

```python
from DroneSwarm2D.core.settings import SIM_WIDTH
```

## How values are read

- **Numbers, booleans and lists** are used as they are.
- **Strings** are evaluated as Python expressions over the keys defined **above** them in the file, so order matters: `"GRID_WIDTH": "SIM_WIDTH // CELL_SIZE"` must come after `SIM_WIDTH` and `CELL_SIZE`.
- A string that cannot be evaluated stays a string (`"Courier New"`), and `"None"` becomes `None`.
- **List items** are evaluated individually, as in `"INTEREST_POINT_CENTER": ["SIM_WIDTH / 2", "SIM_HEIGHT / 2"]`.
- `FULL_WIDTH` / `FULL_HEIGHT` always come from the display size, or from `init(..., screen_size=(w, h))`; in headless mode the default is 1920×1080. Whatever the file says for these two keys is ignored.
- `INTEREST_POINT_CENTER`, `CENTER` and `TYPE_OF_SCENARIO` (the file name) are recomputed after loading.

> ⚠️ Values are evaluated with `eval`. Only load configuration files you trust.

The configuration is loaded **once per process**. Calling `init()` again with another file emits a warning and keeps the first configuration.

## Keys

The *Example* column shows the value used in `example/src/config.json`. Distances are in pixels and speeds in pixels per step.

### Forces

| Key | Example | Description |
|---|---|---|
| `FRIEND_COUNT` | `8` | Total number of friendly drones (radars + AEW + interceptors) |
| `RADAR_COUNT` | `0` | Fixed ground radars, included in `FRIEND_COUNT` |
| `AEW_COUNT` | `0` | Airborne early warning drones, included in `FRIEND_COUNT` |
| `BROKEN_COUNT` | `0` | How many interceptors have faulty sensors that broadcast random data |
| `ENEMY_COUNT` | `8` | Number of attacking drones (including the joystick one, if any) |

### Sensors

| Key | Example | Description |
|---|---|---|
| `FRIEND_DETECTION_RANGE` | `250` | Sensor range of interceptors |
| `AEW_DETECTION_RANGE` | `200` | Sensor range of AEW drones |
| `RADAR_DETECTION_RANGE` | `350` | Sensor range of radars |
| `ENEMY_DETECTION_RANGE` | `100` | Distance at which an enemy notices a friendly drone (and reacts) |
| `PASSIVE_DETECTION` | `true` | Start in passive (triangulation) detection mode; can be toggled in the UI |
| `N_LINE_SIGHT_CROSSING` | `3` | Sensors that must cover an enemy for a passive detection |
| `TRIANGULATION_GRANULARITY` | `32` | Sub-cells per grid cell of the line-of-sight grid |

### Communication

| Key | Example | Description |
|---|---|---|
| `COMMUNICATION_RANGE` | `250` | Maximum link distance; friends are also sensed within this range |
| `N_CONNECTIONS` | `3` | Each drone links to its N nearest drones (links are symmetric) |
| `CICLE_COMM_BY_STEP` | `3` | Communication cycles per step (each one is a retransmission opportunity) |
| `MESSAGE_LOSS_PROBABILITY` | `0.1` | Probability that one exchange is lost |
| `MIN_COMMUNICATION_HOLD` | `3` | Fresh friends in range needed to hold position; below this a drone regroups. `0` makes idle drones return to `INITIAL_DISTANCE` |

### Perception grid

| Key | Example | Description |
|---|---|---|
| `CELL_SIZE` | `20` | Cell size in pixels |
| `GRID_WIDTH` / `GRID_HEIGHT` | `"SIM_WIDTH // CELL_SIZE"` / `"SIM_HEIGHT // CELL_SIZE"` | Grid size in cells |
| `DECAY_FACTOR` | `0.99` | Per-step multiplier of the intensity matrices |
| `PLOT_THRESHOLD` | `0.05` | Cells below this intensity are white in the 3-D plots |

### Screen and geometry

| Key | Example | Description |
|---|---|---|
| `FULL_WIDTH` / `FULL_HEIGHT` | display size | Window size (see above) |
| `SIM_WIDTH` / `SIM_HEIGHT` | `"int(FULL_WIDTH * 0.7)"` / `"FULL_HEIGHT"` | Simulation area |
| `GRAPH_WIDTH` / `GRAPH_HEIGHT` | `"FULL_WIDTH - SIM_WIDTH"` / `"FULL_HEIGHT"` | Graph and control panel |
| `INTEREST_POINT_CENTER` | `["SIM_WIDTH / 2", "SIM_HEIGHT / 2"]` | Recomputed as the centre of the simulation area |
| `INTERNAL_RADIUS` | `"min(SIM_WIDTH, SIM_HEIGHT) / 10"` | Protected area |
| `EXTERNAL_RADIUS` | `"INTERNAL_RADIUS * 4"` | Friendly drones never fly farther than this from the centre |
| `INITIAL_DISTANCE` | `"INTERNAL_RADIUS * 1.4"` | Radius of the first ring of interceptors; enemies are fully aggressive inside it |
| `INITIAL_N_LAYERS` | `1` | Number of rings the interceptors start on (40 px apart) |
| `RADAR_RANGE` | `0` | Distance from the centre at which radars are placed |
| `AEW_RANGE` | `350` | Orbit radius of AEW drones (and where they start) |
| `THRESHOLD_PROJECTION` | `"INTERNAL_RADIUS * 0.5"` | Maximum distance an idle drone moves to stand on an enemy's line of approach |
| `EPSILON` | `2` | Distance tolerance (arrival, interception point) |
| `DMZ` | `[["SIM_WIDTH * 0.35", "SIM_HEIGHT * 0.30", 60], ...]` | Demilitarised zones as `[x, y, radius]` |
| `GEO_TOP_LEFT` / `GEO_BOTTOM_RIGHT` | `[-74.0, 40.8]` / `[-73.9, 40.7]` | Longitude/latitude of the area corners (Tacview export) |

### Motion

| Key | Example | Description |
|---|---|---|
| `BASE_SPEED` | `2.0` | Convenience value referenced by the speeds below |
| `FRIEND_SPEED` | `"BASE_SPEED"` | Interceptor speed |
| `AEW_SPEED` | `"FRIEND_SPEED"` | AEW speed |
| `ENEMY_SPEED` | `"BASE_SPEED"` | Enemy speed |
| `DT_STEP` | `0.6` | Simulated seconds per step (clock and Tacview timestamps) |

### Interest point and combat

| Key | Example | Description |
|---|---|---|
| `INTEREST_POINT_INITIAL_HEALTH` | `100` | Initial health of the asset |
| `INTEREST_POINT_DAMAGE` | `"INTEREST_POINT_INITIAL_HEALTH // ENEMY_COUNT"` | Damage per successful attack |
| `INTEREST_POINT_ATTACK_RANGE` | `"INTERNAL_RADIUS"` | An enemy this close to the centre hits the asset |
| `NEUTRALIZATION_RANGE` | `20` | A friend and an enemy this close fight |
| `NEUTRALIZATION_PROB_FRIEND_ALIVE` | `0.5` | Interceptor wins (enemy destroyed) |
| `NEUTRALIZATION_PROB_ENEMY_ALIVE` | `0.2` | Enemy wins (interceptor destroyed) |
| `NEUTRALIZATION_PROB_BOTH_DEAD` | `"1 - (... + ...)"` | Both destroyed; also the collision probability for AEW and radars |

### Enemies

| Key | Example | Description |
|---|---|---|
| `ENEMY_BEHAVIOR` | `"None"` | Pattern for every enemy (see [architecture](architecture.md#enemies)); `"None"` draws one per enemy |
| `INITIAL_AGGRESSIVENESS` | `0.5` | Attack probability when detected at `EXTERNAL_RADIUS` (grows to 1 at `INITIAL_DISTANCE`) |
| `ESCAPE_STEPS` | `40` | Duration of an escape |

### Faults, interaction and display

| Key | Example | Description |
|---|---|---|
| `UPDATE_STATE_BROKEN` | `100` | A broken drone draws a new fake pattern after this many calls |
| `HOLD_SPREAD` | `true` | Idle interceptors spread around the centre |
| `JOYSTICK` | `"None"` | `"Friend"`: a game pad flies the leading interceptor; `"Enemy"`: it flies one enemy, and friends are only visible near enemies |
| `FONT_FAMILY` | `"Courier New"` | UI font (falls back to pygame's default if missing) |

### Free keys

`EPISODES`, `ANALYSIS_EPISODES` and `TARGET_INFLUENCE` are not used by the library. Your scripts can read them, for example as `settings.EPISODES`, and you can add any other key you need.
