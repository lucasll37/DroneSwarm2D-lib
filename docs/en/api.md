# API reference

🇧🇷 [Versão em português](../pt-BR/api.md) · [Back to the README](../../README.md)

Only the names listed here are considered public. Everything else (names starting with `_`) may change without notice.

## Contents

- [Package: `DroneSwarm2D`](#package-droneswarm2d)
- [Environment: `AirTrafficEnv`](#environment-airtrafficenv)
- [The `info` dictionary](#the-info-dictionary)
- [Behaviours: `DroneSwarm2D.behaviorsDefault`](#behaviours-droneswarm2dbehaviorsdefault)
- [Drones and scenery](#drones-and-scenery)
- [Utilities: `DroneSwarm2D.core.utils`](#utilities-droneswarm2dcoreutils)
- [Drawing: `DroneSwarm2D.core.rendering`](#drawing-droneswarm2dcorerendering)
- [Settings: `DroneSwarm2D.core.settings`](#settings-droneswarm2dcoresettings)
- [Video tool and command line](#video-tool-and-command-line)
- [Deprecated names](#deprecated-names)

## Package: `DroneSwarm2D`

### `init(config_path=None, fullscreen=True, headless=False, screen_size=None)`

Loads the scenario and makes the classes below available as attributes of the package. It must be called before using the rest of the library; later calls return the settings that are already loaded.

| Argument | Meaning |
|---|---|
| `config_path` | JSON file; if `None`, the first `*.json` found in a `config/` directory |
| `fullscreen` | Full-screen window (ignored when `headless`) |
| `headless` | No window (SDL dummy video driver), for servers, CI and batch runs |
| `screen_size` | `(width, height)` overriding the display size; default `(1920, 1080)` when `headless` |

It returns the `DroneSwarm2D.core.settings` module and raises `FileNotFoundError` if the file does not exist.

**Other names in the package:**

- `is_initialized()` returns whether `init` was already called.
- `__version__` is the version string.
- Available after `init`:
  - `AirTrafficEnv`, `FriendDrone`, `EnemyDrone`;
  - `CircleInterestPoint` (alias `InterestPoint`), `CircleDMZ` (alias `DemilitarizedZone`);
  - `generate_sparse_matrix`, `get_friends_hold`, `create_video_from_frames`.

## Environment: `AirTrafficEnv`

```python
AirTrafficEnv(max_steps=5000, mode="human", friend_behavior=None, friend_aew_behavior=None,
              friend_radar_behavior=None, enemy_behavior=None, demilitarized_zones=None, seed=None)
```

| Argument | Meaning |
|---|---|
| `max_steps` | Episode length limit |
| `mode` | `"human"` draws every step (capped at `render_fps`, 30 by default); anything else, such as `None`, runs headless |
| `friend_behavior` / `friend_aew_behavior` / `friend_radar_behavior` | Behaviour instances for interceptors, AEW drones and radars (defaults: `FriendCommonBehaviorDefault`, `FriendAEWBehaviorDefault`, `FriendRadarBehaviorDefault`) |
| `enemy_behavior` | Enemy pattern name (see `EnemyDrone.ENEMY_BEHAVIORS`) or `None` for a random pattern per enemy; usually `settings.ENEMY_BEHAVIOR` |
| `demilitarized_zones` | `[(x, y, radius), ...]`, usually `settings.DMZ`; zones overlapping the protected area are ignored |
| `seed` | Random seed (random if `None`); the same seed always gives the same episode |

**Methods:**

| Method | Description |
|---|---|
| `reset() -> (obs, done)` | Starts a new episode |
| `step(action=None) -> (obs, reward, done, info)` | Advances one step; `action` is ignored (drones act through their behaviours) |
| `render(mode="human")` | Draws a frame; called by `step` |
| `compute_reward(normilizer_factor=1e-4, bonus_factor=10, penalty_factor=2000)` | Reward of the current step |
| `close()` | Shuts pygame down |
| `handle_tacview()` | Writes one Tacview CSV per drone to `./output/tacview_export_<date>/` |
| `plot_individual_states() -> Surface` | 3-D perception plot of the selected drone |
| `is_in_demilitarized_zone(pos) -> bool` | Whether a point is inside a DMZ |

`obs` is a `float32` vector with `[x, y, vx, vy]` for every friendly drone, followed by every enemy. Its length changes as drones are destroyed.

**UI toggles.** Each `toggle_*` method flips the boolean attribute listed next to it:

| Method | Attribute |
|---|---|
| `toggle_graph` | `show_graph` |
| `toggle_pause` | `paused` |
| `toggle_friend_range` | `show_friend_detection_range` |
| `toggle_enemy_range` | `show_enemy_detection_range` |
| `toggle_dashed` | `show_dashed_lines` |
| `toggle_dmz` | `show_dmz`, which also enables the DMZ rule |
| `toggle_comm_range` | `show_friend_comm_range` |
| `toggle_tacview_export` | `export_to_tacview` |
| `toggle_save_frames` | `save_frames` |
| `toggle_target_lines` | `show_target_lines` |
| `toggle_trajectory` | `show_trajectory` |
| `toggle_debug` | `show_debug` |
| `toggle_triangulation` | `use_triangulation` |
| `toggle_return` | `return_to_base` |

`reset_simulation()` and `exit_env()` back the *Reset* and *Exit* buttons.

**Main attributes:**

- `friend_drones`, `enemy_drones` (the drones alive), `interest_point`, `demilitarized_zones`;
- `episode`, `current_step`, `accum_reward`;
- `enemies_neutralized`, `friends_lost`, `friends_returned`, `sucessful_attacks`;
- `selected_drone`, `leader`, `seed`, `render_fps`.

**Module function.** `compute_global_matrices(drones, "enemy_intensity", "enemy_direction")` fuses the perception of several drones: in each cell, the strongest detection wins.

## The `info` dictionary

| Key | Type | Meaning |
|---|---|---|
| `current_step` | `int` | Steps elapsed in the episode |
| `accum_reward` | `float` | Sum of the step rewards |
| `enemies_shotdown` | `int` | Enemies destroyed in engagements |
| `friends_shotdown` | `int` | Friendly drones destroyed (engagements, plus radars destroyed by attacks) |
| `sucessful_attacks` | `int` | Enemies that reached the interest point |
| `interest_point_health` | `int` | Remaining health of the asset |
| `state_percentages` | `dict[str, float]` | Average share (%) of steps the living drones spent in each behaviour label |
| `total_distance_traveled` | `float` | Sum of the distances flown by the living drones |
| `friends_returned` | `int` | Drones that landed with *return to base* |
| `enemies_remaining` / `friends_remaining` | `int` | Drones still alive |

## Behaviours: `DroneSwarm2D.behaviorsDefault`

| Name | Description |
|---|---|
| `BehaviorType` | `StrEnum` with `COMMON`, `AEW`, `RADAR` and `AI`; members compare equal to their names |
| `BaseBehavior(behavior_type)` | Abstract base class; implement `apply(state, joystick_controlled=False) -> (info, velocity)`. Attribute `type` holds the role as a string |
| `FriendCommonBehaviorDefault(friend_activation_threshold_position=0.7, enemy_activation_threshold_position=0.4)` | Greedy assignment + interception + hold states (see [architecture](architecture.md#the-default-interceptor-policy)) |
| `FriendAEWBehaviorDefault(...)` | Orbit at `AEW_RANGE` |
| `FriendRadarBehaviorDefault(...)` | Stay still |

See the [behaviour guide](behaviors.md) for the `state` dictionary and examples.

## Drones and scenery

### `FriendDrone(interest_point_center, position, behavior=None, fixed=False, broken=False)`

**Attributes:**

- motion: `pos`, `vel`, `trajectory`, `distance_traveled`;
- identity and role: `drone_id`, `behavior`, `behavior_type` (property), `fixed`, `broken`, `is_leader`, `selected`;
- communication: `neighbors`;
- perception: `enemy_intensity`, `enemy_direction`, `enemy_timestamp`, `friend_intensity`, `friend_direction`, `friend_timestamp`, `passive_detection_matrix` (built on demand), `direction_vectors`;
- behaviour output: `info`, `state_history`.

**Methods:**

| Method | Description |
|---|---|
| `update(enemy_drones, friend_drones, use_triangulation, return_to_base=False)` | One full step |
| `get_state() -> dict` | The `state` passed to the behaviour |
| `get_state_percentages() -> dict` | Share of steps per behaviour label |
| `update_neighbors(all_drones)` | Chooses the communication links |
| `communication(all_drones)` | Merges the matrices with the neighbours |
| `merge_enemy_matrix(other)` / `merge_friend_matrix(other)` | Copies the cells where `other` is newer |
| `decay_matrices()` | Applies `DECAY_FACTOR` to the intensities |
| `update_local_enemy_detection(friends, enemies)` / `update_local_friend_detection(friends)` | Sensing |
| `update_passive_detection(enemies)` | Traces the lines of sight |
| `draw(surface, show_detection=True, show_comm_range=True, show_trajectory=False, show_debug=False)` | Draws the drone |

**Class methods.** `FriendDrone.set_class_seed(seed=None)` seeds the shared generator and `FriendDrone.reset_clock()` resets the logical clock. The environment calls both on every `reset()`.

### `EnemyDrone(interest_point_center, position=None, behavior_type=None, fixed=False)`

- **Constants.** `ENEMY_BEHAVIORS` lists every pattern name; `RANDOM_ENEMY_BEHAVIORS` lists the ones drawn when `behavior_type` is `None`. Both are defined in `DroneSwarm2D.core.EnemyDrone`.
- **Attributes:** `pos`, `vel`, `drone_id`, `behavior_type`, `aggressiveness`, `detector`, `desperate_attack`, `escape_steps_count`, `start_delay`, `trajectory`, `info`.
- **Methods:** `update(detector=None)`, `draw(surface, show_detection=True, show_trajectory=False, show_debug=False)` and `EnemyDrone.set_class_seed(seed=None)`.

### `CircleInterestPoint(center, internal_radius, external_radius, color=(0, 255, 0), seed=None)`

Attributes `center`, `internal_radius`, `external_radius`, `health` and `health_ratio` (property); method `draw(surface)`.

### `CircleDMZ(center, radius, color=(0, 0, 255), seed=None)`

Attributes `center`, `radius`, `dmz_id`; methods `contains(pos)` and `draw(surface)`.

## Utilities: `DroneSwarm2D.core.utils`

| Function | Description |
|---|---|
| `pos_to_cell(pos, cell_size=CELL_SIZE, grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT)` | Pixel position → cell `(i, j)` (clamped to the grid) |
| `cell_center(cell, cell_size=CELL_SIZE)` | Cell → centre in pixels |
| `active_cells(matrix, threshold)` | Cells with value ≥ threshold, in row-major order |
| `intercept_direction(chaser_pos, chaser_speed, target_pos, target_vel)` | Interception velocity (pure pursuit if impossible) |
| `can_intercept(chaser_pos, chaser_speed, target_pos, target_vel, point_of_interest=None)` | Interception before the target reaches the point |
| `get_friends_hold(state, friend_activation_threshold_position=0.7, enemy_activation_threshold_position=0.4)` | Friendly cells left unassigned by a greedy assignment |
| `generate_sparse_matrix(shape, max_nonzero=20, seed=None)` | Random sparse intensities plus unit directions |
| `smooth_matrix_with_kernel_10x10(matrix, direction, sigma=1.0, flat_radius=1.0)` | Gaussian "bumps" around positive cells (plots) |
| `symmetrical_flat_topped_gaussian_10x10(value, sigma, flat_radius)` | The 10×10 kernel |
| `sim_to_geo(x, y)` | Pixels → `(longitude, latitude)` using `GEO_TOP_LEFT` / `GEO_BOTTOM_RIGHT` |
| `load_svg_as_surface(path)` | SVG → pygame surface |
| `plot_individual_states_matplotlib(state)` | Interactive 3-D plot of a `state` (debugging) |

`draw_dashed_circle` and `draw_dashed_line` are re-exported from `rendering`.

## Drawing: `DroneSwarm2D.core.rendering`

- `get_font(size, bold=False)` returns a cached font; `clear_caches()` drops the cached objects.
- Drawing helpers:
  - `draw_dashed_circle(surface, color, center, radius, dash_length=5, space_length=5, width=1)`;
  - `draw_dashed_line(surface, color, start, end, width=1, dash_length=5, space_length=5)`;
  - `draw_fading_trajectory(surface, points, color, decay_rate, min_alpha=30, width=2)`;
  - `draw_grid(surface)`, `draw_heatmap(surface, intensity, base_color)`, `draw_direction(surface, intensity, direction, threshold)`, `draw_friend_communication(surface, drones, show_dashed=True)`.
- `Button(rect, text, callback, toggled=None, color=(70, 70, 70))` is the control-panel button.

The drawing helpers and `Button` are also importable from `DroneSwarm2D.core.AirTrafficEnv`, as before.

## Settings: `DroneSwarm2D.core.settings`

- `initialize(config_path=None, fullscreen=True, headless=False, screen_size=None)` loads the configuration (called by `DroneSwarm2D.init`).
- `get_config()` returns the `ConfigManager`, which offers `get(key, default)`, `config[key]`, `key in config`, `keys()`, `to_dict()`, plus the attributes `config_path`, `config_name`, `screen`, `clock` and `headless`.
- `is_initialized()` returns whether the configuration was loaded.
- Every configuration key, plus `screen` and `clock`, is a module attribute. Reading one before initialisation raises a `RuntimeError` that explains what to do.

## Video tool and command line

- `create_video_from_frames(frames_dir, output_dir="./video", fps=30, codec="libx264", remove_frames=False)` encodes `frame_<n>.png` files into `<output_dir>/<scenario>.mp4` and returns the path, or `None` when there is nothing to encode.
- `get_frame_number(filename)` extracts the frame number from a file name.

| Command | Description |
|---|---|
| `python -m DroneSwarm2D` | Same as `init` |
| `python -m DroneSwarm2D init [--name NAME] [--template TEMPLATE]` | Creates a project |
| `python -m DroneSwarm2D create_video PATH [--output-dir DIR] [--fps N] [--codec C] [--remove-frames]` | Encodes saved frames |
| `python -m DroneSwarm2D --version` | Prints the version |

The `droneswarm2d` console command is equivalent to `python -m DroneSwarm2D`.

## Deprecated names

These names still work and will be kept for backward compatibility:

| Deprecated | Use instead |
|---|---|
| `FriendDrone.update_neghbors` | `update_neighbors` |
| `AirTrafficEnv.toogle_save_frames` | `toggle_save_frames` |
| `AirTrafficEnv.toggle_trajetory` | `toggle_trajectory` |
| `AirTrafficEnv.toogle_return` | `toggle_return` |
| `AirTrafficEnv.toogle_triangulation` | `toggle_triangulation` |

The misspelled `info` key `sucessful_attacks` and the config key `CICLE_COMM_BY_STEP` are part of the stable interface and keep their names.
