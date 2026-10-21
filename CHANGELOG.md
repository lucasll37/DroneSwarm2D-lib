# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

The public API is backward compatible:

- every previous import path, class, function, argument, configuration key and `info` key still works;
- misspelled method names are kept as aliases.

Some **bug fixes change simulation results**. They are listed under *Fixed → Simulation results* below. Re-run your baselines before comparing with numbers produced by 0.1.0.

### Added

- `DroneSwarm2D.init(..., headless=True, screen_size=(w, h))`: run without a window (servers, CI, batch experiments) and fix the virtual screen size so results do not depend on the monitor.
- Deterministic runs: the same configuration, seed and screen size always produce exactly the same episode.
- New `info` keys: `friends_returned`, `enemies_remaining`, `friends_remaining`.
- New environment attributes: `enemies_neutralized`, `friends_lost`, `friends_returned`, `render_fps`.
- `DroneSwarm2D.is_initialized()`; `InterestPoint` / `DemilitarizedZone` are now set after `init()`, as aliases of the circle classes.
- `FriendDrone.get_state()`, `FriendDrone.behavior_type`, `FriendDrone.reset_clock()`, `CircleDMZ.contains()`, `CircleInterestPoint.health_ratio`.
- `utils.active_cells()` and `utils.cell_center()` helpers for writing fast behaviours.
- `EnemyDrone.ENEMY_BEHAVIORS` / `RANDOM_ENEMY_BEHAVIORS`; unknown enemy patterns now emit a warning (they still fall back to `direct`).
- `core/rendering.py`, which holds the drawing helpers and UI widgets, with cached fonts.
- The example and template `main.py` accept `--headless`, `--windowed`, `--episodes`, `--seed` and `--max-steps`.
- CLI: `--version` and a `droneswarm2d` console command.
- Test suite (143 tests, ~92% coverage) and a GitHub Actions workflow (ruff + pytest).
- `benchmarks/benchmark.py` and the `make test | coverage | lint | format | benchmark | demo-headless | install-dev` targets.
- Bilingual documentation (English by default, Brazilian Portuguese): README, architecture, configuration reference, behaviour guide, API reference, contributing guide, and the template/example guides.
- `[analysis]` and `[dev]` optional dependency groups.

### Changed

- **Performance.** About 7× faster steps at identical results: 122 → 15.7 ms per step for 24 vs 24 drones headless, and 220 → 30 ms per rendered frame. The main changes:
  - perception scans, neighbour selection, sensor coverage, line-of-sight tracing, matrix merges and the 3-D smoothing are vectorised;
  - the 30 MB-per-drone triangulation grid is built only when read;
  - fonts are cached;
  - trajectories are drawn on cropped surfaces.
- Detection timestamps use a logical simulation clock instead of the wall clock.
- Communication neighbours are listed in a deterministic order (they used to come from a `set`).
- Code comments and docstrings are in English, with Portuguese summaries on the public API.
- Core dependencies are reduced to the five packages the library imports. pandas, seaborn, SciPy, scikit-learn, OpenCV, Pillow and ipykernel moved to the `analysis` extra; `requirements.txt` still pins the full research environment. Version ranges are now open-ended within the major version.
- CLI messages are in English.
- `matplotlib.pyplot` is no longer imported by the environment, which makes imports faster and avoids GUI backends.
- `example/src` and the project template share the same code; `make demo` no longer needs the `sys.path` hack.

### Fixed

#### Simulation results

These fixes change the numbers produced by an episode.

- **AEW engagements.** AEW drones also went through the interceptor rules, giving them twice the intended collision probability. AEW and radars now have the same collision-only rule.
- **`info["enemies_shotdown"]`** counted enemies that *hit* the interest point as shot down, and **`info["friends_shotdown"]`** counted drones that landed with *return to base* as lost. Both now count only real losses; landed drones are in `friends_returned`.
- **Reward.**
  - The kill bonus counted successful attackers as kills.
  - The attack penalty was applied one step late, and was lost when the attack ended the episode.
  - The reward is now computed after attacks and engagements.
- **Default interceptor policy:**
  - the pursuit course used the enemy's unit heading as its velocity, which ignored `ENEMY_SPEED`;
  - the "hold on the line of approach" test compared against the last scanned enemy cell instead of the chosen enemy;
  - the count of friends within communication range used slightly shifted cell centres.
- **Broken drones.**
  - Their fake pattern crashed (`IndexError`) once the drone flew near a border.
  - The fake intensities were written misaligned.
  - The "new" pattern was always the same, and identical for every broken drone.

#### Crashes and other bugs

- Behaviours that set `self.type = BehaviorType.RADAR`, as the old examples did, were not recognised as radars (wrong sensor range and sprite, not destroyed by attacks). `BehaviorType` is now a `StrEnum`.
- `ZeroDivisionError` in `reset()` when there were fewer interceptors than `INITIAL_N_LAYERS`.
- Crashes on zero-length vectors:
  - a drone returning to base exactly at the centre;
  - an enemy escaping from a detector at its own position;
  - an idle drone exactly on its interception point.
- Clicking on a drone when no drone was selected raised `AttributeError`.
- The joystick enemy was created before the enemy RNG was seeded, so it was not reproducible.
- `utils.get_friends_hold()` returned `None` whenever enemies were detected.
- `utils.generate_sparse_matrix(seed=...)` reseeded the global `random` / `numpy.random` state.
- `utils.pos_to_cell()` returned negative indices (wrapping around the grid) for negative positions.
- The passive-detection grid clamped every line of sight to the top-left corner of the fine grid.
- `canvas.tostring_rgb()` was removed in matplotlib 3.10; the graph panel now uses `buffer_rgba()`.
- `datetime.utcnow()` is deprecated; it is replaced by a timezone-aware equivalent.
- `SDL_AUDIODRIVER` was set after pygame was initialised and had no effect.
- `create_video`:
  - crashed on PNG files without a frame number;
  - truncated scenario names containing `-`;
  - did not close the writer on errors.
- `make help` printed nothing.
- The example ran in joystick mode (`"JOYSTICK": "Enemy"`), which hides the friendly drones and the heat-map.
- The template README pointed to an image that does not exist.
- The README mentioned a `config/preset/` folder, a `CONFIG_FILE` variable and an "AI" behaviour that do not exist; `pyproject.toml` pointed to the wrong repository URL.

### Deprecated

- `FriendDrone.update_neghbors` → `update_neighbors`
- `AirTrafficEnv.toogle_save_frames` → `toggle_save_frames`
- `AirTrafficEnv.toggle_trajetory` → `toggle_trajectory`
- `AirTrafficEnv.toogle_return` → `toggle_return`
- `AirTrafficEnv.toogle_triangulation` → `toggle_triangulation`

The old names keep working.

## [0.1.0]

First public version.
