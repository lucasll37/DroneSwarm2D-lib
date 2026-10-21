# Architecture and simulation model

🇧🇷 [Versão em português](../pt-BR/architecture.md) · [Back to the README](../../README.md)

This page describes what happens inside a DroneSwarm2D episode: the world, how drones perceive and communicate, how enemies behave, the combat rules, the reward and the guarantees about reproducibility.

## Contents

- [The world](#the-world)
- [The step loop](#the-step-loop)
- [Perception: grid matrices](#perception-grid-matrices)
- [Detection](#detection)
- [Communication](#communication)
- [Behaviours and motion](#behaviours-and-motion)
- [The default interceptor policy](#the-default-interceptor-policy)
- [Enemies](#enemies)
- [Attacks, engagements and demilitarised zones](#attacks-engagements-and-demilitarised-zones)
- [Reward, termination and metrics](#reward-termination-and-metrics)
- [Faulty drones](#faulty-drones)
- [Randomness and reproducibility](#randomness-and-reproducibility)
- [Code map](#code-map)

## The world

- The screen is split into the **simulation area** (`SIM_WIDTH x SIM_HEIGHT` pixels, the left 70% by default) and the **graph panel** on the right.
- Positions are in pixels, with the origin at the top-left corner and *y* growing downwards. Speeds are in pixels per step, and one step represents `DT_STEP` simulated seconds (used for the clock and the Tacview timestamps).
- The **interest point** sits at the centre of the simulation area. It has three radii:
  - `INTERNAL_RADIUS`: the protected area;
  - `INTEREST_POINT_ATTACK_RANGE`: the distance at which an enemy hits the asset;
  - `EXTERNAL_RADIUS`: friendly drones are never allowed farther than this from the centre.
- Perception uses a coarse **grid** of `CELL_SIZE`-pixel cells (`GRID_WIDTH x GRID_HEIGHT`). `pos_to_cell(pos)` and `cell_center(cell)` convert between pixels and cells.

At the start of an episode (`env.reset()`):

- **ground radars** are placed on a circle of radius `RADAR_RANGE`;
- **AEW drones** are placed on a circle of radius `AEW_RANGE`;
- the remaining **interceptors** are placed on `INITIAL_N_LAYERS` rings starting at `INITIAL_DISTANCE` (40 px apart);
- **enemies** spawn on random points of the four borders.

## The step loop

`AirTrafficEnv.step()` runs, in this order:

1. Every friendly drone calls `FriendDrone.update()`, in list order. A drone therefore already sees the moves and messages of the drones updated before it in the same step.
2. Every enemy calls `EnemyDrone.update(detector)`. Here `detector` is the position of the first friend within `ENEMY_DETECTION_RANGE`, or `None`.
3. Attacks are resolved, then engagements.
4. The reward is computed (`compute_reward()`) and the trajectories are logged for Tacview.
5. The frame is drawn if `mode == "human"`.

`FriendDrone.update()` itself does:

```
decay_matrices()                 forget: intensity *= DECAY_FACTOR
update_neighbors(friends)        choose communication links
update_passive_detection(...)    lines of sight (passive mode only)
update_local_enemy_detection()   sense enemies, clear empty cells
update_local_friend_detection()  sense friends, clear empty cells
communication(friends)           merge matrices with neighbours
take_action()                    behaviour -> velocity -> move -> limits
```

## Perception: grid matrices

Each friendly drone stores six arrays:

| Array | Shape | Content |
|---|---|---|
| `enemy_intensity` | `(GRID_W, GRID_H)` | Recency of an enemy detection: 1 when fresh, multiplied by `DECAY_FACTOR` every step |
| `enemy_direction` | `(GRID_W, GRID_H, 2)` | Unit heading of the enemy seen in the cell |
| `enemy_timestamp` | `(GRID_W, GRID_H)` | Logical time of the last update of the cell |
| `friend_intensity` / `friend_direction` / `friend_timestamp` | same | The same information about friendly drones |

Intensity is therefore an *age*. With `DECAY_FACTOR = 0.99`, a value of 0.4 means the cell was last confirmed about 90 steps ago (0.99⁹⁰ ≈ 0.4).

**Clearing.** After sensing, every cell inside 80% of the sensor range that holds no fresh detection (intensity < 1) is reset to 0 and stamped with the current time. "I looked and saw nothing here" is information too, and it spreads like any other.

**Timestamps.** Since this release, timestamps come from a deterministic *logical clock* that ticks on every drone update, instead of the wall clock. The merge rule is the same as before ("newer wins"), but results no longer depend on how fast the computer runs.

## Detection

**Direct mode** (`PASSIVE_DETECTION = false`):

- Every enemy closer than the drone's sensor range is detected. The range is `FRIEND_DETECTION_RANGE` for interceptors, `AEW_DETECTION_RANGE` for AEW drones and `RADAR_DETECTION_RANGE` for radars.
- The first sighting only stores the enemy's position, because a heading needs two observations.
- From the second sighting on, the enemy's cell gets intensity 1, the observed heading and the current timestamp.
- When the enemy moves to another cell, the cell it left is cleared.

**Passive mode** (`PASSIVE_DETECTION = true`, or the *Tog. D. Passive.* button):

- An enemy is only detected when at least `N_LINE_SIGHT_CROSSING` friendly sensors cover it. This models bearing-only sensors that need several lines of sight crossing to locate a target.
- Each drone also traces its lines of sight on a fine grid (`TRIANGULATION_GRANULARITY` sub-cells per cell). The fine grid is exposed as `passive_detection_matrix` and is built only when that attribute is read.
- The UI draws the lines of sight in debug mode.

**Friends** are sensed the same way, within `COMMUNICATION_RANGE`. AEW drones are not tracked.

## Communication

**Links.** Each step a drone links to its `N_CONNECTIONS` nearest drones within `COMMUNICATION_RANGE`, plus every drone that chose it. All links are therefore symmetric. The neighbour list has a deterministic order: the drone's own choices first, by distance, then the drones that chose it.

**Exchanges.** For `CICLE_COMM_BY_STEP` cycles the drone exchanges matrices with each neighbour. Every exchange is lost with probability `MESSAGE_LOSS_PROBABILITY`, so the extra cycles act as retransmissions.

**Merging.** When drone A receives the matrices of drone B, each cell where B's timestamp is newer is copied to A: intensity, heading and timestamp. Information therefore travels several hops per step, and what drones remember spreads through the swarm with its age.

The UI shows the links as dashed lines, plus the message rate and the number of connections.

## Behaviours and motion

Every step a friendly drone builds a `state` dictionary from its own matrices and calls `behavior.apply(state)`. The call returns `(info, velocity)`:

- `info[0]` is a label (for example `"PURSUING"`). It is drawn under the drone and counted in `state_percentages`.
- `info[1]` and `info[2]` are optional points drawn as debug lines.

After moving, the drone is kept inside the simulation area (bouncing off the borders) and within `EXTERNAL_RADIUS` of the centre. If it would leave that radius, it is put back on the boundary and stops.

A drone's role comes from `behavior.type`:

| `BehaviorType` | Sensor range | Notes |
|---|---|---|
| `COMMON` | `FRIEND_DETECTION_RANGE` | Interceptor (armed) |
| `AEW` | `AEW_DETECTION_RANGE` | Not tracked by other friends; collision-only combat |
| `RADAR` | `RADAR_DETECTION_RANGE` | Created with `fixed=True` (never moves); destroyed when the asset is hit; collision-only combat |
| `AI` | `FRIEND_DETECTION_RANGE` | Reserved for learned policies; behaves like `COMMON` |

*Return to base* (button or `env.return_to_base = True`) overrides the behaviours: every drone flies to the centre and is removed when it arrives.

## The default interceptor policy

`FriendCommonBehaviorDefault` is deterministic. Drones that share the same picture therefore reach the same plan without negotiating:

1. **Targets.** Enemy cells with intensity ≥ 0.4 are sorted by distance to the interest point, most urgent first.
2. **Candidates.** Friendly cells with intensity ≥ 0.7 are the candidates. The drone's own cell is always included.
3. **Greedy assignment.** Each target, in order, gets the closest candidate not yet used.
4. **Pursuit.** If the drone's own cell was assigned, the drone flies the interception course `intercept_direction(pos, FRIEND_SPEED, target, heading * ENEMY_SPEED)` and its label is `PURSUING`.
5. **Hold.** Otherwise it holds:
   - `HOLD - WAIT`: the drone is at the centre, or has nothing better to do.
   - `HOLD - NO ENOUGH COMM`: fewer than `MIN_COMMUNICATION_HOLD` fresh friends are in range, so the drone moves towards the centre to regroup.
   - `HOLD - RETURN`: only when `MIN_COMMUNICATION_HOLD == 0`; the drone moves back to `INITIAL_DISTANCE`.
   - `HOLD - INTCPT` / `GO HOLD INTCPT`: an enemy is heading straight at the centre (cosine ≥ 0.9), so the drone waits on, or moves to (if closer than `THRESHOLD_PROJECTION`), the point of the enemy's path closest to itself.
   - `HOLD - SPREAD` (`HOLD_SPREAD = true`): the drone slides around the centre, away from the other idle drones. Only the tangential part of the repulsion is used.

## Enemies

**Spawning.** Enemies appear on a random border point after a random delay of 0–100 steps (formation patterns use their own schedule).

**Patterns.** With `ENEMY_BEHAVIOR = "None"`, each enemy draws one of the randomised patterns below. A string sets the same pattern for every enemy.

| Pattern | Movement |
|---|---|
| `direct` | Straight line to the target |
| `zigzag`, `zigzag_damped`, `zigzag_unstable`, `zigzag_variable_period` | Lateral sinusoidal oscillation (constant, fading, random, varying period) |
| `spiral`, `spiral_bounce`, `spiral_oscillatory` | Spiral in; periodic outward bounces; extra oscillation |
| `alternating` | 50 steps in, 50 steps sideways |
| `bounce_approach` | 60 steps in, 30 steps back with a random deviation |
| `circle_wait_advance` | Circles in place, then advances |
| `formation`, `biformation` | Rows of 5 entering from the right (or from both sides); then `direct` |
| `focal-direct` | All enemies leave from the top-right corner, one every 20 steps |
| `debug`, `u-debug` | Test patterns (orbit; in-sideways-out) |
| `joystick` | Flown with a game pad (`JOYSTICK = "Enemy"`) |

**Reaction to detection.**

- **Aggressiveness** is `INITIAL_AGGRESSIVENESS` at `EXTERNAL_RADIUS` and grows linearly to 1 at `INITIAL_DISTANCE`.
- When a friend is within `ENEMY_DETECTION_RANGE`, the enemy commits, with probability equal to its aggressiveness, to a **desperate attack**: a straight line to the target for the rest of the episode.
- Otherwise it **escapes** away from the detector for `ESCAPE_STEPS` steps and then resumes its pattern.

## Attacks, engagements and demilitarised zones

**Attacks.** An enemy within `INTEREST_POINT_ATTACK_RANGE + EPSILON` of the centre:

- deals `INTEREST_POINT_DAMAGE` to the asset (health never goes below 0);
- counts as a successful attack;
- is removed from the episode;
- destroys one ground radar, if any is left.

**Engagements.** Every friend/enemy pair closer than `NEUTRALIZATION_RANGE` draws one random number *u*:

| Friend role | Outcome |
|---|---|
| Interceptor | *u* ≤ P(friend alive): enemy destroyed · *u* ≤ P(friend alive) + P(enemy alive): friend destroyed · otherwise: both destroyed |
| AEW or radar | *u* ≤ P(both dead): both destroyed (collision) · otherwise nothing happens |

Here P(friend alive) is `NEUTRALIZATION_PROB_FRIEND_ALIVE`, P(enemy alive) is `NEUTRALIZATION_PROB_ENEMY_ALIVE` and P(both dead) is `NEUTRALIZATION_PROB_BOTH_DEAD`. By default the last one is `1 - (FRIEND_ALIVE + ENEMY_ALIVE)`.

**Demilitarised zones.** While the DMZ layer is enabled (default; *Tog. DMZ* button), enemies inside a zone cannot be engaged. Zones that overlap the protected area are discarded.

## Reward, termination and metrics

The reward of a step is computed after attacks and engagements:

```
reward = Σ distance(enemy, centre) × 1e-4
       + 10 × enemies_shot_down × |centre| × 1e-4
       − 2000 × attacks_in_this_step
```

- `|centre|` is the distance from the origin to the centre.
- The factors are the arguments of `compute_reward()`.
- `accum_reward` is the sum of the step rewards over the episode.

An episode ends when no enemy is left or after `max_steps` steps (5000 by default).

The metrics returned in `info` are listed in the [README](../../README.md#running-experiments) and in the [API reference](api.md#the-info-dictionary).

## Faulty drones

The first `BROKEN_COUNT` interceptors are *broken*:

- After sensing, they overwrite their sensor footprint with a random sparse pattern of fake detections.
- They draw a new pattern every `UPDATE_STATE_BROKEN` calls.
- They share the fake data like any other information, so the effect of faulty nodes on the whole network can be studied.
- Each broken drone has its own seeded pattern generator.

## Randomness and reproducibility

Every source of randomness is seeded from the environment `seed`:

- the class-level generators of `FriendDrone` (message loss) and `EnemyDrone` (spawning, patterns, reactions), re-seeded by every `reset()`;
- the environment's own generator (engagement outcomes);
- the per-drone generators of broken drones.

Detection timestamps use a logical clock and neighbour lists have a deterministic order. Consequently the same configuration, seed and screen size always produce exactly the same episode, regardless of machine speed or rendering mode. `tests/test_environment.py::test_same_seed_same_episode` checks this on every change.

## Code map

| Concept | Where |
|---|---|
| Step loop, attacks, engagements, reward, UI | `core/AirTrafficEnv.py` |
| Perception, communication, motion | `core/FriendDrone.py` |
| Enemy patterns and reactions | `core/EnemyDrone.py` |
| Default policies, `BaseBehavior`, `BehaviorType` | `behaviorsDefault/behaviors.py` |
| Grid helpers, interception maths | `core/utils.py` |
| Drawing | `core/rendering.py` |
| Configuration loading | `core/settings.py` |
