# Writing behaviours

🇧🇷 [Versão em português](../pt-BR/behaviors.md) · [Back to the README](../../README.md)

A **behaviour** is the policy of a friendly drone. Every step, each drone hands its *local* perception to its behaviour and flies the velocity it returns. Writing a new defensive tactic means writing one small class.

## Contents

- [Anatomy of a behaviour](#anatomy-of-a-behaviour)
- [The `state` dictionary](#the-state-dictionary)
- [Helpers](#helpers)
- [Plugging behaviours into the environment](#plugging-behaviours-into-the-environment)
- [Examples](#examples)
- [Tips](#tips)
- [Comparing policies](#comparing-policies)

## Anatomy of a behaviour

```python
import numpy as np
import pygame

from DroneSwarm2D.behaviorsDefault import BaseBehavior, BehaviorType
from DroneSwarm2D.core import settings


class MyBehavior(BaseBehavior):
    def __init__(self):
        super().__init__(BehaviorType.COMMON)     # the drone's role

    def apply(self, state, joystick_controlled=False):
        pos = pygame.math.Vector2(*np.squeeze(state["pos"]).tolist())
        # ... decide ...
        to_centre = settings.INTEREST_POINT_CENTER - pos
        velocity = to_centre.normalize() * settings.FRIEND_SPEED if to_centre.length() > 0 else to_centre
        info = ("GOING HOME", None, None, None)  # (label, target point, reference point, extra)
        return info, velocity
```

**The role.** `BehaviorType` sets the drone's role:

- `COMMON`: interceptor;
- `AEW`: early-warning drone with a different sensor range;
- `RADAR`: fixed sensor;
- `AI`: reserved for learned policies; behaves like `COMMON`.

`BehaviorType` members compare equal to their names (`BehaviorType.RADAR == "RADAR"`), and `BaseBehavior` also accepts the name as a string.

**The return value.** `apply` returns `(info, velocity)`:

- `velocity` is a `pygame.math.Vector2`, in pixels per step. Its length is not checked: keep it at most the drone's speed.
- `info[0]` is a short label. It is shown under the drone and aggregated in `info["state_percentages"]`.
- `info[1]` and `info[2]` are optional `Vector2` points. In debug mode a line is drawn from the drone to `info[1]` and from the interest point to `info[2]`.
- `info[3]` is free for your own data.

**Import order.** Import your behaviours **after** `DroneSwarm2D.init(...)`. Also read settings at call time, as `settings.FRIEND_SPEED` with `from DroneSwarm2D.core import settings`, unless you import them after `init()`.

## The `state` dictionary

All arrays keep a leading batch axis of size 1, which is convenient for neural networks. Use `np.squeeze` to drop it.

| Key | Shape | Content |
|---|---|---|
| `drone_id` | `int` | Id of the drone (unique within the episode) |
| `pos` | `(1, 2)` float32 | Position in pixels |
| `enemy_intensity` | `(1, GRID_W, GRID_H)` | Recency of enemy detections (1 = seen now; decays by `DECAY_FACTOR` per step) |
| `enemy_direction` | `(1, GRID_W, GRID_H, 2)` | Unit heading of the enemy seen in each cell |
| `friend_intensity` | `(1, GRID_W, GRID_H)` | Recency of friend detections |
| `friend_direction` | `(1, GRID_W, GRID_H, 2)` | Unit heading of the friend seen in each cell |

The matrices combine what the drone sensed itself with what it heard from the network. They are **not** the ground truth: positions are quantised to cells, information may be stale, and faulty drones may have injected garbage.

> The arrays are *views* of the drone's own memory. Do not modify them in place.

## Helpers

Everything below lives in `DroneSwarm2D.core.utils`:

| Function | Purpose |
|---|---|
| `active_cells(matrix, threshold)` | List of `(i, j)` cells with value ≥ threshold, in row-major order (fast, vectorised) |
| `cell_center((i, j))` | Centre of a cell, in pixels |
| `pos_to_cell(pos)` | Cell containing a position |
| `intercept_direction(pos, speed, target_pos, target_vel)` | Velocity that intercepts a target moving at constant velocity (falls back to pure pursuit) |
| `can_intercept(pos, speed, target_pos, target_vel, poi)` | Whether the interception happens before the target reaches `poi` |
| `get_friends_hold(state, ...)` | Friendly cells left unassigned by a greedy interception assignment |

The direction matrices store **unit** headings. Multiply them by `settings.ENEMY_SPEED` to obtain the enemy velocity that `intercept_direction` expects.

## Plugging behaviours into the environment

```python
env = DroneSwarm2D.AirTrafficEnv(
    friend_behavior=MyBehavior(),            # interceptors
    friend_aew_behavior=MyAEWBehavior(),     # AEW drones (AEW_COUNT)
    friend_radar_behavior=MyRadarBehavior(), # radars (RADAR_COUNT)
)
```

Any argument left out uses the default policy. **One instance is shared by all drones of a role**, so per-drone memory must be keyed by `state["drone_id"]` (see [example 5](#5-memory-commit-to-a-target)).

## Examples

All examples assume these imports (after `DroneSwarm2D.init(...)`):

```python
import numpy as np
import pygame

from DroneSwarm2D.behaviorsDefault import BaseBehavior, BehaviorType, FriendCommonBehaviorDefault
from DroneSwarm2D.core import settings
from DroneSwarm2D.core.utils import active_cells, can_intercept, cell_center, intercept_direction

V = pygame.math.Vector2


def position(state):
    return V(*np.squeeze(state["pos"]).tolist())
```

### 1. Pure pursuit of the closest enemy

```python
class SimplePursuit(BaseBehavior):
    def __init__(self, threshold=0.4):
        super().__init__(BehaviorType.COMMON)
        self.threshold = threshold

    def apply(self, state, joystick_controlled=False):
        pos = position(state)
        cells = active_cells(np.squeeze(state["enemy_intensity"]), self.threshold)
        if not cells:
            return ("IDLE", None, None, None), V(0, 0)
        target = min((cell_center(c) for c in cells), key=pos.distance_to)
        if target == pos:
            return ("ON TARGET", target, None, None), V(0, 0)
        return ("PURSUING", target, None, None), (target - pos).normalize() * settings.FRIEND_SPEED
```

### 2. Interception, only when it can succeed

```python
class SmartIntercept(BaseBehavior):
    def __init__(self):
        super().__init__(BehaviorType.COMMON)

    def apply(self, state, joystick_controlled=False):
        pos = position(state)
        intensity = np.squeeze(state["enemy_intensity"])
        heading = np.squeeze(state["enemy_direction"])
        center = settings.INTEREST_POINT_CENTER

        # Most urgent threats first.
        for cell in sorted(active_cells(intensity, 0.4), key=lambda c: cell_center(c).distance_to(center)):
            target, enemy_vel = cell_center(cell), V(*heading[cell]) * settings.ENEMY_SPEED
            if can_intercept(pos, settings.FRIEND_SPEED, target, enemy_vel, center):
                vel = intercept_direction(pos, settings.FRIEND_SPEED, target, enemy_vel)
                return ("INTERCEPT", target, None, None), vel
        return ("WAIT", None, None, None), V(0, 0)
```

### 3. Patrol orbit when there is nothing to chase

```python
def orbit(pos, radius, speed, k_radial=0.05):
    """Circle the interest point, pulled back towards `radius`."""
    r_vec = pos - settings.INTEREST_POINT_CENTER
    if r_vec.length() == 0:
        r_vec = V(radius, 0)
    tangent = V(-r_vec.y, r_vec.x).normalize()
    return tangent * speed + k_radial * (radius - r_vec.length()) * r_vec.normalize()


class PatrolOrIntercept(SmartIntercept):
    def __init__(self, patrol_radius=150):
        super().__init__()
        self.patrol_radius = patrol_radius

    def apply(self, state, joystick_controlled=False):
        info, vel = super().apply(state, joystick_controlled)
        if info[0] == "WAIT":
            return ("PATROL", None, None, None), orbit(position(state), self.patrol_radius, settings.FRIEND_SPEED)
        return info, vel
```

### 4. A custom AEW orbit

```python
class TightAEW(BaseBehavior):
    def __init__(self, radius_factor=0.8):
        super().__init__(BehaviorType.AEW)
        self.radius_factor = radius_factor

    def apply(self, state, joystick_controlled=False):
        radius = settings.AEW_RANGE * self.radius_factor
        return ("AEW", None, None, None), orbit(position(state), radius, settings.AEW_SPEED)
```

### 5. Memory: commit to a target

The behaviour instance is shared, so store per-drone data in a dictionary keyed by `drone_id`:

```python
class CommittedPursuit(SimplePursuit):
    def __init__(self, commit_steps=30):
        super().__init__()
        self.commit_steps = commit_steps
        self.memory = {}  # drone_id -> (target, steps left)

    def apply(self, state, joystick_controlled=False):
        drone_id, pos = state["drone_id"], position(state)
        target, left = self.memory.get(drone_id, (None, 0))
        if left > 0 and target.distance_to(pos) > settings.EPSILON:
            self.memory[drone_id] = (target, left - 1)
            return ("COMMITTED", target, None, None), (target - pos).normalize() * settings.FRIEND_SPEED

        info, vel = super().apply(state, joystick_controlled)
        if info[0] == "PURSUING":
            self.memory[drone_id] = (info[1], self.commit_steps)
        return info, vel
```

### 6. Extending the default policy

`FriendCommonBehaviorDefault` handles the greedy assignment and pursuit. Override `_hold_position` to change only what idle drones do:

```python
class DefaultButPatrol(FriendCommonBehaviorDefault):
    def _hold_position(self, pos, friend_intensity, enemy_intensity, enemy_direction, friends_hold=None, **kwargs):
        return ("PATROL", None, None, friends_hold), orbit(pos, settings.INITIAL_DISTANCE, settings.FRIEND_SPEED)
```

## Tips

- **Avoid Python loops over the grid.** A 1920×1080 screen has about 3 600 cells and `apply` runs for every drone at every step. Use `active_cells` or NumPy (`np.argwhere`, masks) instead of `np.ndenumerate`.
- **Keep runs reproducible.** Avoid the global `random` / `numpy.random` state. Use your own seeded generator (`random.Random(seed)`) if the policy is stochastic.
- **Label the states.** The labels feed `state_percentages`, which is the easiest way to see what a policy spends its time doing.
- **Return a velocity of at most the drone's speed.** The engine does not clamp it.
- **Debug visually.** Run with `mode="human"`, keep *Tog. Debug* on and return a target point in `info[1]`.

## Comparing policies

Run each policy headless over the same seeds and compare the metrics:

```python
import DroneSwarm2D

settings = DroneSwarm2D.init("config.json", headless=True, screen_size=(1920, 1080))
from behaviors import SmartIntercept  # your module, imported after init()


def evaluate(behavior, seeds):
    results = []
    for seed in seeds:
        env = DroneSwarm2D.AirTrafficEnv(mode=None, seed=seed, friend_behavior=behavior,
                                         enemy_behavior=settings.ENEMY_BEHAVIOR,
                                         demilitarized_zones=settings.DMZ)
        _, done = env.reset()
        while not done:
            _, _, done, info = env.step(None)
        results.append(info)
    return results


default = evaluate(DroneSwarm2D.behaviorsDefault.FriendCommonBehaviorDefault(), range(20))
custom = evaluate(SmartIntercept(), range(20))
print(sum(r["sucessful_attacks"] for r in default), sum(r["sucessful_attacks"] for r in custom))
```

With the `analysis` extra installed, convert the results to a `pandas.DataFrame` and use SciPy's `mannwhitneyu` / `ks_2samp` to test whether the difference is significant. These are the tests used in the thesis.
