# Contributing to DroneSwarm2D

🇧🇷 [Versão em português](CONTRIBUTING.pt-BR.md)

Thank you for your interest! Bug reports, ideas, new behaviours, documentation and code are all welcome.

## Setting up

```bash
git clone https://github.com/lucasll37/DroneSwarm2D-lib.git
cd DroneSwarm2D-lib
python -m venv .venv && source .venv/bin/activate   # or: make create_env (conda)
pip install -e ".[dev]"
```

The Cairo library must be installed (see the [README](README.md#installation)).

## Everyday commands

| Command | What it does |
|---|---|
| `make test` | Runs the test suite headless (about 5 s) |
| `make coverage` | Tests with a coverage report |
| `make lint` / `make format` | Checks / auto-fixes the code with ruff |
| `make demo ARGS="--windowed"` | Runs the example |
| `make benchmark` | Measures the simulation speed |

The tests and the benchmark use SDL's dummy video driver, so they run on machines without a display.

## Guidelines

- **Keep backward compatibility.** Users run experiments with their own scripts. Do not rename or remove public names: add the new name and keep the old one as an alias (see [Deprecated names](docs/en/api.md#deprecated-names)).
- **Keep runs reproducible.** Use the seeded generators (`FriendDrone.class_rng`, `EnemyDrone.class_rng`, `env.rng`) rather than the global `random` / `numpy.random`, and never use the wall clock in simulation logic.
- **Measure before and after performance work.** Use `make benchmark`, and check that the results did not change (the same seed must produce the same trace).
- **Tests.** Every bug fix gets a regression test; every feature gets tests. Tests must not open windows or depend on timing.
- **Style.**
  - Code passes `ruff check`; lines are at most 120 characters.
  - Docstrings use the Google style, in English.
  - Public API docstrings end with a short `PT-BR:` summary.
- **Documentation.** Documentation lives in English (`docs/en`, `README.md`) and Brazilian Portuguese (`docs/pt-BR`, `*.pt-BR.md`). Update both languages in the same pull request.
- **Changelog.** Add a line to `CHANGELOG.md` under *Unreleased*. Say so explicitly when a change alters simulation results.

## Pull requests

1. Create a branch from `main`.
2. Make your change with tests and documentation.
3. Run `make lint test`.
4. Open the pull request and describe *what* changed and *why*. If the change affects results, include before/after numbers.
