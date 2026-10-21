.PHONY: help create_env install install-dev demo demo-headless test coverage lint format benchmark clean

.DEFAULT_GOAL := help

VENV_DIR       := ./.venv
REQUIREMENTS   := requirements.txt
PYTHON         := python -u
ARGS           ?=

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

create_env: ## Create a conda env in ./.venv with the pinned dependencies and the dev tools
	@echo "Creating conda environment in $(VENV_DIR)..."
	conda create --prefix $(VENV_DIR) python=3.12 -y
	@echo "Activating environment and installing dependencies..."
	@bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(VENV_DIR) && pip install --upgrade pip && pip install -r $(REQUIREMENTS) && pip install -e '.[dev]' && echo 'Done.'"

install: ## Install the library in editable mode
	pip install -e .

install-dev: ## Install the library plus test and lint tools
	pip install -e ".[dev]"

demo: ## Run the example (options: make demo ARGS="--windowed --episodes 5")
	@echo "Running the example..."
	PYTHONPATH=$(CURDIR) $(PYTHON) ./example/src/main.py $(ARGS)

demo-headless: ## Run the example without a window
	PYTHONPATH=$(CURDIR) $(PYTHON) ./example/src/main.py --headless $(ARGS)

test: ## Run the test suite
	python -m pytest

coverage: ## Run the tests with a coverage report
	python -m pytest --cov=DroneSwarm2D --cov-report=term-missing

lint: ## Check the code with ruff
	ruff check .

format: ## Fix the automatically fixable lint issues
	ruff check . --fix

benchmark: ## Measure the simulation speed (options: make benchmark ARGS="--friends 50 --passive")
	PYTHONPATH=$(CURDIR) $(PYTHON) benchmarks/benchmark.py $(ARGS)

clean: ## Remove the virtual environment, caches and build files
	@echo "Cleaning build files and caches..."
	rm -rf $(VENV_DIR) ./.pytest_cache ./.ruff_cache ./.coverage ./htmlcov ./build ./dist ./*.egg-info
	find . -type f -name '*.py[co]' -delete
	find . -type d -name "__pycache__" | xargs rm -rf
	@echo "Clean complete."
