# -----------------------------------------------------------------------------
# Makefile for Environment Setup, Running Scripts, and Cleaning Up
#
# This Makefile provides targets for creating a conda environment, installing
# system dependencies, running various Python scripts, creating videos,
# and cleaning build files, caches, and GPU memory.
# -----------------------------------------------------------------------------

.PHONY: help create_env apt_deps run analysis bclone video gif traj multtraj compare sandbox clean_gpu clean help

.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# Variables
# -----------------------------------------------------------------------------
VENV_DIR       := ./.venv
REQUIREMENTS   := requirements.txt
PYTHON         := python -u

# -----------------------------------------------------------------------------
# Target: create_env
# Creates a conda environment with Python 3.9 and installs dependencies.
# -----------------------------------------------------------------------------
create_env:
	@echo "Creating conda environment in $(VENV_DIR)..."
	conda create --prefix $(VENV_DIR) python=3.12 -y
	@echo "Activating environment and installing dependencies..."
	@bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(VENV_DIR) && pip install --upgrade pip && pip install -r $(REQUIREMENTS) && echo 'Done.'"

# -----------------------------------------------------------------------------
# Target: apt_deps
# Installs system dependencies (poppler-utils, ffmpeg, libavcodec-extra, graphviz, etc.).
# -----------------------------------------------------------------------------
apt_deps:
	@echo "Installing system dependencies..."
	sudo apt-get update && sudo apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-dri mesa-utils ccache
	@bash -c "conda install libpython-static -y"

# -----------------------------------------------------------------------------
# Target: run
# Runs the main script of the project.
# -----------------------------------------------------------------------------
run:
	@echo "Running the main script..."
	$(PYTHON) ./src/planning/main.py


# -----------------------------------------------------------------------------
# Target: beahvior cloning
# Runs the behavior cloning script.
# -----------------------------------------------------------------------------
ai:
	@echo "Running the behavior clone script..."
	$(PYTHON) ./src/imitation_learning/main.py

# -----------------------------------------------------------------------------
# Target: analysis
# Runs the analysis script.
# -----------------------------------------------------------------------------
analysis:
	@echo "Running the analysis script..."
	$(PYTHON) ./analysis/main.py
# -----------------------------------------------------------------------------
# Target: video
# Runs the video creation script.
# -----------------------------------------------------------------------------
video:
	@echo "Running the video creation script..."
	$(PYTHON) ./utils/create_video.py

# -----------------------------------------------------------------------------
# Target: gif
# Runs the gif creation script.
# -----------------------------------------------------------------------------
gif:
	@echo "Running the video creation script..."
	$(PYTHON) ./utils/create_gif.py

# -----------------------------------------------------------------------------
# Target: traj
# Runs the enemy behavior visualization script.
# -----------------------------------------------------------------------------
traj:
	@echo "Running the enemy behavior visualization script..."
	$(PYTHON) ./utils/view_enemy_behavior.py

# -----------------------------------------------------------------------------
# Target: multtraj
# Runs the multi enemy behavior visualization script.
# -----------------------------------------------------------------------------
multtraj:
	@echo "Running the multi enemy behavior visualization script..."
	$(PYTHON) ./utils/view_multi_enemy_behavior.py

# -----------------------------------------------------------------------------
# Target: compare
# Runs the script to compare different behaviors.
# -----------------------------------------------------------------------------
compare:
	@echo "Running the behavior comparison script..."
	$(PYTHON) ./utils/compare.py

# -----------------------------------------------------------------------------
# Target: sandbox
# Runs the sandbox script.
# -----------------------------------------------------------------------------
build:
	@echo "Compilando módulos Cython selecionados..."
	@bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(VENV_DIR) && \
		python setup.py build_ext"


# -----------------------------------------------------------------------------
# Target: clean_gpu
# Cleans GPU memory by terminating running processes using nvidia-smi.
# -----------------------------------------------------------------------------
clean_gpu:
	@echo "Cleaning GPU memory..."
	@for pid in $$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits); do \
		if [ "$$pid" != "No running processes found" ]; then \
			echo "Killing process $$pid"; \
			kill -9 $$pid || true; \
		fi; \
	done; \
	echo "GPU memory cleaned!"

# -----------------------------------------------------------------------------
# Target: clean
# Cleans generated files and caches.
# -----------------------------------------------------------------------------
clean:
	@echo "Cleaning build files and caches..."
	rm -rf $(VENV_DIR) ./.pytest_cache
	find . -type f -name '*.py[co]' -delete
	find . -type d -name "__pycache__" | xargs rm -rf
	@echo "Clean complete."


# -----------------------------------------------------------------------------
# Target: help
# Displays this help message.
# -----------------------------------------------------------------------------
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'