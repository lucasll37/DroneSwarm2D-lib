.PHONY: create_env demo clean help

.DEFAULT_GOAL := help

VENV_DIR       := ./.venv
REQUIREMENTS   := requirements.txt
PYTHON         := python -u

create_env:
	@echo "Creating conda environment in $(VENV_DIR)..."
	conda create --prefix $(VENV_DIR) python=3.12 -y
	@echo "Activating environment and installing dependencies..."
	@bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(VENV_DIR) && pip install --upgrade pip && pip install -r $(REQUIREMENTS) && echo 'Done.'"

demo:
	@echo "Running the main script..."
	$(PYTHON) ./example/src/main.py


clean:
	@echo "Cleaning build files and caches..."
	rm -rf $(VENV_DIR) ./.pytest_cache
	find . -type f -name '*.py[co]' -delete
	find . -type d -name "__pycache__" | xargs rm -rf
	@echo "Clean complete."

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'