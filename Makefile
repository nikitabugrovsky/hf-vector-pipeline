# Makefile for the Hugging Face CLI tool
PYTHON = python
SCRIPT = hugging_cli.py

SOURCE_DATA_FILE ?= your-data.csv
TARGET_DATASET_REPO ?= your-username/your-repo
HF_TOKEN ?= your-huggingface-token
EMBEDDING_MODEL_NAME ?= your-embedding-model


# colors
green := \033[36m
white := \033[0m

.PHONY: help push create-db check-env test-db install venv

default: help

help: ## Prints help for targets with comments.
	@cat $(MAKEFILE_LIST) | grep -E '^[a-zA-Z_-]+:.*?## .*$$' | awk 'BEGIN {FS = ":.*?## "}; {printf "$(green)%-30s$(white) %s\n", $$1, $$2}'

check-env:
	@if [ -z "$(SOURCE_DATA_FILE)" -o "$(SOURCE_DATA_FILE)" = "your-data.csv" ]; then \
		echo "Error: SOURCE_DATA_FILE is not set or is set to default placeholder."; \
		exit 1; \
	fi
	@if [ -z "$(TARGET_DATASET_REPO)" -o "$(TARGET_DATASET_REPO)" = "your-username/your-repo" ]; then \
		echo "Error: TARGET_DATASET_REPO is not set or is set to the default placeholder."; \
		exit 1; \
	fi
	@if [ -z "$(HF_TOKEN)" -o "$(HF_TOKEN)" = "your-huggingface-token" ]; then \
		echo "Error: HF_TOKEN is not set or is set to the default placeholder."; \
		exit 1; \
	fi
	@if [ -z "$(EMBEDDING_MODEL_NAME)" -o "$(EMBEDDING_MODEL_NAME)" = "your-embedding-model" ]; then \
		echo "Error: EMBEDDING_MODEL_NAME is not set or is set to the default placeholder."; \
		exit 1; \
	fi

push: install check-env ## Push the dataset to the Huggingface Hub.
	@echo "--> Running Hugging Face client to push the dataset..."
	@uv run $(PYTHON) $(SCRIPT) push
	@echo "--> Operation complete."

create-db: install ## Create sqlite-vec-db from the dataset.
	@echo "--> Running Hugging Face client to create the database..."
	@uv run $(PYTHON) $(SCRIPT) create-db
	@echo "--> Operation complete."

test-db: install ## Test the embedding db.
	@echo "--> Running embedding DB tests..."
	@uv run pytest -v -s -rA tests/test_query_embedding.py
	@echo "--> Tests complete."

test-cli: install ## Test hugging_cli.py.
	@echo "--> Running cli unit tests..."
	@uv run pytest -v -s -rA tests/test_hugging_cli.py
	@echo "--> Tests complete."

install: venv
	@uv pip install .[dev]

venv:
	@uv venv .venv && . .venv/bin/activate

lint:
	uv run -- black . --check

format:
	uv run -- black .

clean:
	rm -rf .venv .uv-cache hf_vector_pipeline.egg-info build __pycache__ tests/__pycache__
