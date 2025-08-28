# Makefile for the Hugging Face CLI tool

SOURCE_DATA_FILE ?= your-data.csv
TARGET_DATASET_REPO ?= your-username/your-repo
HF_TOKEN ?= your-huggingface-token
EMBEDDING_MODEL_NAME ?= your-embedding-model


# colors
green := \033[36m
white := \033[0m

.PHONY: help push check-env

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

push: check-env ## Push the dataset to the Huggingface Hub.
	@echo "--> Ensuring Python script is executable..."
	@chmod +x ./hugging-cli.py
	@echo "--> Running Hugging Face client to push the dataset..."
	@./hugging-cli.py push
	@echo "--> Operation complete."
