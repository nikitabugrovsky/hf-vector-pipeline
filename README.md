# `hf-vector-pipeline`

## Overview

This repository contains a data pipeline for building, managing, and utilizing Hugging Face datasets with vector embeddings. It provides an automated workflow to generate and push datasets to the Hugging Face Hub.

The primary goal is to create a seamless bridge between raw data, the Hugging Face ecosystem, and local vector search applications.

## Features

-   **Automated Dataset Pushing**: Automatically builds a dataset with embeddings and pushes it to the Hugging Face Hub on changes to source files (`.csv`, `.py`).
-   **CI/CD Integration**: Uses GitHub Actions to manage the entire process of dependency installation, building, and deployment.
-   **Dependency Caching**: The workflow is optimized to cache dependencies for faster execution times.

## Workflow Diagram

The diagram below illustrates the two primary workflows of this pipeline.

```mermaid
graph TD
    subgraph "Push Workflow (Automated via GitHub Actions)"
        A[Local File Change <br/>(.csv, .py)] -- git push --> B{GitHub Workflow Triggered};
        B --> C[Install Dependencies & Cache];
        C --> D[Run 'make push'];
        D --> E[Hugging Face Dataset];
    end
```

## Usage

There are two primary ways to use this pipeline: running it locally from your command line or using the automated GitHub Actions workflow.

### Method 1: Local Execution

To run the pipeline from your local machine, you need to set the required environment variables and then execute the `make push` command.

1.  **Set Environment Variables**: You must export the following variables in your shell.

    ```bash
    export SOURCE_DATA_FILE="city-nicknames.csv"
    export TARGET_DATASET_REPO="your-hf-username/your-dataset-name"
    export EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2" # Or any other Sentence Transformer model
    export HF_TOKEN="hf_..." # Your Hugging Face write token
    ```

2.  **Run the Pipeline**: Once the variables are set, execute the push command:

    ```bash
    make push
    ```
    This will run the script to process your source data, generate embeddings, and push the dataset to your specified Hugging Face repository.

### Method 2: Automated GitHub Workflow (Recommended)

For a more automated, hands-off approach, you can fork this repository and leverage the built-in GitHub Actions workflow.

1.  **Fork the Repository**: Click the **Fork** button at the top-right of this page to create your own copy.

2.  **Update Makefile (Optional)**: If you need to change the source file, target repository, or embedding model from their default values, you can edit the `Makefile` in your forked repository.

3.  **Add GitHub Secret**: The workflow requires your Hugging Face token to authenticate.
    *   In your forked repository, go to `Settings` > `Secrets and variables` > `Actions`.
    *   Click `New repository secret`.
    *   Name the secret `HF_TOKEN`.
    *   Paste your Hugging Face write token as the value.

4.  **Trigger the Workflow**: The workflow will automatically run whenever you push a change to a `.csv` or `.py` file on the `main` branch. Simply commit and push your changes, and the action will handle the rest.
