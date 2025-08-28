#!/usr/bin/env python

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import argparse
import os

def parse_arguments() -> argparse.Namespace:
    """Sets up and parses command-line arguments."""
    parser = argparse.ArgumentParser(
                        prog="hugging-cli",
                        description="A CLI tool to work with a dataset to be pushed to the huggingface hub",
                        epilog="",)

    parser.add_argument("-s",
                        "--source-data-file",
                        required=False,
                        default=os.environ.get("SOURCE_DATA_FILE"),
                        help="The source data csv file",
                        dest="source_data_file"
    )
    parser.add_argument("-d",
                        "--target-dataset-repo",
                        required=False,
                        default=os.environ.get("TARGET_DATASET_REPO"),
                        help="The huggingface dataset name, for e.g. nikitabugrovsky/city-nicknames",
                        dest="target_dataset_repo",
    )
    parser.add_argument("-t",
                        "--token",
                        required=False,
                        default=os.environ.get("HF_TOKEN"),
                        help="Huggingface token",
                        dest="token",
    )
    parser.add_argument("-m",
                        "--model-name",
                        default=os.environ.get("EMBEDDING_MODEL_NAME"),
                        help="The sentence-transformer model to use",
                        dest="model_name"
    )
    parser.add_argument("action",
                        nargs="?",
                        choices=["push"],
                        help="Action to execute",
    )

    return parser.parse_args()


class HuggingFaceClient:
    """
    A client to process a local dataset and push it to the Hugging Face Hub.

    This class handles loading a CSV file, structuring its content, generating
    sentence embeddings for the text, and uploading the final dataset to a
    specified Hugging Face repository.

    Args:
        source_data_file (str):
            The local file path for the source CSV data.
        target_dataset_repo (str):
            The name of the target repository on the Hugging Face Hub
            (e.g., 'username/my-dataset').
        token (str):
            The Hugging Face API token for authentication.
        model_name (str, optional):
            The name of the sentence-transformer model to use for generating
            embeddings. Defaults to 'all-MiniLM-L6-v2'.
    """

    def __init__(self, source_data_file: str, target_dataset_repo: str, token: str, model_name: str):
        self.token = token
        self.source_data_file = source_data_file
        self.target_dataset_repo = target_dataset_repo
        self.model = SentenceTransformer(model_name)
        login(token=self.token)

    def _structure_data(self, data: dict) -> dict[str, str]:
        """Combine the columns into a single structured string"""
        combined_text = (
            f"City: {data['City']}; "
            f"Nickname: {data['Nickname']}; "
            f"Country: {data['Country']}"
        )
        return {"combined_text": combined_text}

    def _batch_embeddings(self, batch: dict) -> dict[str, list]:
        """Generate embeddings for a batch of combined text"""
        embeddings = self.model.encode(batch["combined_text"])
        return {"embeddings": embeddings}

    def push_to_hub(self):
        """
        Loads, processes, and pushes the dataset to the Hugging Face Hub.
        """
        print(f"Loading dataset from '{self.source_data_file}'...")
        dataset = load_dataset("csv", data_files=self.source_data_file)

        print("Structuring and combining data columns...")
        dataset_with_combined_text = dataset.map(self._structure_data)

        print("Generating embeddings...")
        dataset_with_embeddings = dataset_with_combined_text.map(
            self._batch_embeddings, batch_size=128, batched=True
        )

        print("Clean up intermediate columns...")
        final_dataset = dataset_with_embeddings.remove_columns(["combined_text"])

        print(f"Pushing dataset to '{self.target_dataset_repo}'...")
        final_dataset.push_to_hub(self.target_dataset_repo)
        print("Successfully pushed dataset to the Hub.")


def main():
    """Main function to parse arguments and run the client."""
    args = parse_arguments()
    if not all([args.source_data_file, args.target_dataset_repo, args.token, args.model_name]):
        print("Error: Missing required arguments. Ensure SOURCE_DATA_FILE, TARGET_DATASET_REPO, TOKEN, EMBEDDING_MODEL_NAME are set via arguments or environment variables.")
        exit(1)

    if args.action is None:
        print("No action specified. Please provide an action, e.g., 'push'.")
        exit(1)

    client = HuggingFaceClient(source_data_file=args.source_data_file,
                               target_dataset_repo=args.target_dataset_repo,
                               token=args.token,
                               model_name=args.model_name
    )
    if args.action == "push":
        client.push_to_hub()
    else:
        print(f"Action '{args.action}' is not supported.")
        exit(1)

if __name__ == "__main__":
    main()
