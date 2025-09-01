#!/usr/bin/env python

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import argparse
import os
import sqlite3
import sqlite_vec
import numpy as np

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
    parser.add_argument("--db-file",
                        default=os.environ.get("DB_FILE", "city-nicknames-vec.db"),
                        help="The path to the SQLite database file",
                        dest="db_file"
    )
    parser.add_argument("action",
                        nargs="?",
                        choices=["push", "create-db"],
                        help="Action to execute",
    )

    return parser.parse_args()


class HuggingFaceClient:
    def __init__(self, source_data_file: str, target_dataset_repo: str, token: str, model_name: str, db_file: str):
        self.token = token
        self.source_data_file = source_data_file
        self.target_dataset_repo = target_dataset_repo
        self.model = SentenceTransformer(model_name)
        self.db_file = db_file
        login(token=self.token)

        self.final_dataset = self._process_dataset()

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

    def _process_dataset(self):
        """Loads and processes the dataset, returning the final dataset with embeddings."""
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
        return final_dataset


    def push_to_hub(self):
        """
        Pushes the processed dataset to the Hugging Face Hub.
        """
        print(f"Pushing dataset to '{self.target_dataset_repo}'...")
        self.final_dataset.push_to_hub(self.target_dataset_repo)
        print("Successfully pushed dataset to the Hub.")

    def create_vec_db(self):
        """
        Creates a SQLite database with vector search capabilities (sqlite-vec).
        """
        print(f"Creating SQLite vector database at '{self.db_file}'...")
        try:
            with sqlite3.connect(self.db_file) as con:
                con.enable_load_extension(True)
                sqlite_vec.load(con)
                con.enable_load_extension(False)

                cur = con.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS city_nicknames (
                        id INTEGER PRIMARY KEY,
                        city TEXT,
                        nickname TEXT,
                        country TEXT
                    )
                """)

                embedding_dim = len(self.final_dataset['train'][0]['embeddings'])
                cur.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_city_nicknames USING vec0(embeddings float[{embedding_dim}])")

                print("Inserting data and embeddings into the database...")
                for row in self.final_dataset['train']:
                    cur.execute("INSERT INTO city_nicknames (city, nickname, country) VALUES (?, ?, ?)",
                                (row['City'], row['Nickname'], row['Country']))
                    last_id = cur.lastrowid

                    embedding = np.array(row['embeddings'])
                    cur.execute("INSERT INTO vec_city_nicknames (rowid, embeddings) VALUES (?, ?)",
                                (last_id, embedding.astype(np.float32)))


            print(f"Successfully created and populated '{self.db_file}'.")

        except sqlite3.OperationalError as e:
            print(f"A database error occurred: {e}")
            if "load_extension" in str(e):
                print("Please ensure you have the sqlite-vec extension installed and available.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

def main():
    """Main function to parse arguments and run the client."""
    args = parse_arguments()
    if not all([args.source_data_file, args.token, args.model_name]):
        print("Error: Missing required arguments. Ensure SOURCE_DATA_FILE, TOKEN, EMBEDDING_MODEL_NAME are set via arguments or environment variables.")
        exit(1)

    if args.action is None:
        print("No action specified. Please provide an action, e.g., 'push' or 'create-db'.")
        exit(1)

    client = HuggingFaceClient(source_data_file=args.source_data_file,
                               target_dataset_repo=args.target_dataset_repo,
                               token=args.token,
                               model_name=args.model_name,
                               db_file=args.db_file
    )
    if args.action == "push":
        if not args.target_dataset_repo:
            print("Error: --target-dataset-repo is required for the 'push' action.")
            exit(1)
        client.push_to_hub()
    elif args.action == "create-db":
        client.create_vec_db()
    else:
        print(f"Action '{args.action}' is not supported.")
        exit(1)

if __name__ == "__main__":
    main()
