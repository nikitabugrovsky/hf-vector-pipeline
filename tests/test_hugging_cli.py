import argparse
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import hugging_cli


class TestHuggingCli(unittest.TestCase):

    @patch("argparse.ArgumentParser.parse_args")
    def test_parse_arguments(self, mock_parse_args):

        mock_args = argparse.Namespace(
            source_data_file="test.csv",
            target_dataset_repo="user/repo",
            token="test_token",
            model_name="test_model",
            db_file="test.db",
            action="push",
        )
        mock_parse_args.return_value = mock_args

        args = hugging_cli.parse_arguments()

        self.assertEqual(args.source_data_file, "test.csv")
        self.assertEqual(args.target_dataset_repo, "user/repo")
        self.assertEqual(args.token, "test_token")
        self.assertEqual(args.model_name, "test_model")
        self.assertEqual(args.db_file, "test.db")
        self.assertEqual(args.action, "push")

    @patch("hugging_cli.login")
    @patch("hugging_cli.SentenceTransformer")
    @patch("hugging_cli.load_dataset")
    def setUp(self, mock_load_dataset, mock_sentence_transformer, mock_login):

        self.mock_model = MagicMock()
        mock_sentence_transformer.return_value = self.mock_model
        self.mock_dataset = MagicMock()
        self.mock_dataset.__iter__.return_value = iter(
            [
                {
                    "City": "New York",
                    "Nickname": "The Big Apple",
                    "Country": "USA",
                    "embeddings": np.array([0.1, 0.2, 0.3]),
                }
            ]
        )
        self.mock_dataset.push_to_hub = MagicMock()

        mock_load_dataset.return_value.map.return_value.map.return_value.remove_columns.return_value = (
            self.mock_dataset
        )

        self.client = hugging_cli.HuggingFaceClient(
            source_data_file="test.csv",
            target_dataset_repo="user/repo",
            token="test_token",
            model_name="test_model",
            db_file="test.db",
        )
        self.client.final_dataset = MagicMock()
        self.client.final_dataset.push_to_hub = self.mock_dataset.push_to_hub
        self.client.final_dataset.__getitem__.return_value = self.mock_dataset

    def test_structure_data(self):
        data = {"City": "New York", "Nickname": "The Big Apple", "Country": "USA"}
        result = self.client._structure_data(data)
        self.assertEqual(
            result,
            {"combined_text": "City: New York; Nickname: The Big Apple; Country: USA"},
        )

    def test_batch_embeddings(self):
        self.client.model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        batch = {
            "combined_text": ["City: New York; Nickname: The Big Apple; Country: USA"]
        }
        result = self.client._batch_embeddings(batch)
        np.testing.assert_array_equal(result["embeddings"], np.array([[0.1, 0.2, 0.3]]))

    def test_push_to_hub(self):
        self.client.push_to_hub()
        self.mock_dataset.push_to_hub.assert_called_once_with("user/repo")

    @patch("sqlite3.connect")
    @patch("hugging_cli.sqlite_vec")
    def test_create_vec_db(self, mock_sqlite_vec, mock_connect):
        mock_con = MagicMock()
        mock_cur = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_con
        mock_con.cursor.return_value = mock_cur
        mock_cur.lastrowid = 1

        self.client.final_dataset = {
            "train": [
                {
                    "City": "New York",
                    "Nickname": "The Big Apple",
                    "Country": "USA",
                    "embeddings": [0.1, 0.2, 0.3],
                }
            ]
        }

        self.client.create_vec_db()

        mock_connect.assert_called_once_with("test.db")
        mock_con.enable_load_extension.assert_any_call(True)
        mock_sqlite_vec.load.assert_called_once_with(mock_con)
        self.assertTrue(mock_cur.execute.call_count > 0)

    @patch("hugging_cli.parse_arguments")
    @patch("hugging_cli.HuggingFaceClient")
    def test_main_push_action(self, mock_client_class, mock_parse_args):
        mock_args = argparse.Namespace(
            source_data_file="test.csv",
            target_dataset_repo="user/repo",
            token="test_token",
            model_name="test_model",
            db_file="test.db",
            action="push",
        )
        mock_parse_args.return_value = mock_args
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance

        hugging_cli.main()

        mock_client_class.assert_called_once()
        mock_client_instance.push_to_hub.assert_called_once()

    @patch("hugging_cli.parse_arguments")
    @patch("hugging_cli.HuggingFaceClient")
    def test_main_create_db_action(self, mock_client_class, mock_parse_args):
        mock_args = argparse.Namespace(
            source_data_file="test.csv",
            target_dataset_repo=None,
            token="test_token",
            model_name="test_model",
            db_file="test.db",
            action="create-db",
        )
        mock_parse_args.return_value = mock_args
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance

        hugging_cli.main()

        mock_client_class.assert_called_once()
        mock_client_instance.create_vec_db.assert_called_once()


if __name__ == "__main__":
    unittest.main()
