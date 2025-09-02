import os
import sqlite3
import unittest

import numpy as np
import sqlite_vec


class TestQueryEmbedding(unittest.TestCase):

    DB_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "city-nicknames-vec.db"
    )
    ID_TO_QUERY = 100

    def setUp(self):
        """Set up a database connection and retrieve the embedding for tests."""
        if not os.path.exists(self.DB_FILE):
            self.fail(f"Database file not found at '{self.DB_FILE}'")

        try:
            self.con = sqlite3.connect(self.DB_FILE)
            self.con.enable_load_extension(True)
            sqlite_vec.load(self.con)
            self.con.enable_load_extension(False)

            cur = self.con.cursor()
            cur.execute(
                "SELECT embeddings FROM vec_city_nicknames WHERE rowid = ?",
                (self.ID_TO_QUERY,),
            )
            result = cur.fetchone()

            if result:
                embedding_blob = result[0]
                self.retrieved_vector = np.frombuffer(embedding_blob, dtype=np.float32)
            else:
                self.fail(f"No embedding found for ID: {self.ID_TO_QUERY}")

        except Exception as e:
            self.fail(f"An error occurred during setup: {e}")

    def tearDown(self):
        """Close the database connection after tests."""
        if hasattr(self, "con"):
            self.con.close()

    def test_retrieved_embedding_shape(self):
        """Tests if the retrieved vector has the expected shape."""
        expected_shape = (384,)
        self.assertEqual(self.retrieved_vector.shape, expected_shape)

    def test_retrieved_embedding_dtype(self):
        """Tests if the retrieved vector has the expected data type."""
        expected_dtype = np.float32
        self.assertEqual(self.retrieved_vector.dtype, expected_dtype)

    def test_retrieved_embedding_values(self):
        """Tests if the retrieved vector's values are close to the expected values."""
        expected_first_five = np.array(
            [0.01499035, 0.09210234, 0.02567635, 0.03007967, 0.1045407],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            self.retrieved_vector[:5], expected_first_five, rtol=1e-5, atol=1e-8
        )


if __name__ == "__main__":
    unittest.main()
