#!/usr/bin/env python

import sqlite3
import sqlite_vec
import numpy as np
import os

def get_embedding_by_id(db_file: str, item_id: int):
    """
    Connects to the sqlite-vec database, retrieves a vector by its ID,
    and decodes it back into a NumPy array.

    Args:
        db_file (str): The path to the SQLite database file.
        item_id (int): The row ID of the item to retrieve.
    """
    if not os.path.exists(db_file):
        print(f"Error: Database file not found at '{db_file}'")
        print("Please run the main script to create it first: python ../hugging-cli.py create-db")
        return None

    print(f"Attempting to retrieve embedding for ID: {item_id} from '{db_file}'")
    try:
        with sqlite3.connect(db_file) as con:

            con.enable_load_extension(True)
            sqlite_vec.load(con)
            con.enable_load_extension(False)

            cur = con.cursor()

            cur.execute("SELECT embeddings FROM vec_city_nicknames WHERE rowid = ?", (item_id,))
            result = cur.fetchone()

            if result:
                embedding_blob = result[0]
                print(f"Successfully retrieved blob of size: {len(embedding_blob)} bytes.")

                embedding_vector = np.frombuffer(embedding_blob, dtype=np.float32)

                print("Successfully decoded the vector.")
                print("Vector shape:", embedding_vector.shape)
                print("Vector (first 5 elements):", embedding_vector[:5])
                return embedding_vector
            else:
                print(f"No embedding found for ID: {item_id}")
                return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DB_FILE = os.path.join(script_dir, '..', 'city-nicknames-vec.db')
    ID_TO_QUERY = 100

    retrieved_vector = get_embedding_by_id(DB_FILE, ID_TO_QUERY)
    expected_shape = (384,)
    if retrieved_vector is not None:
        assert retrieved_vector.shape == expected_shape, f"Expected shape {expected_shape}, but got {retrieved_vector.shape}"
        print("Shape test passed!")

    expected_dtype = np.float32
    assert retrieved_vector.dtype == expected_dtype, f"Expected dtype {expected_dtype}, but got {retrieved_vector.dtype}"
    print("dtype test passed!")

    expected_first_five = np.array([0.01499035, 0.09210234, 0.02567635, 0.03007967, 0.1045407], dtype=np.float32)
    np.testing.assert_allclose(retrieved_vector[:5], expected_first_five, rtol=1e-5, atol=1e-8)
    print("Value test passed!")
