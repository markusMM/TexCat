import unittest
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from your_script import ca_pca


class TestCAPCA(unittest.TestCase):

    def test_ca_pca(self):
        # Create a sample DataFrame
        df = pd.DataFrame({
            'text_enc': [
                np.array([[1, 2, 3]]), 
                np.array([[4, 5, 6]]),
                np.array([[7, 8, 9]])
            ]
        })

        # Create a PCA model
        pca_model = PCA(n_components=2)
        pca_model.fit(df['text_enc'].sum(axis=0))  # Fit PCA model using the sum of all embeddings

        # Call the function
        result_df = ca_pca(df, pca_model)

        # Check if the new column '_pc' is added
        self.assertIn('text_enc_pc', result_df.columns)

        # Check if the values in the new column are numpy arrays
        self.assertTrue(all(isinstance(val, np.ndarray) for val in result_df['text_enc_pc']))

        # Check the shape of the numpy arrays
        self.assertEqual(result_df['text_enc_pc'][0].shape, (1, 2))  # Assuming n_components=2
        self.assertEqual(result_df['text_enc_pc'][1].shape, (1, 2))  # Assuming n_components=2
        self.assertEqual(result_df['text_enc_pc'][2].shape, (1, 2))  # Assuming n_components=2

if __name__ == '__main__':
    unittest.main()
