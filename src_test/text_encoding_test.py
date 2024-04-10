import unittest
import torch
from src.encode_test import average_pool


class TestAveragePool(unittest.TestCase):

    def test_average_pool(self):
        # Create some input tensors
        last_hidden_states = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        attention_mask = torch.tensor([[1, 1], [1, 0]])

        # Call the function
        result = average_pool(last_hidden_states, attention_mask)

        # Define expected result
        expected_result = torch.tensor([[1.5, 3.5], [5.0, 6.0]])

        # Check if the result matches the expected result
        self.assertTrue(torch.allclose(result, expected_result))


class TestEncodeText(unittest.TestCase):

    def test_encode_text(self):
        # Create some sample text
        text = "This is a test sentence."

        # Call the function
        result = encode_text(text)

        # Check if the result is a PyTorch tensor
        self.assertIsInstance(result, torch.Tensor)

        # Check the shape of the resulting tensor 
        # 768 is the output dimension of the model
        self.assertEqual(result.shape, (1, 768))


class TestEncHerbert(unittest.TestCase):

    def test_enc_herbert(self):
        # Create a sample DataFrame
        df = pd.DataFrame({
            'text': ['This is a test sentence.', 'Another test sentence.']
        })

        # Call the function
        result_df = enc_herbert(df)

        # Check if the new column '_enc' is added
        self.assertIn('text_enc', result_df.columns)

        # Check if the values in the new column are numpy arrays
        self.assertTrue(all(isinstance(val, np.ndarray) for val in result_df['text_enc']))

        # Check the shape of the numpy arrays
        self.assertEqual(result_df['text_enc'][0].shape, (1, 768))
        self.assertEqual(result_df['text_enc'][1].shape, (1, 768))


if __name__ == '__main__':
    unittest.main()
