import unittest
import torch

# Adjust import path
try:
    from src.ml.models.lstm_model import LSTMModel
except ModuleNotFoundError:
    import sys
    sys.path.append('../..') # Add project root
    from src.ml.models.lstm_model import LSTMModel

class TestLSTMModel(unittest.TestCase):

    def test_model_forward_pass(self):
        input_size = 10
        hidden_size = 20
        num_layers = 2
        output_size = 1  # For binary classification with Sigmoid
        dropout_prob = 0.2

        batch_size = 3
        sequence_length = 15

        # Instantiate model
        model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob)
        model.eval() # Set to evaluation mode for testing (affects dropout)

        # Create a dummy input tensor
        dummy_input = torch.randn(batch_size, sequence_length, input_size)

        # Perform a forward pass
        with torch.no_grad(): # Ensure no gradients are calculated during test forward pass
            output = model(dummy_input)

        # Assert output shape
        self.assertEqual(output.shape, (batch_size, output_size))

        # Assert output values are between 0 and 1 (due to Sigmoid activation for output_size=1)
        if output_size == 1:
            self.assertTrue((output >= 0).all() and (output <= 1).all())

    def test_model_forward_pass_multiclass(self):
        input_size = 10
        hidden_size = 20
        num_layers = 2
        output_size = 3  # For multi-class classification (no sigmoid in model)
        dropout_prob = 0.2

        batch_size = 3
        sequence_length = 15

        model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob)
        model.eval()

        dummy_input = torch.randn(batch_size, sequence_length, input_size)
        with torch.no_grad():
            output = model(dummy_input)

        self.assertEqual(output.shape, (batch_size, output_size))
        # For multi-class (output_size > 1), the model doesn't apply sigmoid,
        # so output values are not restricted to [0,1] by the model itself.

if __name__ == '__main__':
    unittest.main()
