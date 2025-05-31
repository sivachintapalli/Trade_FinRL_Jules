"""
LSTM Model Architecture Module.

This module defines the `LSTMModel` class, a Long Short-Term Memory (LSTM)
neural network implemented using PyTorch (`torch.nn.Module`). This model is
specifically designed for sequence prediction tasks, making it suitable for
applications like financial time-series forecasting, where understanding
temporal patterns is crucial.

The architecture typically consists of one or more LSTM layers, followed by
dropout for regularization, and a fully connected linear layer to produce
the final output.
"""
# lstm_model.py
# Contains the definition for a basic LSTM model for time series prediction.

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    A Long Short-Term Memory (LSTM) neural network model for sequence modeling.

    This model is composed of one or more LSTM layers, followed by a Dropout layer
    for regularization, and a final fully connected (Linear) layer to map the LSTM
    output to the desired output dimension. If the `output_size` is 1 (common for
    binary classification), a Sigmoid activation function is applied to the output
    of the linear layer.

    Attributes:
        lstm (nn.LSTM): The core LSTM layer(s).
        dropout (nn.Dropout): Dropout layer for regularization.
        fc (nn.Linear): Fully connected layer to produce the final output.
        sigmoid (nn.Sigmoid, optional): Sigmoid activation function, applied if
                                        `output_size` is 1.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout_prob: float = 0.2):
        """
        Initializes the layers of the LSTMModel.

        Args:
            input_size (int): The number of expected features in the input sequences
                              (e.g., number of technical indicators, lagged values).
            hidden_size (int): The number of features in the hidden state of each
                               LSTM cell.
            num_layers (int): The number of recurrent LSTM layers to stack. For example,
                              `num_layers=2` creates a stacked LSTM.
            output_size (int): The dimension of the output from the final linear layer.
                               For binary classification tasks, this is typically 1.
            dropout_prob (float, optional): The dropout probability. This is applied
                                            to the output of the LSTM layers (if `num_layers > 1`,
                                            it's also applied between LSTM layers) and
                                            before the final fully connected layer.
                                            Defaults to 0.2.
        """
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # LSTM layer(s)
        # batch_first=True makes the input/output tensors have shape (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout_prob if num_layers > 1 else 0) # Dropout between LSTM layers if num_layers > 1

        # Dropout layer (applied after LSTM and before FC layer)
        self.dropout = nn.Dropout(dropout_prob)

        # Fully connected linear layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Sigmoid activation for binary classification output
        if output_size == 1:
            self.sigmoid = nn.Sigmoid()
        else:
            # For multi-class classification, Softmax might be used here or handled by the loss function.
            # For regression tasks, no activation or a linear activation might be appropriate.
            self.sigmoid = None # No sigmoid if output_size is not 1


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the LSTM model.

        The input sequence `x` is processed by the LSTM layers. The output from the
        last time step of the final LSTM layer is then passed through a dropout layer
        and a fully connected layer. If configured for binary classification
        (`output_size=1`), a sigmoid activation is applied.

        Args:
            x (torch.Tensor): The input tensor containing a batch of sequences.
                              Shape: `(batch_size, sequence_length, input_size)`.

        Returns:
            torch.Tensor: The output tensor from the model.
                          If `output_size` is 1 (binary classification), the shape is
                          `(batch_size, 1)` and values are probabilities (post-sigmoid).
                          Otherwise, shape is `(batch_size, output_size)` with raw
                          logits from the linear layer.
        """
        # Initialize hidden state (h0) and cell state (c0) with zeros.
        # Shape for h0 and c0: (num_layers, batch_size, hidden_size)
        # These are moved to the same device as the input tensor x.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass input tensor and initial states (h0, c0) through the LSTM layer(s).
        # `out` contains the output features (hidden states) from the last LSTM layer for each time step.
        # Shape of `out`: (batch_size, seq_length, hidden_size)
        # `hn` is the final hidden state for each element in the batch. Shape: (num_layers, batch_size, hidden_size)
        # `cn` is the final cell state for each element in the batch. Shape: (num_layers, batch_size, hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # We are interested in the output of the LSTM from the last time step of the sequence.
        # `out[:, -1, :]` selects all batches, the last time step (-1), and all hidden features.
        # Shape: (batch_size, hidden_size)
        last_time_step_out = out[:, -1, :]

        # Apply dropout to the output of the last time step for regularization.
        out_dropout = self.dropout(last_time_step_out)

        # Pass the dropout-regularized output through the fully connected layer.
        out_fc = self.fc(out_dropout)

        # If a sigmoid layer is defined (i.e., for binary classification with output_size=1),
        # apply it to get probabilities.
        if self.sigmoid:
            out_fc = self.sigmoid(out_fc)

        return out_fc

if __name__ == '__main__':
    # Example Usage
    print("--- Testing LSTMModel ---")
    # Model parameters
    input_dim = 10       # Number of features per time step in the input sequence
    hidden_dim = 20      # Number of features in the LSTM hidden state
    layer_dim = 2        # Number of stacked LSTM layers
    output_dim_binary = 1 # Output dimension for binary classification
    dropout_rate = 0.2

    # Instantiate the model for binary classification
    model_binary = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim_binary, dropout_rate)
    print("\nLSTM Model for Binary Classification Initialized:")
    print(model_binary)

    # Create a dummy input tensor
    # Shape: (batch_size, sequence_length, input_size/features)
    batch_size = 5
    seq_len = 15
    dummy_input_tensor = torch.randn(batch_size, seq_len, input_dim)
    print(f"\nShape of dummy input tensor: {dummy_input_tensor.shape}")

    # Pass the dummy input through the binary classification model
    try:
        output_binary = model_binary(dummy_input_tensor)
        print(f"Shape of binary model output: {output_binary.shape}") # Expected: (batch_size, 1)
        print("Binary model output tensor (first element if batch_size > 0):")
        if batch_size > 0:
            print(output_binary[0])
    except Exception as e:
        print(f"Error during binary model forward pass: {e}")

    # Example with a different output_dim (e.g., for regression or multi-class raw logits)
    output_dim_multi = 3 # Example: 3 output values for regression or 3 classes for multi-class
    model_multi_output = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim_multi, dropout_rate)
    print(f"\nLSTM Model for Multi-Output (dim={output_dim_multi}) Initialized:")
    print(model_multi_output)
    try:
        output_multi = model_multi_output(dummy_input_tensor)
        print(f"Shape of multi-output model output: {output_multi.shape}") # Expected: (batch_size, 3)
        print("Multi-output model output tensor (first element if batch_size > 0):")
        if batch_size > 0:
            print(output_multi[0])

    except Exception as e:
        print(f"Error during multi-output model forward pass: {e}")
    print("\n--- LSTMModel Test Complete ---")
