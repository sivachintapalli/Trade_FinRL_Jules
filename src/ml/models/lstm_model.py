# lstm_model.py
# Contains the definition for a basic LSTM model for time series prediction.

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout_prob: float = 0.2):
        """
        Initializes the LSTMModel.
        Args:
            input_size: The number of expected features in the input x.
            hidden_size: The number of features in the hidden state h.
            num_layers: Number of recurrent layers. E.g., setting num_layers=2 would mean
                        stacking two LSTMs together to form a stacked LSTM,
                        with the second LSTM taking in outputs of the first LSTM and
                        computing the final results.
            output_size: The number of output features. For binary classification, this is typically 1.
            dropout_prob: Dropout probability for the dropout layer.
        """
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # LSTM layer
        # batch_first=True causes input/output tensors to be (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Fully connected linear layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Sigmoid activation for binary classification
        if output_size == 1:
            self.sigmoid = nn.Sigmoid()
        else:
            # For multi-class, Softmax might be used, or handled by loss function (e.g. CrossEntropyLoss)
            # For regression, no activation here or a linear activation.
            self.sigmoid = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the LSTM model.
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).
        Returns:
            Output tensor of shape (batch_size, output_size).
        """
        # Initialize hidden state and cell state with zeros
        # h0 and c0 shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass input and initial states through LSTM layer
        # out: tensor of shape (batch_size, seq_length, hidden_size) - all hidden states
        # hn: tensor of shape (num_layers, batch_size, hidden_size) - last hidden state
        # cn: tensor of shape (num_layers, batch_size, hidden_size) - last cell state
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Take the output of the last time step
        # out[:, -1, :] gives output of all batches at the last time step
        # Shape: (batch_size, hidden_size)
        last_time_step_out = out[:, -1, :]

        # Apply dropout
        out_dropout = self.dropout(last_time_step_out)

        # Pass through the fully connected layer
        out_fc = self.fc(out_dropout)

        # Apply sigmoid if defined (for binary classification)
        if self.sigmoid:
            out_fc = self.sigmoid(out_fc)

        return out_fc

if __name__ == '__main__':
    # Example Usage
    # Model parameters
    input_dim = 10       # Number of features per time step
    hidden_dim = 20      # Number of features in hidden state
    layer_dim = 2        # Number of stacked LSTM layers
    output_dim = 1       # Output dimension (1 for binary classification)
    dropout = 0.2

    # Instantiate the model
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, dropout)
    print("LSTM Model Initialized:")
    print(model)

    # Create a dummy input tensor
    # (batch_size, sequence_length, input_size/features)
    batch_size = 5
    seq_len = 15
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    print(f"\nShape of dummy input: {dummy_input.shape}")

    # Pass the dummy input through the model
    # Ensure the model is in evaluation mode if dropout or batchnorm are used,
    # though for a single forward pass demo it might not strictly matter.
    # model.eval()
    try:
        output = model(dummy_input)
        print(f"Shape of model output: {output.shape}") # Expected: (batch_size, output_dim)
        print("Output tensor (first element if batch > 0):")
        if batch_size > 0:
            print(output[0])
    except Exception as e:
        print(f"Error during model forward pass: {e}")

    # Example with different output_dim (e.g. for regression or multi-class without final activation in model)
    output_dim_multi = 3
    model_multi = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim_multi, dropout)
    print(f"\nLSTM Model for multi-output (dim={output_dim_multi}) Initialized:")
    print(model_multi)
    try:
        output_multi = model_multi(dummy_input)
        print(f"Shape of model_multi output: {output_multi.shape}") # Expected: (batch_size, output_dim_multi)
    except Exception as e:
        print(f"Error during model_multi forward pass: {e}")
