"""
Machine Learning Model Training Module.

This module serves as the main script for training the LSTM-based financial
forecasting model. It orchestrates the entire training pipeline, including:
- Parsing command-line arguments for configurable training parameters.
- Loading and preprocessing financial time-series data using `MLDataPreprocessor`.
- Initializing the `LSTMModel` architecture.
- Executing the training loop, which involves iterating through epochs and batches,
  performing forward and backward passes, and optimizing model weights.
- Calculating training and validation loss, and implementing early stopping
  to prevent overfitting.
- Saving the state dictionary of the best performing model, along with essential
  preprocessing artifacts like the data scaler and model input size, for later use
  in inference.
"""
# train.py
# This file contains the script for training the LSTM model.

import argparse
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import joblib # For saving the scaler
import json # For saving input_size

# Assuming the script is run from the root of the project or src folder is in PYTHONPATH
# Adjust import paths if necessary based on your project structure
try:
    # Assumes 'src' is in PYTHONPATH or script is run from project root.
    from ml.data_preprocessor import MLDataPreprocessor, FinancialDataset
    from ml.models.lstm_model import LSTMModel
except ModuleNotFoundError:
    # Fallback for direct execution from within 'src/ml/' or if 'src' is not directly in path.
    print("Attempting fallback imports for data_preprocessor and lstm_model.")
    from data_preprocessor import MLDataPreprocessor, FinancialDataset
    from models.lstm_model import LSTMModel


def train_model(args):
    """
    Orchestrates the training process for the LSTM model.

    This function handles data loading, preprocessing, model initialization,
    the training loop (including validation and early stopping), and saving
    the trained model and associated artifacts.

    Args:
        args (argparse.Namespace): An object containing parsed command-line arguments
                                   that define training parameters. Key arguments include:
            - data_path (str): Path to the input CSV data file or "dummy".
            - num_dummy_rows (int): Number of rows for dummy data if `data_path` is "dummy".
            - model_save_path (str): Path to save the trained model's state_dict and
                                     other artifacts (scaler, input_size.json).
            - epochs (int): Maximum number of training epochs.
            - batch_size (int): Batch size for training and validation DataLoaders.
            - learning_rate (float): Learning rate for the Adam optimizer.
            - sequence_length (int): Length of input sequences for the LSTM.
            - hidden_size (int): Number of features in the LSTM hidden state.
            - num_layers (int): Number of stacked LSTM layers.
            - dropout (float): Dropout probability for regularization in the LSTM model.
            - early_stopping_patience (int): Number of epochs to wait for validation
                                             loss improvement before stopping early.
    Key Steps:
        1. Setup: Configures the device (CPU/CUDA) and ensures save directory exists.
        2. Data Handling: Loads data (real or dummy), then uses `MLDataPreprocessor` for:
           - Feature engineering.
           - Chronological splitting into train, validation, and test sets.
           - Scaling features using `MinMaxScaler` (scaler is saved).
           - Creating input sequences and corresponding targets.
        3. DataLoader Creation: Initializes PyTorch `FinancialDataset` and `DataLoader`
           for efficient batching and shuffling of training and validation data.
        4. Model Initialization: Instantiates `LSTMModel`, defines `BCELoss` as the
           criterion, and `Adam` as the optimizer. The model's `input_size` (number
           of features) is determined from the data and saved.
        5. Training Loop:
           - Iterates for the specified number of `epochs`.
           - For each epoch:
             - Training phase: Sets model to `train()` mode, iterates through `train_loader`,
               performs forward pass, computes loss, performs backward pass (backpropagation),
               and updates model weights using the optimizer.
             - Validation phase: Sets model to `eval()` mode, iterates through `val_loader`,
               computes validation loss (without gradient calculation).
           - Prints training and validation loss for the epoch.
        6. Model Saving & Early Stopping:
           - If current validation loss is better than the best recorded so far,
             the model's `state_dict` is saved.
           - If validation loss does not improve for `early_stopping_patience` consecutive
             epochs, training is stopped early.
        7. Completion: Prints final status, including best validation loss and model save path.
    """
    # Device Configuration (CPU or GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Ensure the directory for saving the model exists
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

    # --- Data Loading and Preprocessing ---
    print("\n--- Data Loading and Preprocessing ---")
    preprocessor = MLDataPreprocessor()

    # Load data: either from a specified CSV path or generate dummy data
    if args.data_path == "dummy":
        print(f"Using dummy data with {args.num_dummy_rows} rows.")
        dates = pd.date_range(start='2020-01-01', periods=args.num_dummy_rows, freq='B')
        data_payload = { # Renamed 'data' to 'data_payload' to avoid conflict with 'data_df' later
            'Open': np.random.rand(args.num_dummy_rows) * 100 + 50,
            'High': np.random.rand(args.num_dummy_rows) * 100 + 55,
            'Low': np.random.rand(args.num_dummy_rows) * 100 + 45,
            'Close': np.random.rand(args.num_dummy_rows) * 100 + 50,
            'Volume': np.random.rand(args.num_dummy_rows) * 10000 + 1000,
        }
        raw_df = pd.DataFrame(data_payload, index=dates)
    else:
        print(f"Loading data from: {args.data_path}")
        try:
            raw_df = pd.read_csv(args.data_path, index_col='Date', parse_dates=True)
        except FileNotFoundError:
            print(f"Error: Data file not found at '{args.data_path}'. Please provide a valid path or use 'dummy'.")
            return
        except Exception as e:
            print(f"Error loading data from '{args.data_path}': {e}")
            return

    print(f"Initial raw data shape: {raw_df.shape}")

    # Feature Engineering
    data_df = preprocessor.feature_engineer(raw_df)
    print(f"Data shape after feature engineering: {data_df.shape}")
    if data_df.empty or 'target' not in data_df.columns:
        print("Feature engineering resulted in an empty DataFrame or the 'target' column is missing. Exiting.")
        return

    # Split Data into features (X) and target (y)
    X_df = data_df.drop('target', axis=1)
    y_series = data_df['target']
    # Chronological split into training, validation, and test sets
    X_train_df, X_val_df, X_test_df, y_train_series, y_val_series, y_test_series = \
        preprocessor.split_data(X_df, y_series, test_size=0.2, val_size=0.25)

    print(f"Training set shapes: X={X_train_df.shape}, y={y_train_series.shape}")
    print(f"Validation set shapes: X={X_val_df.shape}, y={y_val_series.shape}")
    print(f"Test set shapes: X={X_test_df.shape}, y={y_test_series.shape}")

    if X_train_df.empty or X_val_df.empty:
        print("Training or validation data is empty after splitting. Check data size and split ratios. Exiting.")
        return

    # Scale Data (features only)
    X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, scaler_object = \
        preprocessor.scale_data(X_train_df, X_val_df, X_test_df)

    # Save the fitted scaler object for use during inference
    scaler_save_path = args.model_save_path.replace(".pth", "_scaler.joblib")
    try:
        joblib.dump(scaler_object, scaler_save_path)
        print(f"Scaler object saved to: {scaler_save_path}")
    except Exception as e:
        print(f"Error saving scaler: {e}. Training will continue, but inference might be affected.")

    # Create Sequences for LSTM input
    X_train_seq, y_train_seq = preprocessor.create_sequences(X_train_scaled_df, y_train_series, args.sequence_length)
    X_val_seq, y_val_seq = preprocessor.create_sequences(X_val_scaled_df, y_val_series, args.sequence_length)
    # X_test_seq, y_test_seq = preprocessor.create_sequences(X_test_scaled_df, y_test_series, args.sequence_length) # For final evaluation

    if X_train_seq.shape[0] == 0 or X_val_seq.shape[0] == 0:
        print("No sequences created for training or validation. This might be due to insufficient "
              "data for the chosen sequence_length after feature engineering and splitting. Exiting.")
        return

    print(f"Training sequences shape: X={X_train_seq.shape}, y={y_train_seq.shape}")
    print(f"Validation sequences shape: X={X_val_seq.shape}, y={y_val_seq.shape}")

    # Create PyTorch Datasets and DataLoaders
    train_dataset = FinancialDataset(X_train_seq, y_train_seq)
    val_dataset = FinancialDataset(X_val_seq, y_val_seq)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) # Shuffle training data
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) # No shuffle for validation

    # --- Model, Loss Function, Optimizer Initialization ---
    print("\n--- Model Initialization ---")
    input_size = X_train_seq.shape[2]  # Number of features per time step in a sequence
    output_size = 1  # Binary classification (predict 0 or 1)

    # Save input_size for the model, crucial for inference
    input_size_save_path = args.model_save_path.replace(".pth", "_input_size.json")
    try:
        with open(input_size_save_path, 'w') as f:
            json.dump({'input_size': input_size}, f)
        print(f"Model input_size ({input_size}) saved to: {input_size_save_path}")
    except Exception as e:
        print(f"Error saving input_size: {e}. Training continues.")

    model = LSTMModel(input_size, args.hidden_size, args.num_layers, output_size, dropout_prob=args.dropout).to(device)
    criterion = nn.BCELoss() # Binary Cross-Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print("Model architecture:")
    print(model)
    print(f"Parameters: input_size={input_size}, hidden_size={args.hidden_size}, num_layers={args.num_layers}, output_size={output_size}")

    # --- Training Loop ---
    print("\n--- Training Loop ---")
    best_val_loss = float('inf')
    epochs_no_improve = 0 # Counter for early stopping

    for epoch in range(args.epochs):
        # Training phase
        model.train() # Set model to training mode
        train_loss_epoch = 0.0
        for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            optimizer.zero_grad() # Clear previous gradients
            outputs = model(batch_features) # Forward pass
            loss = criterion(outputs, batch_labels) # Calculate loss
            loss.backward() # Backward pass (compute gradients)
            optimizer.step() # Update model weights
            train_loss_epoch += loss.item()

        avg_train_loss = train_loss_epoch / len(train_loader)

        # Validation phase
        model.eval() # Set model to evaluation mode
        val_loss_epoch = 0.0
        with torch.no_grad(): # Disable gradient calculations for validation
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss_epoch += loss.item()

        avg_val_loss = val_loss_epoch / len(val_loader)

        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early Stopping and Model Saving logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.model_save_path) # Save the best model
            epochs_no_improve = 0
            print(f"Validation loss decreased to {best_val_loss:.4f}. Model saved to '{args.model_save_path}'")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epoch(s). Best val loss: {best_val_loss:.4f}")
            if epochs_no_improve >= args.early_stopping_patience:
                print(f"Early stopping triggered after {args.early_stopping_patience} epochs without improvement.")
                break # Exit training loop

    print("\n--- Training Finished ---")
    if best_val_loss != float('inf'):
        print(f"Best validation loss achieved: {best_val_loss:.4f}")
        print(f"Best model state_dict saved to '{args.model_save_path}'")
    else:
        print("No model was saved as validation loss did not improve from initial infinity.")


if __name__ == '__main__':
    """
    Entry point for training the LSTM model directly from the command line.

    Parses command-line arguments for various training parameters such as data path,
    model hyperparameters (epochs, batch size, learning rate, sequence length,
    LSTM hidden size, number of layers, dropout), model save path, and early
    stopping patience. After parsing, it calls the `train_model` function.
    """
    parser = argparse.ArgumentParser(description="Train LSTM model for financial time-series prediction.")
    parser.add_argument('--data_path', type=str, default="dummy",
                        help="Path to the input data CSV file or 'dummy' to use generated dummy data.")
    parser.add_argument('--num_dummy_rows', type=int, default=500,
                        help="Number of rows for dummy data generation if data_path is 'dummy'.")
    parser.add_argument('--model_save_path', type=str, default='models/lstm_model_v1.pth',
                        help="Path to save the trained model's state dictionary and associated artifacts.")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Maximum number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training and validation.")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="Learning rate for the Adam optimizer.")
    parser.add_argument('--sequence_length', type=int, default=20,
                        help="Length of input sequences for the LSTM model.")
    parser.add_argument('--hidden_size', type=int, default=50,
                        help="Number of features in the LSTM hidden state.")
    parser.add_argument('--num_layers', type=int, default=2,
                        help="Number of stacked LSTM layers.")
    parser.add_argument('--dropout', type=float, default=0.2,
                        help="Dropout probability for regularization in the LSTM model.")
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help="Number of epochs to wait for validation loss improvement before stopping early.")

    cli_args = parser.parse_args()

    # Print the parsed arguments for verification
    print("--- Training Configuration ---")
    for arg_name, value in sorted(vars(cli_args).items()):
        print(f"- {arg_name}: {value}")
    print("------------------------------")

    train_model(cli_args)
