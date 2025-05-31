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

# Assuming the script is run from the root of the project or src folder is in PYTHONPATH
# Adjust import paths if necessary based on your project structure
try:
    from ml.data_preprocessor import MLDataPreprocessor, FinancialDataset
    from ml.models.lstm_model import LSTMModel
except ModuleNotFoundError:
    # Fallback for direct execution or different project structure
    from data_preprocessor import MLDataPreprocessor, FinancialDataset
    from models.lstm_model import LSTMModel


def train_model(args):
    """
    Main function to train the LSTM model.
    """
    # Device Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Ensure model save directory exists
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

    # --- Data Loading and Preprocessing ---
    preprocessor = MLDataPreprocessor()

    # Load data (Placeholder: Using dummy data similar to data_preprocessor.py for now)
    # In a real scenario, args.data_path would point to a CSV file.
    if args.data_path == "dummy":
        print("Using dummy data for training.")
        dates = pd.date_range(start='2020-01-01', periods=args.num_dummy_rows, freq='B')
        data = {
            'Open': np.random.rand(args.num_dummy_rows) * 100 + 50,
            'High': np.random.rand(args.num_dummy_rows) * 100 + 55,
            'Low': np.random.rand(args.num_dummy_rows) * 100 + 45,
            'Close': np.random.rand(args.num_dummy_rows) * 100 + 50,
            'Volume': np.random.rand(args.num_dummy_rows) * 10000 + 1000,
        }
        # Add RSI if required by feature engineering, though current FE doesn't strictly need it
        if 'RSI' in preprocessor.feature_engineer(pd.DataFrame({'Close': [1,2,3], 'Volume': [1,2,3]})).columns: # cheap check
             data['RSI'] = np.random.rand(args.num_dummy_rows) * 100
        raw_df = pd.DataFrame(data, index=dates)
    else:
        print(f"Loading data from {args.data_path}")
        try:
            raw_df = pd.read_csv(args.data_path, index_col='Date', parse_dates=True)
        except FileNotFoundError:
            print(f"Error: Data file not found at {args.data_path}. Please provide a valid path or use 'dummy'.")
            return
        except Exception as e:
            print(f"Error loading data: {e}")
            return

    print(f"Initial raw data shape: {raw_df.shape}")

    # Feature Engineering
    data_df = preprocessor.feature_engineer(raw_df)
    print(f"Data shape after feature engineering: {data_df.shape}")
    if data_df.empty or 'target' not in data_df.columns:
        print("Feature engineering resulted in an empty DataFrame or 'target' column is missing. Exiting.")
        return

    # Split Data
    X_df = data_df.drop('target', axis=1)
    y_series = data_df['target']
    X_train_df, X_val_df, X_test_df, y_train_series, y_val_series, y_test_series = \
        preprocessor.split_data(X_df, y_series, test_size=0.2, val_size=0.25) # val_size is 25% of (train+val)

    print(f"Train shapes: X={X_train_df.shape}, y={y_train_series.shape}")
    print(f"Validation shapes: X={X_val_df.shape}, y={y_val_series.shape}")
    print(f"Test shapes: X={X_test_df.shape}, y={y_test_series.shape}")

    if X_train_df.empty or X_val_df.empty:
        print("Training or validation data is empty after split. Check data size and split ratios. Exiting.")
        return

    # Scale Data
    X_train_scaled_df, X_val_scaled_df, X_test_scaled_df = \
        preprocessor.scale_data(X_train_df, X_val_df, X_test_df)

    # Create Sequences
    X_train_seq, y_train_seq = preprocessor.create_sequences(X_train_scaled_df, y_train_series, args.sequence_length)
    X_val_seq, y_val_seq = preprocessor.create_sequences(X_val_scaled_df, y_val_series, args.sequence_length)
    # X_test_seq, y_test_seq = preprocessor.create_sequences(X_test_scaled_df, y_test_series, args.sequence_length) # For later evaluation

    if X_train_seq.shape[0] == 0 or X_val_seq.shape[0] == 0:
        print("No sequences created for training or validation. " \
              "This might be due to insufficient data for the given sequence length. Exiting.")
        return

    print(f"Train sequences shape: X={X_train_seq.shape}, y={y_train_seq.shape}")
    print(f"Validation sequences shape: X={X_val_seq.shape}, y={y_val_seq.shape}")

    # Create Datasets and DataLoaders
    train_dataset = FinancialDataset(X_train_seq, y_train_seq)
    val_dataset = FinancialDataset(X_val_seq, y_val_seq)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Model, Loss, Optimizer Initialization ---
    input_size = X_train_seq.shape[2]  # Number of features per time step
    output_size = 1  # For binary classification

    model = LSTMModel(input_size, args.hidden_size, args.num_layers, output_size, dropout_prob=args.dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print("\nModel architecture:")
    print(model)
    print(f"Input size: {input_size}, Hidden size: {args.hidden_size}, Num layers: {args.num_layers}")

    # --- Training Loop ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = args.early_stopping_patience

    print("\nStarting training...")
    for epoch in range(args.epochs):
        # Training Phase
        model.train()
        train_loss_epoch = 0
        for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()

        avg_train_loss = train_loss_epoch / len(train_loader)

        # Validation Phase
        model.eval()
        val_loss_epoch = 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss_epoch += loss.item()

        avg_val_loss = val_loss_epoch / len(val_loader)

        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early Stopping & Model Saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.model_save_path)
            epochs_no_improve = 0
            print(f"Validation loss decreased to {best_val_loss:.4f}. Saving model to {args.model_save_path}")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to {args.model_save_path if best_val_loss != float('inf') else 'No model saved (Val loss did not improve)'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train LSTM model for stock prediction.")
    parser.add_argument('--data_path', type=str, default="dummy", help="Path to the SPY data CSV or 'dummy' for dummy data.")
    parser.add_argument('--num_dummy_rows', type=int, default=500, help="Number of rows for dummy data if data_path is 'dummy'.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for Adam optimizer.")
    parser.add_argument('--sequence_length', type=int, default=20, help="Number of time steps in a sequence.")
    parser.add_argument('--hidden_size', type=int, default=50, help="Number of features in LSTM hidden state.")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument('--dropout', type=float, default=0.2, help="Dropout probability.")
    parser.add_argument('--model_save_path', type=str, default='models/lstm_model_v1.pth', help="Path to save the trained model.")
    parser.add_argument('--early_stopping_patience', type=int, default=10, help="Patience for early stopping.")

    args = parser.parse_args()

    # Print arguments
    print("Training arguments:")
    for arg, value in sorted(vars(args).items()):
        print(f"- {arg}: {value}")
    print("-" * 30)

    train_model(args)
