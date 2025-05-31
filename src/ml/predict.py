"""
Machine Learning Model Prediction Module.

This module is responsible for generating predictions using a pre-trained
LSTM model. It includes functionalities to:
- Load a saved LSTM model's state dictionary and associated preprocessing
  artifacts (e.g., scikit-learn scaler, model input size).
- Preprocess new input data consistently with how the training data was processed.
  This involves feature engineering, scaling, and creating sequences.
- Perform inference using the loaded model to obtain raw predictions.
- Convert these raw predictions into discrete trading signals (e.g., binary
  buy/sell signals).

The primary function `get_predictions` encapsulates this entire process, while
`main_cli` provides a command-line interface for execution.
"""
# predict.py
# This file contains the script for making predictions using the trained LSTM model.

import argparse
import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
import joblib # For loading the scaler
import json # For loading input_size

# Assuming the script is run from the root of the project or src folder is in PYTHONPATH
try:
    from ml.data_preprocessor import MLDataPreprocessor, FinancialDataset
    from ml.models.lstm_model import LSTMModel
except ModuleNotFoundError:
    # Fallback for direct execution or different project structure
    print("Attempting fallback imports for data_preprocessor and lstm_model in predict.py.")
    from data_preprocessor import MLDataPreprocessor, FinancialDataset
    from models.lstm_model import LSTMModel


def get_predictions(
    raw_data_df: pd.DataFrame,
    model_path: str,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    batch_size: int = 32
) -> pd.Series:
    """
    Generates binary prediction signals using a trained LSTM model and its artifacts.

    This function loads a saved LSTM model, its corresponding data scaler, and input
    size. It then preprocesses the input `raw_data_df` (feature engineering, scaling,
    sequence creation) in a manner consistent with the model's training phase.
    Finally, it performs inference and converts model outputs (probabilities) into
    binary signals (0 or 1) based on a 0.5 threshold.

    Args:
        raw_data_df (pd.DataFrame): DataFrame containing recent raw OHLCV data (and
                                    any other features the model was trained on, like RSI).
                                    The data should be recent enough to form at least one
                                    sequence of `sequence_length`.
        model_path (str): Path to the saved PyTorch model state dictionary (`.pth` file).
                          The paths to the scaler (`_scaler.joblib`) and input size
                          (`_input_size.json`) files are derived from this base path
                          by replacing the extension.
        sequence_length (int): The sequence length the LSTM model was trained with.
        hidden_size (int): The LSTM hidden layer size of the trained model.
        num_layers (int): The number of LSTM layers in the trained model.
        dropout (float): The dropout rate used in the trained model.
        batch_size (int, optional): Batch size for creating the DataLoader during
                                    prediction. Defaults to 32.

    Returns:
        pd.Series: A pandas Series containing binary prediction signals (0 or 1),
                   indexed by date. The dates correspond to the end of each input
                   sequence for which a prediction is made. Returns an empty Series
                   if prediction fails or if no valid sequences can be formed from
                   the input data.

    Raises:
        FileNotFoundError: If the model file (`.pth`), scaler file (`_scaler.joblib`),
                           or input size file (`_input_size.json`) cannot be found at
                           their expected locations (derived from `model_path`).
        Exception: For other errors encountered during model loading, preprocessing,
                   or inference (e.g., issues with data dimensions).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    scaler_path = model_path.replace(".pth", "_scaler.joblib")
    input_size_path = model_path.replace(".pth", "_input_size.json")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path} (expected alongside model)")
    if not os.path.exists(input_size_path):
        raise FileNotFoundError(f"Input size file not found: {input_size_path} (expected alongside model)")

    preprocessor = MLDataPreprocessor()

    # Load Scaler and Input Size
    scaler = joblib.load(scaler_path)
    with open(input_size_path, 'r') as f:
        input_size_data = json.load(f)
        input_size = input_size_data['input_size']

    # Load Trained Model
    output_size = 1 # Assuming binary classification model
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob=dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set model to evaluation mode

    # --- Data Preprocessing for Prediction ---
    # Feature engineering (consistent with training)
    data_df_predict = preprocessor.feature_engineer(raw_data_df.copy())

    if data_df_predict.empty or 'target' not in data_df_predict.columns:
        # print("Warning: Feature engineering for prediction data resulted in empty DataFrame or 'target' missing.")
        return pd.Series(dtype=int, name="ML_Signal")

    X_predict_df = data_df_predict.drop('target', axis=1) # Target column not used for prediction input

    if X_predict_df.empty:
        # print("Warning: No feature data available for prediction after feature engineering.")
        return pd.Series(dtype=int, name="ML_Signal")

    # Scaling features using the loaded scaler
    X_predict_scaled_np = scaler.transform(X_predict_df)
    X_predict_scaled_df = pd.DataFrame(X_predict_scaled_np, columns=X_predict_df.columns, index=X_predict_df.index)

    # Create Sequences
    # A dummy target series is needed for the preprocessor's create_sequences structure,
    # but its values are not used for generating X_pred_seq.
    dummy_y_for_sequence_creation = pd.Series(np.zeros(len(X_predict_scaled_df)), index=X_predict_scaled_df.index)
    X_pred_seq, _ = preprocessor.create_sequences(
        X_predict_scaled_df, dummy_y_for_sequence_creation, sequence_length
    )

    if X_pred_seq.shape[0] == 0:
        # print("Warning: No sequences created from the prediction data. Input data might be too short.")
        return pd.Series(dtype=int, name="ML_Signal")

    # Determine the dates for which predictions will be made.
    # Predictions correspond to the end of each sequence.
    prediction_dates = X_predict_scaled_df.index[sequence_length - 1:]

    # Create PyTorch Dataset and DataLoader
    # Dummy labels are used as they are not needed for inference.
    dummy_labels_for_predict_dataset = np.zeros(X_pred_seq.shape[0])
    predict_dataset = FinancialDataset(X_pred_seq, dummy_labels_for_predict_dataset)
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)

    # --- Inference Loop ---
    predictions_binary_list = []
    with torch.no_grad(): # Disable gradient calculations during inference
        for batch_features, _ in predict_loader: # Labels from DataLoader are ignored
            batch_features = batch_features.to(device)
            outputs_probs = model(batch_features) # Model output (probabilities if sigmoid is last layer)
            # Convert probabilities to binary predictions (0 or 1) using a 0.5 threshold
            binary_preds = (outputs_probs.cpu().numpy().flatten() > 0.5).astype(int)
            predictions_binary_list.extend(binary_preds.tolist())

    # Align predictions with their corresponding dates
    if len(predictions_binary_list) != len(prediction_dates):
        # This case should ideally not be reached if sequence creation and date tracking are correct.
        # print(f"Warning: Mismatch between number of predictions ({len(predictions_binary_list)}) "
        #       f"and number of prediction dates ({len(prediction_dates)}). Truncating to shorter length.")
        min_len = min(len(predictions_binary_list), len(prediction_dates))
        predictions_binary_list = predictions_binary_list[:min_len]
        prediction_dates = prediction_dates[:min_len]

    predictions_series = pd.Series(predictions_binary_list, index=prediction_dates, name="ML_Signal")
    return predictions_series


def main_cli():
    """
    Command-Line Interface for generating predictions using a trained LSTM model.

    Parses arguments such as model path, data path (or uses dummy data),
    and model hyperparameters (which must match the trained model). It then loads
    the data, calls `get_predictions` to obtain prediction signals, and prints
    the results.
    """
    parser = argparse.ArgumentParser(description="Predict using trained LSTM model.")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the trained model (.pth file). Associated scaler and "
                             "input_size files are expected to be in the same directory "
                             "with derived names.")
    parser.add_argument('--data_path', type=str, default="dummy",
                        help="Path to the input data CSV file for prediction, or 'dummy' "
                             "to use auto-generated dummy data.")
    parser.add_argument('--num_dummy_rows', type=int, default=100,
                        help="Number of rows for dummy data generation if data_path is 'dummy'. "
                             "Ensure this is sufficient for sequence_length.")
    parser.add_argument('--sequence_length', type=int, default=20,
                        help="Sequence length used during model training (must match).")
    # Model hyperparameters should match the loaded model, these are passed to re-initialize it.
    parser.add_argument('--hidden_size', type=int, default=50,
                        help="Number of features in LSTM hidden state (must match trained model).")
    parser.add_argument('--num_layers', type=int, default=2,
                        help="Number of LSTM layers (must match trained model).")
    parser.add_argument('--dropout', type=float, default=0.2,
                        help="Dropout probability used during training (must match trained model).")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for DataLoader during prediction.")
    args = parser.parse_args()

    print("--- Prediction Configuration (CLI) ---")
    for arg_name, value in sorted(vars(args).items()):
        print(f"- {arg_name}: {value}")
    print("------------------------------------")

    if args.data_path == "dummy":
        print("Using dummy data for CLI prediction.")
        # Ensure enough dummy rows for at least one sequence after feature engineering lags
        num_rows = max(args.num_dummy_rows, args.sequence_length + 60) # Added buffer for lags
        dates = pd.date_range(start='2023-01-01', periods=num_rows, freq='B')
        data_payload = {
            'Open': np.random.rand(num_rows) * 100 + 50,
            'High': np.random.rand(num_rows) * 100 + 55,
            'Low': np.random.rand(num_rows) * 100 + 45,
            'Close': np.random.rand(num_rows) * 100 + 50,
            'Volume': np.random.rand(num_rows) * 10000 + 1000,
        }
        raw_df_for_prediction = pd.DataFrame(data_payload, index=dates)
    else:
        print(f"Loading data for CLI prediction from: {args.data_path}")
        try:
            raw_df_for_prediction = pd.read_csv(args.data_path, index_col='Date', parse_dates=True)
        except FileNotFoundError:
            print(f"Error: Data file not found at '{args.data_path}'. Exiting.")
            return
        except Exception as e:
            print(f"Error loading data from '{args.data_path}': {e}. Exiting.")
            return

    print(f"Raw prediction data input shape: {raw_df_for_prediction.shape}")

    try:
        predictions = get_predictions(
            raw_data_df=raw_df_for_prediction,
            model_path=args.model_path,
            sequence_length=args.sequence_length,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            batch_size=args.batch_size
        )

        if not predictions.empty:
            print("\n--- Generated Predictions ---")
            print(predictions.head(min(20, len(predictions)))) # Print first 20 or fewer
            print(f"\nTotal number of predictions generated: {len(predictions)}")
            # To see all predictions, one might save to CSV, e.g.:
            # predictions.to_csv("ml_predictions.csv")
            # print("Predictions saved to ml_predictions.csv")
        else:
            print("\nNo predictions were generated. This could be due to insufficient input data "
                  "to form sequences or other preprocessing issues.")

    except FileNotFoundError as e:
        print(f"Prediction Error: {e}. Please ensure model and associated files exist.")
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
    print("\n--- Prediction Script Finished ---")


if __name__ == '__main__':
    main_cli()
    # The redundant argparse setup and predict(args) call previously here are removed.
    # main_cli() is now the sole entry point for CLI execution.
