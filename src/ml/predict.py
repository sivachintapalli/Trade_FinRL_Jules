# predict.py
# This file contains the script for making predictions using the trained LSTM model.

import argparse
import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
# from sklearn.preprocessing import MinMaxScaler # No longer fitting scaler here
import joblib # For loading the scaler
import json # For loading input_size

# Assuming the script is run from the root of the project or src folder is in PYTHONPATH
try:
    from ml.data_preprocessor import MLDataPreprocessor, FinancialDataset
    from ml.models.lstm_model import LSTMModel
except ModuleNotFoundError:
    # Fallback for direct execution or different project structure
    from data_preprocessor import MLDataPreprocessor, FinancialDataset
    from models.lstm_model import LSTMModel

# Removed get_input_size_for_model function as it's no longer needed.

def get_predictions(
    raw_data_df: pd.DataFrame,
    model_path: str,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    batch_size: int = 32 # Default batch_size for prediction, can be overridden
) -> pd.Series:
    """
    Generates predictions using a trained LSTM model and associated artifacts.

    Args:
        raw_data_df: Pandas DataFrame with raw OHLCV data (and potentially RSI).
        model_path: Path to the trained model (.pth file).
        sequence_length: Sequence length used during training.
        hidden_size: Hidden size of the LSTM model.
        num_layers: Number of layers in the LSTM model.
        dropout: Dropout probability of the LSTM model.
        batch_size: Batch size for DataLoader during prediction.

    Returns:
        A Pandas Series containing binary predictions (0 or 1), indexed by date.
        The index corresponds to the dates for which predictions could be made.
        Returns an empty Series if prediction fails or no predictions can be made.
    Raises:
        FileNotFoundError: If model, scaler, or input_size file is not found.
        Exception: For other errors during the process.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}") # Less verbose for function use

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    scaler_path = model_path.replace(".pth", "_scaler.joblib")
    input_size_path = model_path.replace(".pth", "_input_size.json")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    if not os.path.exists(input_size_path):
        raise FileNotFoundError(f"Input size file not found at {input_size_path}")

    preprocessor = MLDataPreprocessor()

    # Load Scaler and Input Size
    scaler = joblib.load(scaler_path)
    with open(input_size_path, 'r') as f:
        input_size_data = json.load(f)
        input_size = input_size_data['input_size']

    # Load Trained Model
    output_size = 1 # Binary classification
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob=dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Data Preprocessing for Prediction
    # Note: feature_engineer drops rows with NaNs, including the last row due to target shift.
    # The predictions will correspond to the dates *after* these drops.
    data_df_predict = preprocessor.feature_engineer(raw_data_df.copy())

    if data_df_predict.empty or 'target' not in data_df_predict.columns:
        # print("Warning: Feature engineering for prediction data resulted in an empty DataFrame or 'target' column missing.")
        return pd.Series(dtype=int)

    X_predict_df = data_df_predict.drop('target', axis=1)
    # y_predict_series_actuals = data_df_predict['target'] # Actuals, not strictly needed for prediction output only

    if X_predict_df.empty:
        # print("Warning: No data available for prediction after feature engineering.")
        return pd.Series(dtype=int)

    # Scaling
    X_predict_scaled_np = scaler.transform(X_predict_df)
    X_predict_scaled_df = pd.DataFrame(X_predict_scaled_np, columns=X_predict_df.columns, index=X_predict_df.index)

    # Create Sequences
    # For prediction, we only need X_pred_seq. y_pred_seq_actuals is used if evaluating.
    # We need a dummy y_series for create_sequences if actuals are not used.
    # However, our current create_sequences expects y_series to align with features_df.
    # The dates for predictions will be the index of X_predict_scaled_df from which sequences are made.
    # The last date for which a sequence can be formed is the last date in X_predict_scaled_df.
    # The prediction corresponds to the *end* of each sequence.

    # To get predictions aligned with original dates, we need to track indices carefully.
    # The y_series passed to create_sequences is just for structure, its values don't matter for X_seq generation.
    dummy_y_for_sequence = pd.Series(np.zeros(len(X_predict_scaled_df)), index=X_predict_scaled_df.index)
    X_pred_seq, _ = preprocessor.create_sequences(
        X_predict_scaled_df, dummy_y_for_sequence, sequence_length
    )

    if X_pred_seq.shape[0] == 0:
        # print("Warning: No sequences created from the prediction data.")
        return pd.Series(dtype=int)

    # The dates for the predictions will correspond to the end of each sequence.
    # The index of X_predict_scaled_df starts from some date.
    # If sequence_length is L, the first sequence ends at index L-1 of X_predict_scaled_df.
    # The prediction for this sequence corresponds to the date at index L-1.
    prediction_dates = X_predict_scaled_df.index[sequence_length - 1:]


    # Create Dataset and DataLoader
    # For prediction, labels in FinancialDataset can be dummy if not used for evaluation within this function
    dummy_labels_for_dataset = np.zeros(X_pred_seq.shape[0])
    predict_dataset = FinancialDataset(X_pred_seq, dummy_labels_for_dataset)
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)

    # Inference Loop
    predictions_binary_list = []
    with torch.no_grad():
        for batch_features, _ in predict_loader: # Labels ignored
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            probs = outputs.cpu().numpy().flatten()
            binary = (probs > 0.5).astype(int)
            predictions_binary_list.extend(binary.tolist())

    if len(predictions_binary_list) != len(prediction_dates):
        # This should ideally not happen if logic is correct
        # print(f"Warning: Mismatch between number of predictions ({len(predictions_binary_list)}) and dates ({len(prediction_dates)}).")
        # Attempt to align, but this indicates a potential issue.
        min_len = min(len(predictions_binary_list), len(prediction_dates))
        predictions_binary_list = predictions_binary_list[:min_len]
        prediction_dates = prediction_dates[:min_len]

    predictions_series = pd.Series(predictions_binary_list, index=prediction_dates, name="ML_Signal")
    return predictions_series


def main_cli():
    """ CLI execution function """
    parser = argparse.ArgumentParser(description="Predict using trained LSTM model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model (.pth file). Associated scaler and input_size files will be derived from this path.")
    parser.add_argument('--data_path', type=str, default="dummy", help="Path to the data CSV for prediction or 'dummy'.")
    parser.add_argument('--num_dummy_rows', type=int, default=100, help="Number of rows for dummy data if data_path is 'dummy'.")
    parser.add_argument('--sequence_length', type=int, default=20, help="Sequence length (must match training).")
    parser.add_argument('--hidden_size', type=int, default=50, help="Number of features in LSTM hidden state (must match trained model).")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of LSTM layers (must match trained model).")
    parser.add_argument('--dropout', type=float, default=0.2, help="Dropout probability (must match trained model).")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for prediction.")
    args = parser.parse_args()

    print("Prediction arguments (CLI):")
    for arg, value in sorted(vars(args).items()):
        print(f"- {arg}: {value}")
    print("-" * 30)

    if args.data_path == "dummy":
        print("Using dummy data for CLI prediction.")
        num_rows = max(args.num_dummy_rows, args.sequence_length + 60)
        dates = pd.date_range(start='2023-01-01', periods=num_rows, freq='B')
        data_payload = {
            'Open': np.random.rand(num_rows) * 100 + 50,
            'High': np.random.rand(num_rows) * 100 + 55,
            'Low': np.random.rand(num_rows) * 100 + 45,
            'Close': np.random.rand(num_rows) * 100 + 50,
            'Volume': np.random.rand(num_rows) * 10000 + 1000,
        }
        # Add RSI if feature engineer needs it (crude check)
        temp_preprocessor = MLDataPreprocessor()
        temp_fe_check_df = pd.DataFrame({'Close': [1]*10, 'Volume': [1]*10})
        if 'RSI' in temp_preprocessor.feature_engineer(temp_fe_check_df).columns:
             data_payload['RSI'] = np.random.rand(num_rows) * 100
        raw_df_for_prediction = pd.DataFrame(data_payload, index=dates)
    else:
        print(f"Loading data for CLI prediction from {args.data_path}")
        try:
            raw_df_for_prediction = pd.read_csv(args.data_path, index_col='Date', parse_dates=True)
        except FileNotFoundError:
            print(f"Error: Data file not found at {args.data_path}. Exiting.")
            return
        except Exception as e:
            print(f"Error loading data: {e}. Exiting.")
            return

    print(f"Raw prediction data shape: {raw_df_for_prediction.shape}")

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
            print("\nPredictions (first 20 or less):")
            print(predictions.head(20))
            print(f"\nNumber of predictions generated: {len(predictions)}")
            # Here you could also compare with actuals if y_predict_series was processed from get_predictions
            # For now, this CLI just shows the binary signals.
        else:
            print("No predictions were generated.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")


if __name__ == '__main__':
    # predict(args) # The old main CLI call structure
    main_cli() # New CLI entry point
    parser = argparse.ArgumentParser(description="Predict using trained LSTM model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model (.pth file). Associated scaler and input_size files will be derived from this path.")
    parser.add_argument('--data_path', type=str, default="dummy", help="Path to the data CSV for prediction or 'dummy'.")
    parser.add_argument('--num_dummy_rows', type=int, default=100, help="Number of rows for dummy data if data_path is 'dummy'.")
    # num_dummy_rows_for_input_calc is removed as input_size is now loaded from file
    parser.add_argument('--sequence_length', type=int, default=20, help="Sequence length (must match training).")
    # Model parameters - should match the loaded model
    parser.add_argument('--hidden_size', type=int, default=50, help="Number of features in LSTM hidden state (must match trained model).")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of LSTM layers (must match trained model).")
    parser.add_argument('--dropout', type=float, default=0.2, help="Dropout probability (must match trained model).")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for prediction.")


    args = parser.parse_args()

    print("Prediction arguments:")
    for arg, value in sorted(vars(args).items()):
        print(f"- {arg}: {value}")
    print("-" * 30)

    predict(args)
