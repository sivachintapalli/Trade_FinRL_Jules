# predict.py
# This file contains the script for making predictions using the trained LSTM model.

import argparse
import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler # For the temporary scaling solution

# Assuming the script is run from the root of the project or src folder is in PYTHONPATH
try:
    from ml.data_preprocessor import MLDataPreprocessor, FinancialDataset
    from ml.models.lstm_model import LSTMModel
except ModuleNotFoundError:
    # Fallback for direct execution or different project structure
    from data_preprocessor import MLDataPreprocessor, FinancialDataset
    from models.lstm_model import LSTMModel

def get_input_size_for_model(preprocessor: MLDataPreprocessor, sequence_length: int, num_dummy_rows_for_input_calc: int) -> int:
    """
    Determines the input_size for the LSTM model by processing dummy data.
    This is a workaround for not having model configuration saved with the model.
    """
    print("Determining model input_size using dummy data...")
    # Create enough rows for feature engineering (lags) and sequence creation
    # Needs at least sequence_length + lags + 1 rows after feature engineering to get one sequence
    # Default lags in feature_engineer is 5. So, sequence_length + 5.
    # Add a few more for safety margin with dropna.
    required_rows_after_fe = sequence_length
    # feature_engineer shifts by 5 for lags, and 1 for target. Dropna removes these.
    # So, to get `required_rows_after_fe` rows from feature_engineer, we need approx.
    # required_rows_after_fe + 5 (lags) + 1 (target shift) rows fed into it.
    # And create_sequences needs `sequence_length` rows.
    # Let's use a simpler approach: generate more than enough dummy data.

    dummy_raw_data_rows = max(num_dummy_rows_for_input_calc, sequence_length + 20) # Ensure enough data

    dummy_data_dict = {
        'Open': np.random.rand(dummy_raw_data_rows) * 100,
        'High': np.random.rand(dummy_raw_data_rows) * 100,
        'Low': np.random.rand(dummy_raw_data_rows) * 100,
        'Close': np.random.rand(dummy_raw_data_rows) * 100,
        'Volume': np.random.rand(dummy_raw_data_rows) * 1000
    }
    # Check if RSI is used by the current feature engineering logic
    # This is a bit of a hack; ideally feature_engineer would declare its needs
    temp_fe_check_df = pd.DataFrame({'Close': [1,2,3,4,5,6,7,8,9,10], 'Volume': [1,2,3,4,5,6,7,8,9,10]})
    if 'RSI' in preprocessor.feature_engineer(temp_fe_check_df).columns:
        dummy_data_dict['RSI'] = np.random.rand(dummy_raw_data_rows) * 100

    dummy_raw_df = pd.DataFrame(dummy_data_dict)

    dummy_featured_df = preprocessor.feature_engineer(dummy_raw_df)
    if dummy_featured_df.shape[0] < sequence_length:
        raise ValueError(
            f"Dummy data feature engineering resulted in {dummy_featured_df.shape[0]} rows, "
            f"which is less than sequence_length {sequence_length}. "
            f"Increase num_dummy_rows_for_input_calc (currently {num_dummy_rows_for_input_calc})."
        )

    # The target column might be all NaNs if dummy_raw_data_rows is small, handle this.
    # We only need X features for input_size calculation.
    if 'target' in dummy_featured_df:
        dummy_X = dummy_featured_df.drop('target', axis=1)
        dummy_y_series = dummy_featured_df['target'].fillna(0) # fillna just to pass to create_sequences
    else: # Should not happen with current feature_engineer if enough rows
        dummy_X = dummy_featured_df
        dummy_y_series = pd.Series(np.zeros(len(dummy_X)))


    # Temporary scaling for dummy data
    scaler = MinMaxScaler()
    dummy_X_scaled_np = scaler.fit_transform(dummy_X)
    dummy_X_scaled_df = pd.DataFrame(dummy_X_scaled_np, columns=dummy_X.columns, index=dummy_X.index)

    dummy_seq_X, _ = preprocessor.create_sequences(dummy_X_scaled_df, dummy_y_series, sequence_length)

    if dummy_seq_X.shape[0] == 0:
        raise ValueError(
            f"Sequence creation with dummy data resulted in 0 sequences. "
            f"This means not enough data after feature engineering ({dummy_featured_df.shape[0]} rows) "
            f"for sequence_length {sequence_length}. "
            f"Increase num_dummy_rows_for_input_calc (currently {num_dummy_rows_for_input_calc})."
        )
    input_size = dummy_seq_X.shape[2]
    print(f"Determined input_size: {input_size}")
    return input_size

def predict(args):
    """
    Main function to make predictions using the trained LSTM model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    preprocessor = MLDataPreprocessor()

    # --- Determine input_size for model loading ---
    try:
        input_size = get_input_size_for_model(preprocessor, args.sequence_length, args.num_dummy_rows_for_input_calc)
    except ValueError as e:
        print(f"Error determining input_size: {e}")
        return

    # --- Load Trained Model ---
    output_size = 1 # Binary classification
    model = LSTMModel(input_size, args.hidden_size, args.num_layers, output_size, dropout_prob=args.dropout).to(device)

    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure model parameters (hidden_size, num_layers, etc.) match the saved model.")
        return

    model.eval()
    print("Model loaded successfully.")

    # --- Data Loading and Preprocessing for Prediction ---
    if args.data_path == "dummy":
        print("Using dummy data for prediction.")
        # Generate more data for prediction than for input_size calculation
        num_rows = max(args.num_dummy_rows, args.sequence_length + 60) # Ensure enough for FE, sequences, and some batches
        dates = pd.date_range(start='2023-01-01', periods=num_rows, freq='B')
        data = {
            'Open': np.random.rand(num_rows) * 100 + 50,
            'High': np.random.rand(num_rows) * 100 + 55,
            'Low': np.random.rand(num_rows) * 100 + 45,
            'Close': np.random.rand(num_rows) * 100 + 50,
            'Volume': np.random.rand(num_rows) * 10000 + 1000,
        }
        temp_fe_check_df = pd.DataFrame({'Close': [1,2,3,4,5,6,7,8,9,10], 'Volume': [1,2,3,4,5,6,7,8,9,10]})
        if 'RSI' in preprocessor.feature_engineer(temp_fe_check_df).columns:
             data['RSI'] = np.random.rand(num_rows) * 100
        raw_df_predict = pd.DataFrame(data, index=dates)
    else:
        print(f"Loading data for prediction from {args.data_path}")
        try:
            raw_df_predict = pd.read_csv(args.data_path, index_col='Date', parse_dates=True)
        except FileNotFoundError:
            print(f"Error: Data file not found at {args.data_path}. Please provide a valid path or use 'dummy'.")
            return
        except Exception as e:
            print(f"Error loading data: {e}")
            return

    print(f"Raw prediction data shape: {raw_df_predict.shape}")

    # Feature Engineering
    # The 'target' created here will be used as actuals for evaluation if available.
    # For true future prediction, target would not be available for the last row.
    # feature_engineer handles shifting Close by -1 to make target; this means the last row's target is NaN.
    # This NaN target row will be dropped by dropna() in feature_engineer.
    # So, the last true features usable for prediction are for the second to last day of original raw_df_predict.
    data_df_predict = preprocessor.feature_engineer(raw_df_predict)
    print(f"Prediction data shape after feature engineering: {data_df_predict.shape}")

    if data_df_predict.empty or 'target' not in data_df_predict.columns:
        print("Feature engineering for prediction data resulted in an empty DataFrame or 'target' column missing. Exiting.")
        return

    # For prediction, we'd typically use all available data up to the point of prediction.
    # If this is a "test set" evaluation, we use the features and the engineered target as actuals.
    X_predict_df = data_df_predict.drop('target', axis=1)
    y_predict_series = data_df_predict['target'] # These are the 'actuals' for comparison

    if X_predict_df.empty:
        print("No data available for prediction after feature engineering. Exiting.")
        return

    # Scaling (Workaround: Fit a new scaler on the prediction (test) data)
    # THIS IS NOT BEST PRACTICE. A scaler fitted on training data should be saved and loaded.
    print("Applying scaling to prediction data (using a new scaler fitted on this data - for PoC only).")
    scaler_predict = MinMaxScaler()
    X_predict_scaled_np = scaler_predict.fit_transform(X_predict_df)
    X_predict_scaled_df = pd.DataFrame(X_predict_scaled_np, columns=X_predict_df.columns, index=X_predict_df.index)

    # Create Sequences
    X_pred_seq, y_pred_seq_actuals = preprocessor.create_sequences(
        X_predict_scaled_df, y_predict_series, args.sequence_length
    )

    if X_pred_seq.shape[0] == 0:
        print("No sequences created from the prediction data. " \
              "Ensure enough data rows for sequence_length. Exiting.")
        return

    print(f"Prediction sequences shape: X={X_pred_seq.shape}, y_actuals={y_pred_seq_actuals.shape}")

    # Create Dataset and DataLoader
    predict_dataset = FinancialDataset(X_pred_seq, y_pred_seq_actuals)
    # Batch size can be different for prediction, e.g., 1 if predicting one by one, or larger for efficiency.
    predict_loader = DataLoader(predict_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Inference Loop ---
    predictions_probs = []
    predictions_binary = []
    actual_labels = []

    print("\nStarting prediction...")
    with torch.no_grad():
        for batch_features, batch_actual_labels in predict_loader:
            batch_features = batch_features.to(device)

            outputs = model(batch_features) # Model outputs probabilities due to Sigmoid

            probs = outputs.cpu().numpy().flatten()
            binary = (probs > 0.5).astype(int)

            predictions_probs.extend(probs.tolist())
            predictions_binary.extend(binary.tolist())
            actual_labels.extend(batch_actual_labels.cpu().numpy().flatten().tolist())

    print("Prediction finished.")

    # --- Display/Save Predictions ---
    if not predictions_binary:
        print("No predictions were made.")
        return

    print("\nSample Predictions (first 10 or less):")
    for i in range(min(len(predictions_binary), 10)):
        print(f"Actual: {int(actual_labels[i])}, Predicted Prob: {predictions_probs[i]:.4f}, Predicted Label: {predictions_binary[i]}")

    # Calculate and print basic accuracy (if actual labels are meaningful)
    accuracy = np.mean(np.array(predictions_binary) == np.array(actual_labels))
    print(f"\nOverall Accuracy on the provided data: {accuracy*100:.2f}%")

    # For true future prediction (last sequence of available data)
    # You would take the last sequence from X_pred_seq, convert to tensor, and pass to model.
    # Example:
    # last_sequence_features = torch.tensor(X_pred_seq[-1:], dtype=torch.float32).to(device)
    # with torch.no_grad():
    #     future_prob = model(last_sequence_features).cpu().numpy().flatten()[0]
    #     future_label = (future_prob > 0.5).astype(int)
    # print(f"\nPrediction for the next period (based on the last sequence): Prob={future_prob:.4f}, Label={future_label}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict using trained LSTM model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model (.pth file).")
    parser.add_argument('--data_path', type=str, default="dummy", help="Path to the data CSV for prediction or 'dummy'.")
    parser.add_argument('--num_dummy_rows', type=int, default=100, help="Number of rows for dummy data if data_path is 'dummy'.")
    parser.add_argument('--num_dummy_rows_for_input_calc', type=int, default=100, help="Number of dummy rows for input_size calculation.")
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
