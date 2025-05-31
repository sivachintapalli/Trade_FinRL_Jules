# data_preprocessor.py
# This file will contain code for preprocessing data for the ML model.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch # Added torch import
from torch.utils.data import Dataset, DataLoader

class MLDataPreprocessor:
    def __init__(self):
        """
        Initializes the MLDataPreprocessor.
        Future configurations can be added here.
        """
        pass

    def feature_engineer(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers features for the ML model.
        Args:
            data_df: Pandas DataFrame with OHLCV columns (must include 'Close', 'Volume').
                     'RSI' is optional and will be used if present.
        Returns:
            Pandas DataFrame with new features and the target variable.
        """
        df = data_df.copy()

        # Calculate lagged features for 'Close' and 'Volume'
        for lag in range(1, 6):
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)

        # Calculate price change percentage
        df['pct_change'] = df['Close'].pct_change() * 100

        # Define a target variable: 1 if next day's Close > current day's Close, else 0
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

        # Handle NaNs produced by lagging and target creation
        df.dropna(inplace=True)

        return df

    def split_data(self, features_df: pd.DataFrame, target_series: pd.Series,
                   test_size: float = 0.2, val_size: float = 0.25):
        """
        Splits data into training, validation, and test sets chronologically.
        Args:
            features_df: DataFrame of features.
            target_series: Series for the target variable.
            test_size: Proportion of the dataset to include in the test split.
            val_size: Proportion of the (train + validation) dataset to include in the validation split.
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features_df, target_series, test_size=test_size, shuffle=False
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, shuffle=False
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_data(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
        """
        Scales the feature data using MinMaxScaler.
        The scaler is fitted on X_train only.
        Args:
            X_train: Training features DataFrame.
            X_val: Validation features DataFrame.
            X_test: Test features DataFrame.
        Returns:
            Scaled X_train, X_val, X_test as Pandas DataFrames.
        """
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        return X_train_scaled_df, X_val_scaled_df, X_test_scaled_df

    def create_sequences(self, features_df: pd.DataFrame, target_series: pd.Series, sequence_length: int):
        """
        Creates sequences from time-series data for LSTM or other sequence models.
        Args:
            features_df: Pandas DataFrame of features.
            target_series: Pandas Series of target variable.
            sequence_length: The number of time steps for each sequence.
        Returns:
            np.array of feature sequences, np.array of labels.
        """
        feature_data = features_df.to_numpy()
        target_data = target_series.to_numpy()

        X_sequences, y_sequences = [], []
        # Iterate from sequence_length -1 up to the length of the data minus 1
        # (because target is target_data[i], and features are from i-sequence_length+1 to i)
        for i in range(sequence_length - 1, len(feature_data)):
            # Sequence is from (i - sequence_length + 1) up to i (inclusive)
            sequence = feature_data[i - sequence_length + 1 : i + 1]
            label = target_data[i] # The label corresponds to the end of the sequence
            X_sequences.append(sequence)
            y_sequences.append(label)

        return np.array(X_sequences), np.array(y_sequences)

class FinancialDataset(Dataset):
    """
    PyTorch Dataset for financial time series data.
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: Numpy array of input features (sequences).
            labels: Numpy array of corresponding labels.
        """
        # Convert to torch tensors and ensure correct dtype
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1) # Reshape for BCELoss

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.features)

    def __getitem__(self, idx: int):
        """
        Returns a single sample (feature sequence and label) at the given index.
        """
        return self.features[idx], self.labels[idx]

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    dates = pd.date_range(start='2023-01-01', periods=200, freq='B') # Increased periods for more data
    data = {
        'Open': np.random.rand(200) * 100,
        'High': np.random.rand(200) * 100 + 5,
        'Low': np.random.rand(200) * 100 - 5,
        'Close': np.random.rand(200) * 100,
        'Volume': np.random.rand(200) * 10000,
        'RSI': np.random.rand(200) * 100
    }
    dummy_df = pd.DataFrame(data, index=dates)

    preprocessor = MLDataPreprocessor()

    # 1. Feature Engineering
    featured_df = preprocessor.feature_engineer(dummy_df)
    print("Featured DataFrame:")
    print(featured_df.head())
    print(f"\nShape of featured_df: {featured_df.shape}")

    if not featured_df.empty and 'target' in featured_df.columns:
        y = featured_df['target']
        X = featured_df.drop(columns=['target'])

        # 2. Split Data
        X_train_df, X_val_df, X_test_df, y_train_s, y_val_s, y_test_s = preprocessor.split_data(X, y)
        print(f"\nShapes: X_train_df={X_train_df.shape}, X_val_df={X_val_df.shape}, X_test_df={X_test_df.shape}")

        # 3. Scale Data
        X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.scale_data(X_train_df, X_val_df, X_test_df)
        print("\nScaled X_train_scaled head:")
        print(X_train_scaled.head())

        # 4. Create Sequences
        sequence_length = 10
        X_train_seq, y_train_seq = preprocessor.create_sequences(X_train_scaled, y_train_s, sequence_length)
        X_val_seq, y_val_seq = preprocessor.create_sequences(X_val_scaled, y_val_s, sequence_length)
        X_test_seq, y_test_seq = preprocessor.create_sequences(X_test_scaled, y_test_s, sequence_length)

        if X_train_seq.size > 0:
            print(f"\nShapes after sequence creation (Train): X={X_train_seq.shape}, y={y_train_seq.shape}")
            print(f"Shapes after sequence creation (Val): X={X_val_seq.shape}, y={y_val_seq.shape}")
            print(f"Shapes after sequence creation (Test): X={X_test_seq.shape}, y={y_test_seq.shape}")

            # 5. FinancialDataset and DataLoader
            train_dataset = FinancialDataset(X_train_seq, y_train_seq)
            val_dataset = FinancialDataset(X_val_seq, y_val_seq)
            # test_dataset = FinancialDataset(X_test_seq, y_test_seq) # Example for test

            if len(train_dataset) > 0:
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

                print(f"\nFinancialDataset for training created with {len(train_dataset)} samples.")
                print(f"FinancialDataset for validation created with {len(val_dataset)} samples.")
                # print(f"FinancialDataset for test created with {len(test_dataset)} samples.")

                # Print a sample batch shape
                for batch_features, batch_labels in train_loader:
                    print("\nSample batch from DataLoader:")
                    print(f"Batch features shape: {batch_features.shape}") # (batch_size, seq_len, num_features)
                    print(f"Batch labels shape: {batch_labels.shape}")     # (batch_size, 1)
                    break
            else:
                print("\nTrain dataset is empty. Cannot create DataLoader.")
        else:
            print("\nSequence creation resulted in empty arrays. Check sequence_length and data size.")
    else:
        print("\nFeatured DataFrame is empty or 'target' column missing. Skipping further steps.")
