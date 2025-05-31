"""
Machine Learning Data Preprocessing Module.

This module provides the `MLDataPreprocessor` class, which encapsulates various
steps for preparing financial time-series data for machine learning models,
particularly sequence-based models like LSTMs. These steps include:
- Feature engineering: Creating lagged features, percentage changes, and a binary target variable.
- Data splitting: Chronologically splitting data into training, validation, and test sets.
- Data scaling: Applying MinMaxScaler to numerical features.
- Sequence creation: Transforming time-series data into input sequences and corresponding targets.

Additionally, this module includes the `FinancialDataset` class, a PyTorch `Dataset`
implementation tailored for handling the sequenced financial data, making it ready
for use with PyTorch DataLoaders.
"""
# data_preprocessor.py
# This file will contain code for preprocessing data for the ML model.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

class MLDataPreprocessor:
    """
    A class to handle various preprocessing tasks for machine learning on financial data.

    This includes feature engineering, data splitting, scaling, and creating
    sequences suitable for time-series models.
    """
    def __init__(self):
        """
        Initializes the MLDataPreprocessor.

        Currently, the initialization is simple. Future enhancements could include
        configuration options for different preprocessing strategies.
        """
        pass

    def feature_engineer(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers features from raw OHLCV data to create a richer dataset for the ML model.

        Calculates lagged features for 'Close' and 'Volume', percentage change in 'Close',
        and defines a binary target variable based on whether the next day's 'Close'
        price is higher than the current day's 'Close' price.

        Args:
            data_df (pd.DataFrame): Input DataFrame with OHLCV columns. Must include
                                    'Close' and 'Volume'. 'RSI' is optional and
                                    will be used if present (though current implementation
                                    doesn't explicitly use RSI in this method).

        Returns:
            pd.DataFrame: A new DataFrame with the engineered features (e.g.,
                          'close_lag_1', 'volume_lag_1', 'pct_change') and
                          the 'target' variable. Rows with NaN values resulting
                          from lag operations or target creation are dropped.
        """
        df = data_df.copy()

        # Calculate lagged features for 'Close' and 'Volume' for the past 5 days
        for lag in range(1, 6):
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)

        # Calculate daily percentage change in 'Close' price
        df['pct_change'] = df['Close'].pct_change() * 100

        # Define a binary target variable:
        # 1 if the next day's 'Close' price is greater than the current day's 'Close' price,
        # 0 otherwise.
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

        # Drop rows with NaN values that were introduced by shift operations (lags, target).
        df.dropna(inplace=True)

        return df

    def split_data(self, features_df: pd.DataFrame, target_series: pd.Series,
                   test_size: float = 0.2, val_size: float = 0.25):
        """
        Splits the feature and target data into training, validation, and test sets
        chronologically.

        The data is first split into a training+validation set and a test set.
        Then, the training+validation set is further split into a training set and a
        validation set. This chronological splitting is crucial for time-series data
        to prevent data leakage from future periods into past periods.

        Args:
            features_df (pd.DataFrame): DataFrame containing all features.
            target_series (pd.Series): Series containing the target variable.
            test_size (float, optional): Proportion of the entire dataset to allocate
                                         to the test set. Defaults to 0.2 (20%).
            val_size (float, optional): Proportion of the remaining (training + validation)
                                        dataset to allocate to the validation set.
                                        Defaults to 0.25 (meaning 25% of the 80% remaining
                                        after test split, which is 20% of the original).

        Returns:
            tuple: A tuple containing six DataFrames/Series in the order:
                   (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        # First, split into training+validation and test sets (chronological)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features_df, target_series, test_size=test_size, shuffle=False
        )
        # Then, split training+validation into training and validation sets (chronological)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, shuffle=False # val_size is of X_train_val
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_data(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
        """
        Scales numerical features using `MinMaxScaler`.

        The scaler is fitted *only* on the training data (`X_train`) to prevent
        data leakage from the validation or test sets. The fitted scaler is then
        used to transform `X_train`, `X_val`, and `X_test`.

        Args:
            X_train (pd.DataFrame): Training features DataFrame.
            X_val (pd.DataFrame): Validation features DataFrame.
            X_test (pd.DataFrame): Test features DataFrame.

        Returns:
            tuple: A tuple containing:
                - `X_train_scaled_df (pd.DataFrame)`: Scaled training features.
                - `X_val_scaled_df (pd.DataFrame)`: Scaled validation features.
                - `X_test_scaled_df (pd.DataFrame)`: Scaled test features.
                - `scaler (sklearn.preprocessing.MinMaxScaler)`: The fitted scaler object.
        """
        scaler = MinMaxScaler()
        # Fit the scaler on the training data and transform it
        X_train_scaled_np = scaler.fit_transform(X_train)
        # Transform the validation and test data using the fitted scaler
        X_val_scaled_np = scaler.transform(X_val)
        X_test_scaled_np = scaler.transform(X_test)

        # Convert scaled NumPy arrays back to DataFrames, preserving columns and index
        X_train_scaled_df = pd.DataFrame(X_train_scaled_np, columns=X_train.columns, index=X_train.index)
        X_val_scaled_df = pd.DataFrame(X_val_scaled_np, columns=X_val.columns, index=X_val.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled_np, columns=X_test.columns, index=X_test.index)

        return X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, scaler

    def create_sequences(self, features_df: pd.DataFrame, target_series: pd.Series, sequence_length: int):
        """
        Transforms time-series feature and target data into sequences suitable for
        models like LSTMs.

        For each point in time `i` (from `sequence_length - 1` onwards), a sequence
        of features from `i - sequence_length + 1` to `i` is created. The corresponding
        target is the target value at time `i`.

        Args:
            features_df (pd.DataFrame): DataFrame of (typically scaled) features.
            target_series (pd.Series): Series of target variable values.
            sequence_length (int): The number of time steps to include in each
                                   feature sequence.

        Returns:
            tuple: A tuple containing:
                - `X_sequences (np.ndarray)`: A 3D NumPy array of feature sequences,
                  where the shape is (number_of_sequences, sequence_length, number_of_features).
                - `y_sequences (np.ndarray)`: A 1D NumPy array of corresponding labels
                  (target values).
        """
        feature_data_np = features_df.to_numpy()
        target_data_np = target_series.to_numpy()

        X_sequences, y_sequences = [], []
        # Iterate from the first point where a full sequence can be formed
        for i in range(sequence_length - 1, len(feature_data_np)):
            # Extract the sequence of features: from (i - sequence_length + 1) up to i (inclusive)
            sequence = feature_data_np[i - sequence_length + 1 : i + 1]
            # The label corresponds to the target at the end of this sequence
            label = target_data_np[i]
            X_sequences.append(sequence)
            y_sequences.append(label)

        return np.array(X_sequences), np.array(y_sequences)

class FinancialDataset(Dataset):
    """
    PyTorch `Dataset` for handling financial time-series sequences.

    This class takes NumPy arrays of feature sequences and labels, converts them
    to PyTorch tensors, and makes them accessible by index, as required by
    PyTorch's `DataLoader`.
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initializes the FinancialDataset.

        Args:
            features (np.ndarray): A 3D NumPy array of input feature sequences
                                   (num_samples, sequence_length, num_features).
            labels (np.ndarray): A 1D NumPy array of corresponding labels.
        """
        # Convert NumPy arrays to PyTorch tensors
        # Features are expected to be float for model input
        self.features = torch.tensor(features, dtype=torch.float32)
        # Labels are also float for BCELoss (binary classification target)
        # Reshape labels to (num_samples, 1) for compatibility with loss functions like BCELoss
        self.labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        """
        Returns the total number of samples (sequences) in the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample (feature sequence and its label) from the dataset
        at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the feature sequence
                                               tensor and the label tensor.
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
        'RSI': np.random.rand(200) * 100 # RSI is not explicitly used in feature_engineer currently
    }
    dummy_df = pd.DataFrame(data, index=dates)

    print("--- Testing MLDataPreprocessor ---")
    preprocessor = MLDataPreprocessor()

    # 1. Feature Engineering
    print("\nStep 1: Feature Engineering...")
    featured_df = preprocessor.feature_engineer(dummy_df.copy())
    print("Featured DataFrame (head):")
    print(featured_df.head())
    print(f"Shape of featured_df: {featured_df.shape}")

    if not featured_df.empty and 'target' in featured_df.columns:
        y = featured_df['target']
        X = featured_df.drop(columns=['target'])

        # 2. Split Data
        print("\nStep 2: Splitting Data...")
        X_train_df, X_val_df, X_test_df, y_train_s, y_val_s, y_test_s = preprocessor.split_data(X, y)
        print(f"Shapes: X_train_df={X_train_df.shape}, X_val_df={X_val_df.shape}, X_test_df={X_test_df.shape}")
        print(f"Shapes: y_train_s={y_train_s.shape}, y_val_s={y_val_s.shape}, y_test_s={y_test_s.shape}")

        # 3. Scale Data
        print("\nStep 3: Scaling Data...")
        X_train_scaled, X_val_scaled, X_test_scaled, fitted_scaler = preprocessor.scale_data(X_train_df, X_val_df, X_test_df)
        print("Scaled X_train_scaled (head):")
        print(X_train_scaled.head())
        # print(f"Fitted Scaler: {fitted_scaler}") # Can be verbose

        # 4. Create Sequences
        print("\nStep 4: Creating Sequences...")
        sequence_length = 10
        X_train_seq, y_train_seq = preprocessor.create_sequences(X_train_scaled, y_train_s, sequence_length)
        X_val_seq, y_val_seq = preprocessor.create_sequences(X_val_scaled, y_val_s, sequence_length)
        X_test_seq, y_test_seq = preprocessor.create_sequences(X_test_scaled, y_test_s, sequence_length)

        if X_train_seq.size > 0:
            print(f"Shapes after sequence creation (Train): X={X_train_seq.shape}, y={y_train_seq.shape}")
            print(f"Shapes after sequence creation (Val): X={X_val_seq.shape}, y={y_val_seq.shape}")
            print(f"Shapes after sequence creation (Test): X={X_test_seq.shape}, y={y_test_seq.shape}")

            # 5. FinancialDataset and DataLoader
            print("\nStep 5: Creating FinancialDataset and DataLoader...")
            train_dataset = FinancialDataset(X_train_seq, y_train_seq)
            val_dataset = FinancialDataset(X_val_seq, y_val_seq)
            # test_dataset = FinancialDataset(X_test_seq, y_test_seq) # Example for test

            if len(train_dataset) > 0:
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Shuffle for training

                print(f"FinancialDataset for training created with {len(train_dataset)} samples.")
                print(f"FinancialDataset for validation created with {len(val_dataset)} samples.")

                # Print a sample batch shape from DataLoader
                for batch_features, batch_labels in train_loader:
                    print("\nSample batch from DataLoader:")
                    print(f"Batch features shape: {batch_features.shape}") # Expected: (batch_size, seq_len, num_features)
                    print(f"Batch labels shape: {batch_labels.shape}")     # Expected: (batch_size, 1)
                    break # Only need one sample batch
            else:
                print("Train dataset is empty. Cannot create DataLoader.")
        else:
            print("Sequence creation resulted in empty arrays. This might be due to insufficient data "
                  "for the chosen sequence_length after initial processing and splitting.")
    else:
        print("Featured DataFrame is empty or 'target' column missing. Aborting further preprocessing steps.")
    print("\n--- MLDataPreprocessor Test Complete ---")

[end of src/ml/data_preprocessor.py]
