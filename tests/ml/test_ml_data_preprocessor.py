import unittest
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

# Adjust import path based on where the test is run from.
# Assuming 'tests' is a top-level directory and Python path includes the project root 'src'.
try:
    from src.ml.data_preprocessor import MLDataPreprocessor, FinancialDataset
except ModuleNotFoundError: # Fallback if run from 'tests' directory or specific CI setups
    import sys
    sys.path.append('../..') # Add project root to path
    from src.ml.data_preprocessor import MLDataPreprocessor, FinancialDataset


class TestMLDataPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = MLDataPreprocessor()
        # Create a small, consistent dummy raw_df for use in multiple tests
        data = {
            'Open': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            'High': [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5],
            'Low': [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5],
            'Close': [10.2, 11.2, 12.2, 13.2, 14.2, 15.2, 16.2, 17.2, 18.2, 19.2, 20.2, 21.2, 22.2, 23.2, 24.2, 25.2, 26.2, 27.2, 28.2, 29.2],
            'Volume': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290]
        }
        self.raw_df = pd.DataFrame(data, index=pd.date_range(start='2023-01-01', periods=20, freq='B'))
        # Add RSI if feature_engineer uses it. For now, FE in data_preprocessor.py doesn't add RSI itself.
        # self.raw_df['RSI'] = np.random.rand(20) * 100

    def test_feature_engineer(self):
        featured_df = self.preprocessor.feature_engineer(self.raw_df.copy())

        self.assertIn('target', featured_df.columns)
        self.assertIn('close_lag_1', featured_df.columns)
        self.assertIn('volume_lag_5', featured_df.columns)
        self.assertIn('pct_change', featured_df.columns)

        self.assertFalse(featured_df.isnull().values.any(), "NaNs found after feature engineering")

        # Original df has 20 rows.
        # Lagging up to 5 periods means first 5 rows will have NaNs for those lags.
        # Target creation shifts 'Close' by -1, so last row's target is NaN.
        # dropna() will remove these. Max removed is 5 (for lags) + 1 (for target if it's the limiting factor).
        # Expected rows: 20 - 5 (max_lag) - 1 (target shift at end) = 14.
        # Let's verify the current implementation's dropna behavior.
        # Max lag is 5. Target is shift(-1). So, first 5 rows are removed due to lag. Last row due to target.
        # So, 20 - 5 - 1 = 14 rows.
        # However, the implementation's `dropna()` acts on all NaNs.
        # `close_lag_5` makes rows 0-4 NaN. `target` makes row 19 NaN.
        # So, rows 0,1,2,3,4 and 19 are dropped. Total 6 rows dropped. 20 - 6 = 14 rows.
        self.assertEqual(featured_df.shape[0], 20 - 5 - 1) # 5 for lags, 1 for target shift
        # Number of features: Original (5) + Lags (5 close + 5 volume = 10) + pct_change (1) + target (1) = 17
        self.assertEqual(featured_df.shape[1], 5 + (2 * 5) + 1 + 1)


    def test_split_data(self):
        # Create a dummy DataFrame with 100 rows
        data = {'feature1': np.arange(100), 'feature2': np.arange(100, 200)}
        X = pd.DataFrame(data, index=pd.date_range(start='2023-01-01', periods=100, freq='B'))
        y = pd.Series(np.random.randint(0, 2, 100), index=X.index)

        # test_size=0.2 -> 20 test samples. Remaining 80 for train+val
        # val_size=0.25 (of train+val) -> 0.25 * 80 = 20 val samples. Train will be 60.
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.preprocessor.split_data(X, y, test_size=0.2, val_size=0.25)

        self.assertEqual(X_train.shape[0], 60)
        self.assertEqual(y_train.shape[0], 60)
        self.assertEqual(X_val.shape[0], 20)
        self.assertEqual(y_val.shape[0], 20)
        self.assertEqual(X_test.shape[0], 20)
        self.assertEqual(y_test.shape[0], 20)

        # Assert chronological split (max index of train < min index of val, etc.)
        self.assertTrue(X_train.index.max() < X_val.index.min())
        self.assertTrue(X_val.index.max() < X_test.index.min())
        self.assertTrue(y_train.index.max() < y_val.index.min())
        self.assertTrue(y_val.index.max() < y_test.index.min())

    def test_scale_data(self):
        X_train = pd.DataFrame(np.random.rand(60, 3) * 100, columns=['a', 'b', 'c']) # Values from 0 to 100
        X_val = pd.DataFrame(np.random.rand(20, 3) * 100, columns=['a', 'b', 'c'])
        X_test = pd.DataFrame(np.random.rand(20, 3) * 100, columns=['a', 'b', 'c'])

        X_train_scaled, X_val_scaled, X_test_scaled = \
            self.preprocessor.scale_data(X_train.copy(), X_val.copy(), X_test.copy())

        self.assertIsInstance(X_train_scaled, pd.DataFrame)
        self.assertIsInstance(X_val_scaled, pd.DataFrame)
        self.assertIsInstance(X_test_scaled, pd.DataFrame)

        # Assert that all values in scaled DataFrames are between 0 and 1 (inclusive)
        self.assertTrue((X_train_scaled.values >= 0).all() and (X_train_scaled.values <= 1).all())
        # Val and Test might go slightly out of [0,1] if their original range exceeded train's range
        # For this test, we'll check they are broadly scaled. A strict [0,1] check is for train.
        # self.assertTrue((X_val_scaled.values >= 0).all() and (X_val_scaled.values <= 1).all())
        # self.assertTrue((X_test_scaled.values >= 0).all() and (X_test_scaled.values <= 1).all())
        self.assertEqual(X_train_scaled.shape, X_train.shape)
        self.assertEqual(X_val_scaled.shape, X_val.shape)
        self.assertEqual(X_test_scaled.shape, X_test.shape)

    def test_create_sequences(self):
        num_rows = 20
        num_features = 3
        sequence_length = 5
        features_df = pd.DataFrame(np.random.rand(num_rows, num_features), columns=[f'f{i}' for i in range(num_features)])
        target_series = pd.Series(np.random.randint(0, 2, num_rows))

        X_seq, y_seq = self.preprocessor.create_sequences(features_df, target_series, sequence_length)

        expected_num_sequences = num_rows - sequence_length + 1
        self.assertEqual(X_seq.shape, (expected_num_sequences, sequence_length, num_features))
        self.assertEqual(y_seq.shape, (expected_num_sequences,))

    def test_financial_dataset(self):
        num_samples = 50
        sequence_length = 10
        num_features = 5
        X_seq_np = np.random.rand(num_samples, sequence_length, num_features).astype(np.float32)
        y_seq_np = np.random.randint(0, 2, num_samples).astype(np.float32)

        dataset = FinancialDataset(X_seq_np, y_seq_np)

        self.assertEqual(len(dataset), num_samples)

        features, label = dataset[0]

        self.assertIsInstance(features, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)

        self.assertEqual(features.shape, (sequence_length, num_features))
        self.assertEqual(label.shape, (1,)) # Labels are reshaped to (-1, 1)

        self.assertEqual(features.dtype, torch.float32)
        self.assertEqual(label.dtype, torch.float32)

if __name__ == '__main__':
    unittest.main()
