# Trade_FinRL_Jules

## Machine Learning Module

This project includes a Machine Learning (ML) module for time-series prediction on SPY ticker data. The current implementation uses an LSTM model for binary classification (predicting up/down price movement).

### Directory Structure

The ML code is primarily located in the `src/ml/` directory:
-   `data_preprocessor.py`: Handles data loading, feature engineering, scaling, and sequence creation.
-   `models/lstm_model.py`: Defines the LSTM model architecture.
-   `train.py`: Script for training the LSTM model.
-   `predict.py`: Script for making predictions using a trained model.
-   `ml_visualization.py`: Provides utilities to plot price data with ML signals.

Unit tests for the ML module are in `tests/ml/`.

### Prerequisites

Ensure you have Python 3.7+ installed. Dependencies can be installed using:
```bash
pip install -r requirements.txt
```

### Training the Model

To train the LSTM model, run the `train.py` script. It uses dummy data by default.
```bash
python src/ml/train.py --epochs 5 --batch_size 16 --sequence_length 20 --num_dummy_rows 200
```
Key arguments for `train.py`:
-   `--data_path`: Path to your SPY data CSV (not used if `--num_dummy_rows` is set for dummy data).
-   `--epochs`: Number of training epochs.
-   `--batch_size`: Batch size for training.
-   `--learning_rate`: Learning rate for the optimizer.
-   `--sequence_length`: Length of input sequences for the LSTM.
-   `--hidden_size`: Number of units in LSTM hidden layers.
-   `--num_layers`: Number of LSTM layers.
-   `--dropout`: Dropout probability for LSTM.
-   `--model_save_path`: Path to save the best trained model.
-   `--early_stopping_patience`: Number of epochs to wait for improvement before stopping.
-   `--num_dummy_rows`: If specified, generates dummy data with this many rows for training.

The trained model will be saved (by default to `models/lstm_model_v1.pth`).

### Making Predictions

To make predictions using a trained model, run `predict.py`. It also uses dummy data by default.
```bash
python src/ml/predict.py --model_path models/lstm_model_v1.pth --sequence_length 20 --num_dummy_rows_for_input_calc 100 --num_dummy_rows 100
```
Key arguments for `predict.py`:
-   `--model_path`: Path to the trained model file.
-   `--data_path`: Path to new data for prediction (not used if `--num_dummy_rows` is set).
-   `--sequence_length`, `--hidden_size`, `--num_layers`, `--dropout`: Should match the parameters of the loaded model.
-   `--num_dummy_rows_for_input_calc`: Rows for dummy data to calculate model input size.
-   `--num_dummy_rows`: If specified, generates dummy data for prediction.

The script will output sample predictions and accuracy if actual labels are available.

### Visualizing Predictions

The `src/ml/ml_visualization.py` script contains a function `plot_price_with_signals` that can be used to plot candlestick charts with buy/sell signals. It can be integrated into a larger application or used for standalone analysis. An example of its usage is in its `if __name__ == '__main__':` block, which saves an HTML plot.
```bash
python src/ml/ml_visualization.py
```
This will generate `price_with_signals_example.html`.

### Running Tests

To run the unit tests for the ML module:
```bash
python -m unittest discover tests
```