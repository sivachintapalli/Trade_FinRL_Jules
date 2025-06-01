
 Trading Analysis Platform

## Introduction

The Trading Analysis Platform is a Streamlit-based web application designed for interactive financial analysis. It empowers users to load historical stock data, visualize price movements with candlestick charts, apply various technical indicators (both built-in and custom-developed), and leverage machine learning models to generate trading signals.

## Prerequisites

Before you begin, ensure you have the following installed:

*   **Python:** Version 3.9 or newer is recommended. You can verify your Python version by running `python --version`. Compatibility with specific libraries can be checked against the versions listed in `requirements.txt`.
*   **pip:** Python's package installer, which usually comes with Python. You can verify by running `pip --version`.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd <repository_directory_name>
    ```

2.  **Create a Virtual Environment (Recommended):**
    This isolates project dependencies.
    ```bash
    python -m venv venv
    ```
    Activate the environment:
    *   On Windows: `.\venv\Scripts\activate`
    *   On macOS/Linux: `source venv/bin/activate`

3.  **Install Dependencies:**
    Install all required packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

Once the installation is complete, you can run the Trading Analysis Platform using Streamlit:

```bash
streamlit run src/app.py
```

This command will typically open the application automatically in your default web browser. If not, your console will display a local URL (e.g., `http://localhost:8501`) that you can navigate to.

## Features Overview

*   **Data Loading:**
    *   Fetch historical stock data for any ticker symbol supported by Yahoo Finance.
    *   Select various time periods for analysis (e.g., "1mo", "1y", "max").
    *   Utilizes caching for faster subsequent loads.
*   **Interactive Candlestick Charts:**
    *   Displays OHLC (Open, High, Low, Close) price data.
    *   Includes trading volume visualization.
*   **Built-in RSI Indicator:**
    *   Calculate and plot the Relative Strength Index (RSI).
    *   Adjustable period for RSI calculation directly from the UI.
*   **Custom Technical Indicators:**
    *   Dynamically discovers and loads custom indicator plugins from the `src/indicators/` directory.
    *   Supports user-configurable parameters for each custom indicator via the UI.
    *   Calculated indicator values are plotted on the main chart or as separate subplots if necessary.
*   **Machine Learning Integration:**
    *   Load a pre-trained LSTM (Long Short-Term Memory) model to generate buy/sell trading signals based on price action. The default model is `models/lstm_model_v1.pth`.
    *   Option to trigger the training of a new ML model directly from the UI. The current training script (`src/ml/train.py`) is configured for SPY ticker data by default.
*   **Interactive User Interface:**
    *   Built with Streamlit for a responsive and user-friendly experience.
    *   A sidebar provides easy access to controls for data loading, indicator configuration, and ML operations.

## Project Structure

A brief overview of key directories within the project:

*   `src/`: Contains the main source code for the application.
    *   `app.py`: The main Streamlit application script.
    *   `core/`: Core logic for data management, indicator calculation, and custom indicator discovery.
    *   `charts/`: Modules related to plot generation (e.g., candlestick charts).
    *   `indicators/`: Directory for custom technical indicator plugin modules. New indicators placed here are auto-discovered.
    *   `ml/`: Machine learning components, including model definitions (`models/`), training scripts (`train.py`), and prediction logic (`predict.py`).
*   `data/`:
    *   `cache/`: Stores cached financial data to speed up loading times.
*   `models/`: Default location for trained machine learning model files (e.g., `.pth` files).
*   `tests/`: Contains unit tests for various parts of the application.
*   `requirements.txt`: Lists all Python dependencies for the project.
*   `README.md`: This file – providing an overview and instructions for the project.

This structure helps in organizing the codebase and separating concerns, making it easier to navigate and maintain.
