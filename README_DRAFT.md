# Trading Analysis Platform

## Introduction

The Trading Analysis Platform is a Streamlit application designed for financial market analysis. It provides tools for data loading, interactive charting, applying technical indicators (both built-in and custom), and integrating machine learning models to generate trading signals. This platform aims to offer a flexible and user-friendly environment for traders and analysts to explore market data and test strategies.

## Prerequisites

Before you begin, ensure you have the following installed:

-   **Python**: Version 3.9 or higher is recommended. You can verify specific library compatibility by checking the versions listed in `requirements.txt`.
-   **pip**: Python's package installer, used to install project dependencies. It usually comes with Python installations.

## Installation

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
    (Replace `<repository_url>` and `<repository_directory>` with the actual URL and desired local directory name).

2.  **Create a Virtual Environment** (Recommended):
    This helps manage project dependencies separately.
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    -   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    -   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install Dependencies**:
    Install all required packages using pip and the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

To run the Trading Analysis Platform:

1.  Ensure your virtual environment is activated (if you created one).
2.  Navigate to the project's root directory in your terminal.
3.  Execute the following command:
    ```bash
    streamlit run src/app.py
    ```
4.  Streamlit will typically open the application automatically in your default web browser. If not, it will display a local URL (e.g., `http://localhost:8501`) that you can open manually.

## Features Overview

-   **Data Loading**:
    -   Specify stock tickers (e.g., AAPL, SPY) and historical data periods.
    -   Data is fetched from `yfinance`.
-   **Interactive Candlestick Charts**:
    -   Displays historical price data (Open, High, Low, Close) and trading volume.
    -   Users can pan, zoom, and inspect data points.
-   **Built-in RSI Indicator**:
    -   Relative Strength Index (RSI) can be added to the chart.
    -   The period for RSI calculation is adjustable by the user.
-   **Custom Technical Indicators**:
    -   Indicators are dynamically loaded from Python files placed in the `src/indicators/` directory.
    -   Each custom indicator can have user-configurable parameters via the Streamlit UI.
    -   Selected custom indicators are plotted on the main price chart or as separate subplots.
-   **Machine Learning Integration**:
    -   **Trading Signals**: Load a pre-trained LSTM model to generate buy/sell/hold trading signals based on historical data.
    -   **Model Training**: Option to trigger the training of a new LSTM model. Currently, this is configured for the SPY ticker using the script `src/ml/train.py`. (Note: Training can be time-consuming).
-   **Interactive User Interface**:
    -   Built with Streamlit, providing a clean and responsive web interface.
    -   A sidebar is used for most controls, including data input, indicator selection, and ML model parameters.

## Project Structure

A brief overview of the main directories:

-   `src/`: Contains the main application source code.
    -   `app.py`: The main Streamlit application entry point.
    -   `core/`: Core logic for data management, indicator calculation, and custom indicator interfacing.
    -   `charts/`: Modules related to generating and displaying charts (e.g., Plotly).
    -   `indicators/`: Directory for custom technical indicator plugins. Each `.py` file here can define a new indicator.
    -   `ml/`: Machine learning components.
        -   `models/`: Contains ML model definitions (e.g., LSTM).
        -   `train.py`: Script for training ML models.
        -   `predict.py`: Script/module for generating predictions/signals.
-   `data/`:
    -   `cache/`: Used for caching downloaded financial data to speed up loading times.
    -   Potentially stores trained ML models or other data assets.
-   `tests/`: Contains unit tests for various parts of the application.
-   `requirements.txt`: Lists all Python package dependencies.
-   `README.md`: This file, providing an overview of the project.

This structure helps organize the codebase and separates concerns for easier maintenance and development.
