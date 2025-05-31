# FinRL-based Trading Analysis Platform

This project aims to build a specialized trading analysis tool, conceptually similar to TradingView, with a core focus on leveraging the FinRL library. Key functionalities will include advanced chart analysis for the SPY ticker, the ability to fine-tune open-source Machine Learning (ML) models, plotting high/low prices, calculating and displaying the Relative Strength Index (RSI), and providing a framework for users to define and visualize custom technical indicators.

## Project Goals
*   Develop a platform for visualizing SPY ticker data, including candlestick charts and high/low price overlays.
*   Implement the standard Relative Strength Index (RSI) and design a flexible framework for custom indicators.
*   Integrate capabilities to acquire SPY data, prepare it for, and use it to tune and analyze outputs from open-source ML models.
*   Leverage FinRL as the core library for financial data handling.

## Current Status
Phase 1 (Core Setup and Data Acquisition) is in progress/complete.

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd trading_analysis_platform
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application (example - current main entry point for testing data manager):**
    To test the data fetching functionality:
    ```bash
    python src/core/data_manager.py
    ```
    This will download SPY data (if not cached or cache is stale) and print the head of the DataFrame.
    The main application entry point (`src/app.py`) will be developed in later phases.

## Project Structure
```
trading_analysis_platform/
├── data/                 # Data files (cache, processed)
├── notebooks/            # Jupyter notebooks for experimentation
├── src/                  # Source code
│   ├── app.py            # Main application entry point
│   ├── core/             # Core modules (data, indicators)
│   ├── charts/           # Charting logic
│   ├── ml/               # Machine Learning modules
│   ├── indicators/       # Custom indicator plugins
│   └── config/           # Configuration files (settings.yaml)
├── tests/                # Test scripts
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Configuration
Application settings, such as data sources, API keys (though none are used yet), and default parameters, are managed in `src/config/settings.yaml`.

## Next Steps
*   Implementation of basic charting (candlesticks, RSI).
*   Development of the custom indicator framework.
*   Integration of ML model fine-tuning and prediction.

## Contributing
Details for contributing to this project will be updated later.
