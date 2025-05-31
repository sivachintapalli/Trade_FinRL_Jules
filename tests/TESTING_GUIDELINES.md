# Testing Guidelines for Trading Analysis Platform

## 1. Introduction

The purpose of this document is to provide guidance for developers and testers on how to effectively test the various components of the Trading Analysis Platform. Consistent and thorough testing is crucial for ensuring the quality, reliability, and correctness of financial calculations, data processing, machine learning predictions, and the overall user experience.

This document covers different types of testing relevant to this project, points to detailed test case documentation, and offers advice on setting up a testing environment and writing new tests.

## 2. Types of Testing

### Unit Tests

*   **Focus:** Unit tests are designed to verify the smallest pieces of testable software in the application, such as individual functions, methods within a class, or entire classes (modules). They ensure that each component works correctly in isolation.
*   **Framework:** This project uses `pytest` for writing and running automated unit tests. Existing unit tests can be found in the `tests/` directory, typically mirroring the structure of the `src/` directory.
*   **Running Tests:** To run the automated unit tests, navigate to the project's root directory in your terminal and execute:
    ```bash
    pytest
    ```
    or
    ```bash
    python -m pytest
    ```
*   **Test Cases:** While automated tests cover specific code paths, more comprehensive test *scenarios* (which can guide manual testing or inspire new automated tests) are documented in markdown files. These are listed in Section 3.

### Manual UI/Application Testing

*   **Focus:** This type of testing involves interacting with the Streamlit application through its user interface, simulating how an end-user would operate the platform. It's essential for verifying user workflows, UI responsiveness, and visual correctness of charts.
*   **Key Areas for Manual Testing:**
    *   **Data Loading:**
        *   Load data for various valid stock tickers (common and less common).
        *   Test different historical periods ("1mo", "1y", "max", etc.).
        *   Attempt to load data for invalid/non-existent tickers.
    *   **Built-in Indicators:**
        *   Add and remove built-in indicators (e.g., RSI).
        *   Change indicator parameters (e.g., RSI period) and verify chart updates and recalculations.
    *   **Custom Indicators:**
        *   Discover and add custom indicators from the `src/indicators/` directory.
        *   Test with different valid and invalid parameters for each custom indicator.
        *   Verify chart updates correctly when custom indicators are added or removed.
    *   **Machine Learning Signals:**
        *   Load a pre-trained ML model (if available).
        *   Generate and display ML trading signals on the price chart.
        *   Verify signals appear as expected (e.g., correct markers at correct locations).
    *   **UI Responsiveness & Interactivity:**
        *   Check for smooth interaction with widgets (sliders, dropdowns, buttons).
        *   Ensure the application layout adapts reasonably to different interactions.
    *   **Error Handling in UI:**
        *   Attempt actions in an incorrect order (e.g., adding an indicator before loading data).
        *   Input invalid values into parameter fields.
        *   Verify that user-friendly error messages or warnings are displayed.

### Integration Testing (Conceptual)

*   **Focus:** Integration testing aims to verify the interaction between different components or modules of the application. For example, ensuring that data fetched by the `DataManager` is correctly processed by an indicator in `IndicatorCalculator`, and then accurately plotted by the `PlotGenerator`.
*   **Approach:** Many of the detailed test cases documented in the markdown files (see Section 3) inherently test these integrations, even if executed manually or as part of broader unit tests. Formal, separate integration tests can be developed for critical workflows if needed.

## 3. Documented Test Cases

Detailed test scenarios for different modules have been documented in markdown files. These documents serve as a comprehensive guide for manual testing efforts and can also be used as a blueprint for writing new automated unit or integration tests.

*   **Core Components:**
    *   [Data Manager Test Cases](./core/test_data_manager_cases.md)
    *   [Indicator Calculator Test Cases](./core/test_indicator_calculator_cases.md)
*   **Indicators:**
    *   [Custom Indicator Test Cases](./indicators/test_custom_indicator_cases.md) (covers general custom indicator behavior and specific examples like `SimpleMovingAverage`)
*   **Charting:**
    *   [Plot Generator Test Cases](./charts/test_plot_generator_cases.md)
*   **Machine Learning:**
    *   [Machine Learning Components Test Cases](./ml/test_ml_components_cases.md) (covers data preprocessing, model structure, training, and prediction scripts)

## 4. Setting up a Testing Environment (General Advice)

*   **Dependencies:** Ensure all project dependencies are installed by running `pip install -r requirements.txt` (or equivalent if using other package managers like Conda).
*   **ML Models:** For testing ML model training or prediction features:
    *   Ensure any required pre-trained model files (`.pth`), scalers (`.joblib`), or other artifacts are available in the expected locations (typically `models/`).
    *   Have sample data (CSV files or access to dummy data generation) ready for training or prediction scripts.
*   **Cache Functionality:** When testing caching behavior (e.g., for `DataManager`), be aware of the cache directory (default: `data/cache/`). You might need to inspect or clear this directory's contents for certain test scenarios.
*   **Environment Variables:** If the application uses any environment variables, ensure they are set correctly in your testing environment.

## 5. Writing New Tests

A robust application relies on continuous addition of tests as new features are developed or bugs are fixed.

### Unit Tests (`pytest`)

*   **When to Write:** New unit tests should be written for any new core logic, calculation functions, data processing steps, or significant methods within classes.
*   **Location:** Place new test files in the appropriate subdirectory within `tests/` that mirrors the structure of `src/`. For example, a test for `src/core/new_module.py` would go into `tests/core/test_new_module.py`.
*   **Naming:** Follow existing naming conventions (e.g., test files prefixed with `test_`, test functions prefixed with `test_`).
*   **Best Practices:**
    *   Tests should be independent and isolated.
    *   Use clear assertions to check for expected outcomes.
    *   Mock external dependencies (like network calls to `yfinance`) where appropriate to make tests faster and more reliable.

### Defining New Test Cases (Markdown)

*   For significant new features or modules, consider creating a new `.md` file (similar to those listed in Section 3) to document detailed test scenarios.
*   If extending existing functionality, update the relevant existing test case markdown file with new scenarios.
*   This documentation helps structure manual testing and serves as a reference for future test automation.

## 6. Specific Testing Notes

### Custom Indicators

*   When a new custom indicator is added to the `src/indicators/` directory:
    *   Thoroughly test its initialization with valid and invalid parameters (refer to `test_custom_indicator_cases.md`).
    *   Verify its `calculate` method against known correct outputs for sample data.
    *   Test its integration into the Streamlit UI: ensure it's discoverable, parameters are displayed correctly, and the chart updates as expected.

### Machine Learning Models

*   **Training Pipeline:** Successfully running the `train.py` script with a dataset (even a small dummy one) is a key test of the training pipeline. Verify that model files, scalers, and other artifacts are saved correctly.
*   **Prediction Pipeline:** Using `predict.py` (or the `get_predictions` function) with a trained model and new data tests the prediction pipeline. Verify that input data is processed correctly and that signals are generated in the expected format.
*   **Output Validation:** Where possible, compare ML model outputs (e.g., prediction signals, classifications) against baseline expectations, manual analysis (on small datasets), or results from simpler models.
*   **Robustness:** Test how the ML components handle variations in input data (e.g., slightly different feature ranges if not scaled, missing recent data).

### Data Integrity

*   Financial data can be noisy and sometimes contain errors or missing values.
*   Always test how different components (data manager, indicators, ML preprocessor) handle:
    *   Missing data (`NaN` values) in input DataFrames.
    *   Unexpected values or data types (though type hinting and validation should catch many of these).
    *   Empty DataFrames or Series.

## 7. Reporting Bugs

Clear bug reports are essential for efficient debugging. If you encounter a bug:

1.  **Create an Issue:** If the project is hosted on a platform like GitHub, create a new issue in the repository's issue tracker.
2.  **Bug Report Details:** Include the following information:
    *   **Title:** A clear and concise summary of the bug.
    *   **Steps to Reproduce:** Detailed, step-by-step instructions on how to trigger the bug.
    *   **Expected Behavior:** What you expected to happen.
    *   **Actual Behavior:** What actually happened, including any error messages, stack traces, or incorrect outputs.
    *   **Environment (if relevant):** Operating system, Python version, versions of key libraries.
    *   **Screenshots/Logs:** Attach screenshots of the UI or relevant log output if they help illustrate the bug.
    *   **Severity/Priority (optional):** Your assessment of the bug's impact.

By following these guidelines, we can collectively contribute to a more stable, reliable, and accurate Trading Analysis Platform.
