import os
import importlib.util
import inspect
import pandas as pd
from typing import Dict, Type, Any, Optional # Added Any and Optional
from src.core.custom_indicator_interface import CustomIndicator

# load_custom_indicators function remains the same as implemented in the previous step
def load_custom_indicators(indicators_path: str = "src/indicators") -> Dict[str, Type[CustomIndicator]]:
    """
    Dynamically loads custom indicator classes from Python files in the specified directory.

    Args:
        indicators_path: The path to the directory containing custom indicator Python files.

    Returns:
        A dictionary where keys are indicator class names and values are the classes themselves.
    """
    custom_indicators: Dict[str, Type[CustomIndicator]] = {}
    if not os.path.exists(indicators_path):
        # Instead of printing, consider logging or raising an exception for critical errors
        # For now, printing a warning is fine as per previous implementation.
        print(f"Warning: Indicators directory not found: {indicators_path}")
        return custom_indicators
    if not os.path.isdir(indicators_path):
        print(f"Warning: Indicators path is not a directory: {indicators_path}")
        return custom_indicators

    for filename in os.listdir(indicators_path):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]
            file_path = os.path.join(indicators_path, filename)

            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    # Add module to sys.modules to handle relative imports within indicator files if any
                    # sys.modules[module_name] = module
                    spec.loader.exec_module(module)

                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and                            issubclass(obj, CustomIndicator) and                            obj is not CustomIndicator:
                            if name in custom_indicators:
                                print(f"Warning: Duplicate indicator class name '{name}' found in {filename}. It will overwrite the previous one.")
                            custom_indicators[name] = obj
                            # print(f"Successfully loaded indicator: {name} from {filename}") # Optional: reduce verbosity
                else:
                    print(f"Warning: Could not load module specification from {file_path}")

            except ImportError as e:
                print(f"Error importing module {module_name} from {file_path}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred while loading indicator from {file_path}: {e}")

    return custom_indicators

def calculate_indicator(
    data_df: pd.DataFrame,
    indicator_name: str,
    indicators_map: Dict[str, Type[CustomIndicator]],
    **params: Any  # Using Any for kwargs type
) -> Optional[pd.Series | pd.DataFrame]: # Return type updated
    """
    Calculates a specific indicator using the loaded indicator map and provided parameters.

    Args:
        data_df: Pandas DataFrame with OHLCV data.
        indicator_name: The name of the indicator class to use.
        indicators_map: A dictionary mapping indicator names to their classes
                        (typically from load_custom_indicators).
        **params: Arbitrary keyword arguments to be passed to the indicator's constructor.

    Returns:
        A Pandas Series or DataFrame containing the calculated indicator values,
        or None if an error occurs.
    """
    if not isinstance(data_df, pd.DataFrame):
        print("Error: data_df must be a Pandas DataFrame.")
        return None

    if not data_df.index.is_unique:
        print("Warning: DataFrame index is not unique. This might cause issues with indicator calculations.")

    if indicator_name not in indicators_map:
        print(f"Error: Indicator '{indicator_name}' not found in loaded indicators map.")
        available_indicators = list(indicators_map.keys())
        if available_indicators:
            print(f"Available indicators are: {', '.join(available_indicators)}")
        else:
            print("No indicators are available in the map.")
        return None

    IndicatorClass = indicators_map[indicator_name]

    try:
        # Instantiate the indicator with parameters
        indicator_instance = IndicatorClass(**params)
        # Calculate the indicator
        result = indicator_instance.calculate(data_df.copy()) # Pass a copy to avoid modifying original df

        if not isinstance(result, (pd.Series, pd.DataFrame)):
            print(f"Error: Indicator '{indicator_name}' did not return a Pandas Series or DataFrame.")
            return None

        if not result.index.equals(data_df.index):
            print(f"Warning: Index of the result from '{indicator_name}' does not match the input DataFrame's index. Re-indexing result.")
            # Attempt to reindex. This might introduce NaNs if alignment is not perfect.
            # Or, decide if this should be a stricter error. For now, reindex.
            result = result.reindex(data_df.index)

        return result
    except TypeError as e:
        # Catch errors related to incorrect parameters passed to __init__ or calculate
        print(f"Error instantiating or calculating indicator '{indicator_name}' with params {params}. Possible Mismatch in parameters. Details: {e}")
        # You might want to inspect IndicatorClass.__init__ signature here for better error messages
        # For example, inspect.signature(IndicatorClass.__init__)
        return None
    except ValueError as e:
        # Catch errors raised by the indicator's calculate method (e.g., column not found)
        print(f"Error during calculation of indicator '{indicator_name}': {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while calculating indicator {indicator_name} with params {params}: {e}")
        return None

if __name__ == '__main__':
    # Create dummy data for testing calculate_indicator
    data = {
        'Open': [i for i in range(100, 125)], # Increased data points for better SMA
        'High': [i + 2 for i in range(100, 125)],
        'Low': [i - 2 for i in range(100, 125)],
        'Close': [i + 0.5 for i in range(100, 125)],
        'Volume': [i * 10 for i in range(100, 125)]
    }
    idx = pd.date_range(start='2023-01-01', periods=25, freq='D')
    sample_df = pd.DataFrame(data, index=idx)

    print("Loading custom indicators...")
    # Assuming src/indicators/dummy_sma.py exists from previous step
    # It should define MyDummySMA(period=10, column='Close')
    loaded_indicators = load_custom_indicators()

    if not loaded_indicators:
        print("No custom indicators were loaded. Ensure 'src/indicators/dummy_sma.py' exists and is valid.")
    else:
        print("\nAvailable custom indicators:")
        for name in loaded_indicators:
            print(f"- {name}")

        # Test Case 1: Calculate MyDummySMA with valid parameters
        print("\nCalculating MyDummySMA(period=7, column='Close')...")
        sma_values = calculate_indicator(sample_df, "MyDummySMA", loaded_indicators, period=7, column='Close')
        if sma_values is not None:
            print("MyDummySMA calculated values (first 10):")
            print(sma_values.head(10))
        else:
            print("Failed to calculate MyDummySMA.")

        # Test Case 2: Calculate MyDummySMA with a different column and period
        print("\nCalculating MyDummySMA(period=5, column='High')...")
        sma_high_values = calculate_indicator(sample_df, "MyDummySMA", loaded_indicators, period=5, column='High')
        if sma_high_values is not None:
            print("MyDummySMA (High, 5) calculated values (first 10):")
            print(sma_high_values.head(10))
        else:
            print("Failed to calculate MyDummySMA for High.")

        # Test Case 3: Calculate MyDummySMA with missing required parameters (should use defaults or error if no defaults)
        # MyDummySMA has defaults, so this should work.
        print("\nCalculating MyDummySMA with default parameters (period=10, column='Close')...")
        sma_default_values = calculate_indicator(sample_df, "MyDummySMA", loaded_indicators)
        if sma_default_values is not None:
            print("MyDummySMA (default) calculated values (first 15):")
            print(sma_default_values.head(15)) # print more to see effect of period=10
        else:
            print("Failed to calculate MyDummySMA with default params.")

        # Test Case 4: Error case - Indicator not found
        print("\nAttempting to calculate a non-existent indicator...")
        non_existent_values = calculate_indicator(sample_df, "NonExistentIndicator", loaded_indicators, period=5)
        if non_existent_values is None:
            print("Correctly handled non-existent indicator.")

        # Test Case 5: Error case - Parameter mismatch (e.g., wrong parameter name)
        print("\nAttempting to calculate MyDummySMA with incorrect parameter name (e.g., 'window' instead of 'period')...")
        wrong_param_values = calculate_indicator(sample_df, "MyDummySMA", loaded_indicators, window=5, column='Close')
        if wrong_param_values is None:
            print("Correctly handled parameter mismatch for MyDummySMA.")

        # Test Case 6: Error case - Calculation error (e.g., column not found in DataFrame)
        print("\nAttempting to calculate MyDummySMA with a column not in DataFrame...")
        col_not_found_values = calculate_indicator(sample_df, "MyDummySMA", loaded_indicators, period=5, column='NonExistentColumn')
        if col_not_found_values is None:
            print("Correctly handled calculation error (column not found) for MyDummySMA.")

        # Test Case 7: DataFrame is not a DataFrame
        print("\nAttempting to calculate with invalid data_df type...")
        invalid_df_values = calculate_indicator("not a dataframe", "MyDummySMA", loaded_indicators, period=5) # type: ignore
        if invalid_df_values is None:
            print("Correctly handled invalid data_df type.")

    print("\nScript execution finished.")
