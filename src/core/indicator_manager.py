"""
Custom Indicator Management Module.

This module is responsible for discovering and dynamically loading custom technical
indicator classes. It scans a specified directory (typically `src/indicators/`)
for Python files that define classes inheriting from the
`src.core.custom_indicator_interface.CustomIndicator` base class.

The primary function, `discover_indicators`, provides the mechanism to make these
custom indicators available to the main application, allowing for an extensible
set of analytical tools.
"""
import os
import importlib.util
import inspect
import sys
from src.core.custom_indicator_interface import CustomIndicator

def discover_indicators(indicators_dir: str = "src/indicators") -> dict[str, type[CustomIndicator]]:
    """
    Scans the specified directory for Python files, attempts to import them,
    and discovers classes that are subclasses of `CustomIndicator`.

    The function iterates over each `.py` file in the `indicators_dir` (excluding
    files starting with `__`), imports it as a module, and then inspects the
    module's members. Any class found that inherits from `CustomIndicator` (but
    is not `CustomIndicator` itself) is added to the returned dictionary.

    Args:
        indicators_dir (str, optional): The directory path to scan for custom
                                        indicator Python files.
                                        Defaults to "src/indicators".

    Returns:
        dict[str, type[CustomIndicator]]: A dictionary where keys are the string
                                          names of the discovered indicator classes,
                                          and values are the corresponding class
                                          objects themselves. If the directory does
                                          not exist or no indicators are found,
                                          an empty dictionary is returned.

    Potential Issues and Limitations:
        - Import Errors: If an indicator file contains syntax errors, missing
          dependencies, or other import-related issues, an error message will be
          printed to `sys.stderr`, and that specific file will be skipped. The
          discovery process will continue with other files.
        - Naming Conflicts: If multiple indicator files define classes with the
          same name, the class from the module that is processed last will
          overwrite any previous ones in the returned dictionary. Unique class
          names across all indicator files are recommended.
        - Relative Imports: The function temporarily adds the imported module to
          `sys.modules` to support basic relative imports within the indicator
          modules. However, this might not be sufficient for complex package
          structures within individual indicator files. Properly packaging
          indicators or ensuring they are self-contained is advisable.
        - Correctness of Indicators: This function only discovers classes based on
          inheritance. It does not validate the correctness or completeness of the
          `CustomIndicator` implementation (e.g., whether `calculate` method is
          properly defined, though the ABC structure should enforce this).
        - Directory Existence: If `indicators_dir` does not exist, an empty
          dictionary is returned, and a message might not be explicitly printed
          by this function (though the calling code might handle it).
    """
    discovered_indicators = {}
    if not os.path.isdir(indicators_dir):
        # Optionally, log this more formally if a logging system is in place.
        # print(f"Indicator directory '{indicators_dir}' not found or is not a directory.", file=sys.stderr)
        return discovered_indicators

    for filename in os.listdir(indicators_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]
            file_path = os.path.join(indicators_dir, filename)

            # Store original sys.modules keys to detect additions by exec_module
            original_sys_modules_keys = set(sys.modules.keys())

            try:
                # Dynamically import the module from the file path
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    # Crucial for allowing the imported module to find itself (e.g. for relative imports)
                    # and for inspect.getmembers to work correctly with the module.
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                else:
                    # print(f"Could not create module spec for {module_name} from {file_path}", file=sys.stderr)
                    continue # Skip this file

                # Inspect the module for classes that are subclasses of CustomIndicator
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and \
                       issubclass(obj, CustomIndicator) and \
                       obj is not CustomIndicator: # Ensure it's not CustomIndicator itself
                        # print(f"Discovered indicator class: {name} in module: {module_name}")
                        if name in discovered_indicators:
                            print(f"Warning: Duplicate indicator class name '{name}'. "
                                  f"Replacing existing entry from a different file.", file=sys.stderr)
                        discovered_indicators[name] = obj

            except ImportError as e:
                print(f"Error importing module {module_name} from {file_path}: {e}", file=sys.stderr)
            except Exception as e:
                # Catch any other exceptions during module loading or inspection
                print(f"An unexpected error occurred while processing {file_path}: {e}", file=sys.stderr)
            finally:
                # Clean up: remove module from sys.modules if it was added by this function
                # This is important to avoid polluting sys.modules and to allow for potential reloads
                # if the indicator files change during an application's lifetime (though full reload
                # capabilities are more complex than this).
                if module_name in sys.modules and module_name not in original_sys_modules_keys:
                    del sys.modules[module_name]
    return discovered_indicators

if __name__ == '__main__':
    # --- Test Setup ---
    # Create a temporary directory for test indicators
    INDICATORS_TEST_DIR = "temp_test_indicators_manager" # Unique name for this test
    # Define path for a dummy CustomIndicator interface, relative to this script's location
    # This helps make the test runnable even if the main project structure isn't fully in sys.path
    # when running this file directly.
    DUMMY_INTERFACE_CONTENT = """
from abc import ABC, abstractmethod
import pandas as pd

class CustomIndicator(ABC):
    def __init__(self, **kwargs):
        # Simple kwargs to attributes for testing
        for key, value in kwargs.items():
            setattr(self, key, value)
    @abstractmethod
    def calculate(self, data_df: pd.DataFrame) -> pd.Series | pd.DataFrame:
        pass
"""
    # Path to the actual interface, assuming this script is in src/core
    actual_interface_module_path_for_import = "src.core.custom_indicator_interface"

    # Create the test directory
    if not os.path.exists(INDICATORS_TEST_DIR):
        os.makedirs(INDICATORS_TEST_DIR)

    # Create a dummy interface file *inside* the test directory for reliable import during test
    dummy_interface_file_path = os.path.join(INDICATORS_TEST_DIR, "custom_indicator_interface_dummy.py")
    with open(dummy_interface_file_path, "w") as f:
        f.write(DUMMY_INTERFACE_CONTENT)

    # Add the test directory to sys.path to allow importing the dummy interface
    sys.path.insert(0, INDICATORS_TEST_DIR)
    from custom_indicator_interface_dummy import CustomIndicator as DummyCustomIndicatorForTest

    # --- Create Dummy Indicator Files ---
    with open(os.path.join(INDICATORS_TEST_DIR, "sma_test.py"), "w") as f:
        f.write(f"""
from custom_indicator_interface_dummy import CustomIndicator as DummyCustomIndicatorForTest
import pandas as pd

class SimpleMovingAverageTest(DummyCustomIndicatorForTest):
    def __init__(self, period: int = 20, column: str = 'Close'):
        super().__init__(period=period, column=column)
        self.period = period
        self.column = column

    def calculate(self, data_df: pd.DataFrame) -> pd.Series:
        if self.column not in data_df: return pd.Series(dtype='float64')
        return data_df[self.column].rolling(window=self.period).mean()

class AnotherIndicatorTest(DummyCustomIndicatorForTest):
    def __init__(self, length: int = 10, source: str = 'Open', factor: float = 2.0, enabled: bool = True):
        super().__init__(length=length, source=source, factor=factor, enabled=enabled)
        self.length = length
        self.source = source
        self.factor = factor
        self.enabled = enabled

    def calculate(self, data_df: pd.DataFrame) -> pd.Series:
        if self.source not in data_df: return pd.Series(dtype='float64')
        return data_df[self.source].rolling(window=self.length).std() * self.factor
""")

    with open(os.path.join(INDICATORS_TEST_DIR, "non_indicator_test.py"), "w") as f:
        f.write("class NotAnIndicatorTest:\\n    pass\\n")

    with open(os.path.join(INDICATORS_TEST_DIR, "error_indicator_test.py"), "w") as f:
        f.write("class ErrorIndicatorTest(DummyCustomIndicatorForTest):\\n    1/0\\n") # Syntax error effectively

    print(f"--- Testing Indicator Discovery in '{INDICATORS_TEST_DIR}' ---")
    # Temporarily replace CustomIndicator with our dummy for the test scope
    # This is a bit of a hack for standalone testing. In an integrated test suite,
    # you'd ensure the real CustomIndicator is used.
    # For this script, we rely on the fact that `discover_indicators` takes `CustomIndicator`
    # from its global scope, which we can't easily patch without more complex mocking.
    # The test files `sma_test.py` etc., are written to import `DummyCustomIndicatorForTest`
    # which is a stand-in for `CustomIndicator`. The test will validate if `discover_indicators`
    # correctly identifies subclasses of this dummy.
    # To make `discover_indicators` use the dummy for its `issubclass` check, we'd have to
    # modify its source or pass `DummyCustomIndicatorForTest` as an argument.
    # Instead, the test files use `DummyCustomIndicatorForTest` and we check if they are found.
    # The `CustomIndicator` import at the top of this file will be the real one if project is structured.
    # For this test to be truly isolated, `discover_indicators` would need to accept the base class.

    # For the sake of this standalone test, we will make `discover_indicators`
    # use the `DummyCustomIndicatorForTest` by monkeypatching the global CustomIndicator
    # within this test block.
    original_custom_indicator_ref = CustomIndicator # Save the original
    # HACK: Temporarily replace the global CustomIndicator with our dummy for testing discovery
    # This is generally not good practice but useful for a self-contained test script.
    globals()['CustomIndicator'] = DummyCustomIndicatorForTest

    discovered = discover_indicators(INDICATORS_TEST_DIR)
    print(f"Discovered indicators: {{k: v.__name__ for k, v in discovered.items()}}")

    assert "SimpleMovingAverageTest" in discovered, "SMA Test not found"
    assert issubclass(discovered["SimpleMovingAverageTest"], DummyCustomIndicatorForTest)
    assert "AnotherIndicatorTest" in discovered, "Another Indicator Test not found"
    assert issubclass(discovered["AnotherIndicatorTest"], DummyCustomIndicatorForTest)
    assert "NotAnIndicatorTest" not in discovered, "Non-indicator class should not be discovered"
    assert "ErrorIndicatorTest" not in discovered, "Indicator with error should not be loaded"
    assert len(discovered) == 2, f"Expected 2 indicators, found {len(discovered)}"

    print("\\n--- Inspecting Parameters for AnotherIndicatorTest ---")
    if "AnotherIndicatorTest" in discovered:
        sig = inspect.signature(discovered["AnotherIndicatorTest"].__init__)
        params = sig.parameters
        for name, param in params.items():
            if name == 'self' or name == 'kwargs': continue # kwargs added by dummy
            print(f"  Param: '{name}', Type: {param.annotation}, Default: {param.default}")
        assert 'length' in params and params['length'].annotation == int
        assert 'source' in params and params['source'].annotation == str

    print("\n--- Indicator Discovery Test Complete ---")

    # Restore original CustomIndicator
    globals()['CustomIndicator'] = original_custom_indicator_ref

    # --- Cleanup ---
    try:
        os.remove(dummy_interface_file_path)
        os.remove(os.path.join(INDICATORS_TEST_DIR, "sma_test.py"))
        os.remove(os.path.join(INDICATORS_TEST_DIR, "non_indicator_test.py"))
        os.remove(os.path.join(INDICATORS_TEST_DIR, "error_indicator_test.py"))

        # Clean up __pycache__ if it exists
        pycache_dir = os.path.join(INDICATORS_TEST_DIR, "__pycache__")
        if os.path.exists(pycache_dir):
            for f_cache in os.listdir(pycache_dir):
                 os.remove(os.path.join(pycache_dir, f_cache))
            os.rmdir(pycache_dir)

        os.rmdir(INDICATORS_TEST_DIR)
        print("Cleaned up dummy files and directory.")
    except OSError as e:
        print(f"Error during cleanup: {e}", file=sys.stderr)
    finally:
        # Remove the test directory from sys.path
        if INDICATORS_TEST_DIR in sys.path[0]: # If it was the first one added
            sys.path.pop(0)
        elif INDICATORS_TEST_DIR in sys.path: # If it was added elsewhere (less likely with insert(0))
             sys.path.remove(INDICATORS_TEST_DIR)
