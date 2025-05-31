import os
import importlib.util
import inspect
import sys
from src.core.custom_indicator_interface import CustomIndicator

def discover_indicators(indicators_dir: str = "src/indicators") -> dict[str, type[CustomIndicator]]:
    """
    Scans the given directory for Python files, imports them,
    and discovers classes that are subclasses of CustomIndicator.

    Args:
        indicators_dir: The directory path to scan for indicator Python files.

    Returns:
        A dictionary where keys are indicator class names (as strings)
        and values are the indicator classes themselves.
    """
    discovered_indicators = {}
    if not os.path.isdir(indicators_dir):
        # print(f"Indicator directory '{indicators_dir}' not found.", file=sys.stderr)
        return discovered_indicators

    for filename in os.listdir(indicators_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]
            file_path = os.path.join(indicators_dir, filename)

            try:
                # Import the module
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    # Add module to sys.modules temporarily if it's not there,
                    # to allow relative imports within the indicator module if any.
                    # More robust solutions might involve proper packaging of indicators.
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                else:
                    # print(f"Could not load spec for module {module_name} from {file_path}", file=sys.stderr)
                    continue

                # Inspect the module for CustomIndicator subclasses
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and \
                       issubclass(obj, CustomIndicator) and \
                       obj is not CustomIndicator:
                        # print(f"Discovered indicator: {name} in {module_name}")
                        discovered_indicators[name] = obj

            except ImportError as e:
                print(f"Error importing module {module_name} from {file_path}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"An unexpected error occurred while processing {file_path}: {e}", file=sys.stderr)
            finally:
                # Clean up sys.modules if module was added
                if module_name in sys.modules and sys.modules[module_name] is module:
                    del sys.modules[module_name]


    return discovered_indicators

if __name__ == '__main__':
    # Create dummy indicators for testing discovery
    INDICATORS_TEST_DIR = "temp_test_indicators"
    INTERFACE_FILE = "../core/custom_indicator_interface.py" # Adjust path as needed

    if not os.path.exists(INDICATORS_TEST_DIR):
        os.makedirs(INDICATORS_TEST_DIR)

    # Dummy CustomIndicator interface for testing if the real one is not found
    custom_indicator_content = """
from abc import ABC, abstractmethod
import pandas as pd

class CustomIndicator(ABC):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    @abstractmethod
    def calculate(self, data_df: pd.DataFrame) -> pd.Series | pd.DataFrame:
        pass
"""
    if not os.path.exists(INTERFACE_FILE):
        # Fallback, create a dummy interface file in the temp dir for the test to run
        # This is not ideal but helps make the test self-contained if run standalone
        # and paths are tricky.
        # A better approach for standalone testing would be to mock CustomIndicator.
        print(f"Warning: CustomIndicator interface not found at '{INTERFACE_FILE}'. Using dummy for test.", file=sys.stderr)
        INTERFACE_FILE = os.path.join(INDICATORS_TEST_DIR, "custom_indicator_interface.py")
        with open(INTERFACE_FILE, "w") as f:
            f.write(custom_indicator_content)
        # Need to adjust sys.path for the test to find this dummy interface
        sys.path.insert(0, INDICATORS_TEST_DIR)
        from custom_indicator_interface import CustomIndicator
    else:
        # Ensure src directory is in path for finding `src.core.custom_indicator_interface`
        # when running this script directly for testing.
        # This assumes the script is run from a directory where 'src' is a subdirectory
        # or 'src' is already in PYTHONPATH.
        # For example, if script is in `src/core/` and run from `src/core/`,
        # we need to go up two levels to add the project root.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from src.core.custom_indicator_interface import CustomIndicator


    # Create dummy indicator files
    with open(os.path.join(INDICATORS_TEST_DIR, "sma.py"), "w") as f:
        f.write(f\"\"\"
from {os.path.basename(INTERFACE_FILE)[:-3] if 'custom_indicator_interface' in INTERFACE_FILE else 'src.core.custom_indicator_interface'} import CustomIndicator
import pandas as pd

class SimpleMovingAverage(CustomIndicator):
    def __init__(self, period: int = 20, column: str = 'Close'):
        super().__init__(period=period, column=column)
        self.period = period
        self.column = column

    def calculate(self, data_df: pd.DataFrame) -> pd.Series:
        return data_df[self.column].rolling(window=self.period).mean()

class AnotherIndicator(CustomIndicator):
    def __init__(self, length: int = 10, source: str = 'Open', factor: float = 2.0, enabled: bool = True):
        super().__init__(length=length, source=source, factor=factor, enabled=enabled)
        # Actual params stored by super().__init__
        self.length = length
        self.source = source
        self.factor = factor
        self.enabled = enabled


    def calculate(self, data_df: pd.DataFrame) -> pd.Series:
        return data_df[self.source].rolling(window=self.length).std() * self.factor
\"\"\")

    with open(os.path.join(INDICATORS_TEST_DIR, "non_indicator.py"), "w") as f:
        f.write("class NotAnIndicator:\\n    pass\\n")

    with open(os.path.join(INDICATORS_TEST_DIR, "empty.py"), "w") as f:
        f.write("# This is an empty file\\n")

    print(f"Testing indicator discovery in '{INDICATORS_TEST_DIR}':")
    indicators = discover_indicators(INDICATORS_TEST_DIR)
    print(f"Discovered indicators: {indicators}")

    assert "SimpleMovingAverage" in indicators
    assert "AnotherIndicator" in indicators
    assert len(indicators) == 2

    # Test parameter inspection for one of the discovered indicators
    if "AnotherIndicator" in indicators:
        sig = inspect.signature(indicators["AnotherIndicator"].__init__)
        params = sig.parameters
        print("\\nParameters for AnotherIndicator:")
        for name, param in params.items():
            if name == 'self':
                continue
            print(f"  {name}: type={param.annotation}, default={param.default}")
        assert 'length' in params
        assert params['length'].annotation == int
        assert params['length'].default == 10
        assert 'factor' in params
        assert params['factor'].annotation == float

    print("\\nIndicator discovery test complete.")

    # Clean up dummy files and directory
    try:
        if 'custom_indicator_interface' in INTERFACE_FILE and INDICATORS_TEST_DIR in INTERFACE_FILE : # if dummy interface was created
             os.remove(INTERFACE_FILE)
        os.remove(os.path.join(INDICATORS_TEST_DIR, "sma.py"))
        os.remove(os.path.join(INDICATORS_TEST_DIR, "non_indicator.py"))
        os.remove(os.path.join(INDICATORS_TEST_DIR, "empty.py"))
        if os.path.exists(os.path.join(INDICATORS_TEST_DIR, "__pycache__")):
            for f_cache in os.listdir(os.path.join(INDICATORS_TEST_DIR, "__pycache__")):
                 os.remove(os.path.join(INDICATORS_TEST_DIR, "__pycache__", f_cache))
            os.rmdir(os.path.join(INDICATORS_TEST_DIR, "__pycache__"))
        os.rmdir(INDICATORS_TEST_DIR)
        print("Cleaned up dummy files and directory.")
    except OSError as e:
        print(f"Error during cleanup: {e}", file=sys.stderr)

    # Remove path added for dummy interface if necessary
    if INDICATORS_TEST_DIR in sys.path and INTERFACE_FILE.startswith(INDICATORS_TEST_DIR):
        sys.path.pop(0)
    if project_root in sys.path and project_root not in os.environ.get("PYTHONPATH", ""):
        sys.path.remove(project_root)
