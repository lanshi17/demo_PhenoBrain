import importlib.util
from pathlib import Path


UTILS_PATH = Path(__file__).resolve().parents[1] / 'core' / 'utils' / 'utils.py'


def test_utils_module_imports_under_python_312():
    spec = importlib.util.spec_from_file_location('legacy_utils_module', str(UTILS_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, 'equal')
