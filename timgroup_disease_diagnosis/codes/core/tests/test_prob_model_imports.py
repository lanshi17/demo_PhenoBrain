import importlib.util
from pathlib import Path


NB_MODEL_PATH = Path(__file__).resolve().parents[1] / 'core' / 'predict' / 'prob_model' / 'nb_model.py'


def test_nb_model_imports_under_modern_sklearn():
    spec = importlib.util.spec_from_file_location('legacy_nb_model', str(NB_MODEL_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, 'HPOProbMNBModel')
