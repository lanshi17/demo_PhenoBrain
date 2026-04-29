import importlib.util
import sys
from pathlib import Path

import pytest


CORE_ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = CORE_ROOT / 'core' / 'utils' / 'utils.py'
MODEL_TESTOR_PATH = CORE_ROOT / 'core' / 'predict' / 'model_testor.py'
LR_NEURON_MODEL_PATH = CORE_ROOT / 'core' / 'predict' / 'ml_model' / 'lr_neuron_model.py'


def test_utils_module_imports_under_python_312():
    spec = importlib.util.spec_from_file_location('legacy_utils_module', str(UTILS_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, 'equal')


def test_model_testor_imports_with_current_scipy():
    sys.path.insert(0, str(CORE_ROOT))
    spec = importlib.util.spec_from_file_location('legacy_model_testor_module', str(MODEL_TESTOR_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, 'ModelTestor')


def test_lr_neuron_model_imports_with_tensorflow_2():
    sys.path.insert(0, str(CORE_ROOT))
    spec = importlib.util.spec_from_file_location('legacy_lr_neuron_model_module', str(LR_NEURON_MODEL_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, 'LRNeuronModel')


def test_lr_neuron_l2_loss_works_without_tf_contrib():
    sys.path.insert(0, str(CORE_ROOT))
    spec = importlib.util.spec_from_file_location('legacy_lr_neuron_model_module_for_l2', str(LR_NEURON_MODEL_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tf = module.tf

    class Config:
        w_decay = 0.5

    graph = tf.Graph()
    with graph.as_default():
        tf.Variable([[1.0, 2.0]], dtype=tf.float32)
        loss = module.LRNeuronModel.get_l2_loss(object(), Config())
        init_op = tf.global_variables_initializer()
    with tf.Session(graph=graph) as session:
        session.run(init_op)
        assert session.run(loss) == pytest.approx(1.25)
