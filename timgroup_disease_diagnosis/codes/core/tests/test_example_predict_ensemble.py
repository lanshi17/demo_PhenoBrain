import importlib.util
import sys
import types
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / 'core' / 'script' / 'example_predict_ensemble.py'


def load_module():
    spec = importlib.util.spec_from_file_location('example_predict_ensemble', str(SCRIPT_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('x')


def install_fake_core_modules(monkeypatch):
    core_pkg = types.ModuleType('core')
    predict_pkg = types.ModuleType('core.predict')
    utils_pkg = types.ModuleType('core.utils')

    ensemble_mod = types.ModuleType('core.predict.ensemble')
    ml_mod = types.ModuleType('core.predict.ml_model')
    prob_mod = types.ModuleType('core.predict.prob_model')
    sim_mod = types.ModuleType('core.predict.sim_model')
    reader_mod = types.ModuleType('core.reader')
    constant_mod = types.ModuleType('core.utils.constant')

    class FakeReader:
        def __init__(self, keep_dnames=None, rm_no_use_hpo=False):
            self.keep_dnames = keep_dnames
            self.rm_no_use_hpo = rm_no_use_hpo

    class FakeOrderedMultiModel:
        def __init__(self, model_inits=None, hpo_reader=None, model_name=None, model_list=None, **kwargs):
            self.model_inits = model_inits
            self.hpo_reader = hpo_reader
            self.model_name = model_name
            self.name = model_name
            self.model_list = model_list
            self.kwargs = kwargs

        def query(self, hpo_list, topk):
            return [('unused', 0.0)]

    class FakeOrderStatisticMultiModel(FakeOrderedMultiModel):
        pass

    class FakeRandomModel:
        pass

    class FakeICTODQAcrossModel:
        pass

    class FakeHPOProbMNBModel:
        pass

    class FakeCNBModel:
        pass

    class FakeLRNeuronModel:
        pass

    class FakeMICAModel:
        def __init__(self, hpo_reader=None, model_name=None, **kwargs):
            self.hpo_reader = hpo_reader
            self.model_name = model_name
            self.kwargs = kwargs

    class FakeMICALinModel:
        def __init__(self, hpo_reader=None, model_name=None, **kwargs):
            self.hpo_reader = hpo_reader
            self.model_name = model_name
            self.kwargs = kwargs

    class FakeMICAJCModel:
        def __init__(self, hpo_reader=None, model_name=None, **kwargs):
            self.hpo_reader = hpo_reader
            self.model_name = model_name
            self.kwargs = kwargs

    class FakeMinICModel:
        def __init__(self, hpo_reader=None, model_name=None, **kwargs):
            self.hpo_reader = hpo_reader
            self.model_name = model_name
            self.kwargs = kwargs

    class FakeRBPModel:
        def __init__(self, hpo_reader=None, model_name=None, **kwargs):
            self.hpo_reader = hpo_reader
            self.model_name = model_name
            self.name = model_name
            self.kwargs = kwargs

    class FakeGDDPFisherModel:
        def __init__(self, hpo_reader=None, model_name=None, **kwargs):
            self.hpo_reader = hpo_reader
            self.model_name = model_name
            self.name = model_name
            self.kwargs = kwargs

    class FakeBOQAModel:
        def __init__(self, hpo_reader=None, model_name=None, **kwargs):
            self.hpo_reader = hpo_reader
            self.model_name = model_name
            self.name = model_name
            self.kwargs = kwargs

    ensemble_mod.OrderStatisticMultiModel = FakeOrderStatisticMultiModel
    ensemble_mod.OrderedMultiModel = FakeOrderedMultiModel
    ensemble_mod.RandomModel = FakeRandomModel
    ml_mod.LRNeuronModel = FakeLRNeuronModel
    prob_mod.BOQAModel = FakeBOQAModel
    prob_mod.CNBModel = FakeCNBModel
    prob_mod.HPOProbMNBModel = FakeHPOProbMNBModel
    sim_mod.GDDPFisherModel = FakeGDDPFisherModel
    sim_mod.ICTODQAcrossModel = FakeICTODQAcrossModel
    sim_mod.MICAModel = FakeMICAModel
    sim_mod.MICALinModel = FakeMICALinModel
    sim_mod.MICAJCModel = FakeMICAJCModel
    sim_mod.MinICModel = FakeMinICModel
    sim_mod.RBPModel = FakeRBPModel
    reader_mod.HPOIntegratedDatasetReader = FakeReader
    constant_mod.PHELIST_ANCESTOR = 'PHELIST_ANCESTOR'
    constant_mod.PHELIST_REDUCE = 'PHELIST_REDUCE'
    constant_mod.PREDICT_MODE = 'PREDICT_MODE'
    constant_mod.VEC_TYPE_0_1 = 'VEC_TYPE_0_1'

    core_pkg.predict = predict_pkg
    core_pkg.utils = utils_pkg
    core_pkg.reader = reader_mod
    predict_pkg.ensemble = ensemble_mod
    predict_pkg.ml_model = ml_mod
    predict_pkg.prob_model = prob_mod
    predict_pkg.sim_model = sim_mod
    utils_pkg.constant = constant_mod

    fake_modules = {
        'core': core_pkg,
        'core.predict': predict_pkg,
        'core.predict.ensemble': ensemble_mod,
        'core.predict.ml_model': ml_mod,
        'core.predict.prob_model': prob_mod,
        'core.predict.sim_model': sim_mod,
        'core.reader': reader_mod,
        'core.utils': utils_pkg,
        'core.utils.constant': constant_mod,
    }
    for name, fake_module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, fake_module)

    return {
        'FakeOrderedMultiModel': FakeOrderedMultiModel,
        'FakeOrderStatisticMultiModel': FakeOrderStatisticMultiModel,
        'FakeRandomModel': FakeRandomModel,
        'FakeICTODQAcrossModel': FakeICTODQAcrossModel,
        'FakeHPOProbMNBModel': FakeHPOProbMNBModel,
        'FakeCNBModel': FakeCNBModel,
        'FakeLRNeuronModel': FakeLRNeuronModel,
        'FakeMICAModel': FakeMICAModel,
        'FakeMICALinModel': FakeMICALinModel,
        'FakeMICAJCModel': FakeMICAJCModel,
        'FakeMinICModel': FakeMinICModel,
        'FakeRBPModel': FakeRBPModel,
        'FakeGDDPFisherModel': FakeGDDPFisherModel,
        'FakeBOQAModel': FakeBOQAModel,
    }


def test_example_script_exists_and_exports_builder():
    assert SCRIPT_PATH.exists()

    module = load_module()

    assert hasattr(module, 'build_ensemble_model')
    assert hasattr(module, 'predict_ensemble')


def test_predict_ensemble_delegates_to_built_model(monkeypatch):
    module = load_module()

    class DummyModel:
        def query(self, hpo_list, topk):
            assert hpo_list == ['HP:0000118']
            assert topk == 1
            return [('RD:1', 1.0)]

    monkeypatch.setattr(module, 'build_ensemble_model', lambda: DummyModel())

    assert module.predict_ensemble(['HP:0000118'], topk=1) == [('RD:1', 1.0)]



def test_predict_ensemble_suppresses_internal_prints(monkeypatch, capsys):
    module = load_module()

    class DummyModel:
        def query(self, hpo_list, topk):
            print('query noise')
            return [('RD:1', 1.0)]

    def noisy_build():
        print('build noise')
        return DummyModel()

    monkeypatch.setattr(module, 'build_ensemble_model', noisy_build)

    assert module.predict_ensemble(['HP:0000118'], topk=1) == [('RD:1', 1.0)]
    captured = capsys.readouterr()
    assert captured.out == ''



def test_check_required_assets_requires_tf_checkpoint_data_file(tmp_path, monkeypatch):
    module = load_module()
    model_dir = tmp_path / 'model'
    monkeypatch.setattr(module, '_MODEL_DIR', model_dir)

    touch(model_dir / 'INTEGRATE_CCRD_OMIM_ORPHA/ICTODQAcrossModel/ICTODQAcross-Ave/dis_vec_mat.npz')
    touch(model_dir / 'INTEGRATE_CCRD_OMIM_ORPHA/HPOProbMNBModel/HPOProbMNB/dis_hpo_ances_mat.npz')
    touch(model_dir / 'INTEGRATE_CCRD_OMIM_ORPHA/CNBModel/CNB.joblib')
    touch(model_dir / 'INTEGRATE_CCRD_OMIM_ORPHA/LRNeuronModel/NN-Mixup-1/model.ckpt.index')

    with pytest.raises(FileNotFoundError) as exc_info:
        module._check_required_assets()

    assert 'model.ckpt.data-00000-of-00001' in str(exc_info.value)



def test_model_candidates_exposes_baseline_registry_in_expected_order():
    module = load_module()

    assert [candidate['name'] for candidate in module.MODEL_CANDIDATES] == [
        'ICTODQAcross-Ave-Random',
        'HPOProbMNB-Random',
        'CNB-Random',
        'NN-Mixup-Random-1',
        'MICAModel',
        'MICALinModel',
        'MICAJCModel',
        'MinICModel',
        'RBPModel',
        'GDDPFisherModel',
        'BOQAModel',
    ]



def test_build_available_models_defaults_to_icto_and_hpoprob(tmp_path, monkeypatch):
    module = load_module()
    install_fake_core_modules(monkeypatch)
    monkeypatch.setattr(module, '_MODEL_DIR', tmp_path / 'model')
    monkeypatch.setattr(module, '_PROJECT_CORE_DIR', tmp_path / 'core')

    models = module.build_available_models()

    assert [model.model_name for model in models] == [
        'ICTODQAcross-Ave-Random',
        'HPOProbMNB-Random',
        'MICAModel',
        'MICALinModel',
        'MICAJCModel',
        'MinICModel',
        'RBPModel',
        'GDDPFisherModel',
    ]



def test_build_available_models_includes_cnb_when_asset_exists(tmp_path, monkeypatch):
    module = load_module()
    install_fake_core_modules(monkeypatch)
    model_dir = tmp_path / 'model'
    monkeypatch.setattr(module, '_MODEL_DIR', model_dir)
    monkeypatch.setattr(module, '_PROJECT_CORE_DIR', tmp_path / 'core')
    touch(model_dir / 'INTEGRATE_CCRD_OMIM_ORPHA/CNBModel/CNB.joblib')

    models = module.build_available_models()

    assert [model.model_name for model in models] == [
        'ICTODQAcross-Ave-Random',
        'HPOProbMNB-Random',
        'CNB-Random',
        'MICAModel',
        'MICALinModel',
        'MICAJCModel',
        'MinICModel',
        'RBPModel',
        'GDDPFisherModel',
    ]



def test_build_available_models_includes_mica_family_and_minic_via_candidate_registry(tmp_path, monkeypatch):
    module = load_module()
    install_fake_core_modules(monkeypatch)
    monkeypatch.setattr(module, '_MODEL_DIR', tmp_path / 'model')

    monkeypatch.setattr(
        module,
        '_is_candidate_available',
        lambda candidate: candidate['name'] != 'CNB-Random' and candidate['name'] != 'NN-Mixup-Random-1',
        raising=False,
    )

    models = module.build_available_models()

    assert [model.model_name for model in models] == [
        'ICTODQAcross-Ave-Random',
        'HPOProbMNB-Random',
        'MICAModel',
        'MICALinModel',
        'MICAJCModel',
        'MinICModel',
        'RBPModel',
        'GDDPFisherModel',
        'BOQAModel',
    ]


def test_model_candidates_includes_rbp_model():
    module = load_module()

    assert any(candidate['name'] == 'RBPModel' for candidate in module.MODEL_CANDIDATES)


def test_build_available_models_includes_rbp_when_candidate_available(monkeypatch):
    module = load_module()
    install_fake_core_modules(monkeypatch)
    monkeypatch.setattr(
        module,
        '_is_candidate_available',
        lambda candidate: candidate['name'] in {
            'ICTODQAcross-Ave-Random',
            'HPOProbMNB-Random',
            'RBPModel',
        },
    )

    models = module.build_available_models()

    assert [model.name for model in models] == [
        'ICTODQAcross-Ave-Random',
        'HPOProbMNB-Random',
        'RBPModel',
    ]


def test_model_candidates_includes_gddp_fisher_model():
    module = load_module()

    assert any(candidate['name'] == 'GDDPFisherModel' for candidate in module.MODEL_CANDIDATES)


def test_build_available_models_includes_gddp_when_candidate_available(monkeypatch):
    module = load_module()
    install_fake_core_modules(monkeypatch)
    monkeypatch.setattr(
        module,
        '_is_candidate_available',
        lambda candidate: candidate['name'] in {
            'ICTODQAcross-Ave-Random',
            'HPOProbMNB-Random',
            'GDDPFisherModel',
        },
    )

    models = module.build_available_models()

    assert [model.name for model in models] == [
        'ICTODQAcross-Ave-Random',
        'HPOProbMNB-Random',
        'GDDPFisherModel',
    ]


def test_model_candidates_includes_boqa_model():
    module = load_module()

    assert any(candidate['name'] == 'BOQAModel' for candidate in module.MODEL_CANDIDATES)


def test_boqa_candidate_requires_legacy_hpoa_annotation(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, '_PROJECT_CORE_DIR', tmp_path / 'core')

    boqa_candidate = next(candidate for candidate in module.MODEL_CANDIDATES if candidate['name'] == 'BOQAModel')

    assert module._is_candidate_available(boqa_candidate) is False


def test_boqa_candidate_accepts_2022_hpo_assets(tmp_path, monkeypatch):
    module = load_module()
    project_core_dir = tmp_path / 'core'
    monkeypatch.setattr(module, '_PROJECT_CORE_DIR', project_core_dir)
    monkeypatch.setattr(module.shutil, 'which', lambda name: '/usr/bin/java' if name == 'java' else None)

    (project_core_dir / 'data/raw/HPO/2022/Ontology').mkdir(parents=True, exist_ok=True)
    (project_core_dir / 'data/raw/HPO/2022/Annotations').mkdir(parents=True, exist_ok=True)
    (project_core_dir / 'core/predict/prob_model/boqa-master/out/artifacts/boqa_jar').mkdir(parents=True, exist_ok=True)
    (project_core_dir / 'data/raw/HPO/2022/Ontology/hp.obo').write_text('format-version: 1.2')
    (project_core_dir / 'data/raw/HPO/2022/Annotations/phenotype.hpoa').write_text('#description')
    (project_core_dir / 'core/predict/prob_model/boqa-master/out/artifacts/boqa_jar/boqa.jar').write_text('jar')

    boqa_candidate = next(candidate for candidate in module.MODEL_CANDIDATES if candidate['name'] == 'BOQAModel')

    assert module._is_candidate_available(boqa_candidate) is True


def test_build_available_models_includes_boqa_when_candidate_available(monkeypatch):
    module = load_module()
    install_fake_core_modules(monkeypatch)
    monkeypatch.setattr(
        module,
        '_is_candidate_available',
        lambda candidate: candidate['name'] in {
            'ICTODQAcross-Ave-Random',
            'HPOProbMNB-Random',
            'BOQAModel',
        },
    )

    models = module.build_available_models()

    assert [model.name for model in models] == [
        'ICTODQAcross-Ave-Random',
        'HPOProbMNB-Random',
        'BOQAModel',
    ]



def test_cnb_candidate_accepts_flat_download_layout(tmp_path, monkeypatch):
    module = load_module()
    model_dir = tmp_path / 'model'
    monkeypatch.setattr(module, '_MODEL_DIR', model_dir)
    touch(model_dir / 'CNBModel/CNB.joblib')

    cnb_candidate = next(candidate for candidate in module.MODEL_CANDIDATES if candidate['name'] == 'CNB-Random')

    assert module._is_candidate_available(cnb_candidate) is True



def test_build_available_models_uses_flat_cnb_save_folder(tmp_path, monkeypatch):
    module = load_module()
    install_fake_core_modules(monkeypatch)
    model_dir = tmp_path / 'model'
    monkeypatch.setattr(module, '_MODEL_DIR', model_dir)
    touch(model_dir / 'CNBModel/CNB.joblib')

    models = module.build_available_models()
    cnb_model = next(model for model in models if model.model_name == 'CNB-Random')

    assert cnb_model.model_inits[0][2] == {
        'model_name': 'CNB',
        'save_folder': str(model_dir / 'CNBModel'),
    }



def test_mlp_candidate_accepts_flat_download_layout(tmp_path, monkeypatch):
    module = load_module()
    model_dir = tmp_path / 'model'
    monkeypatch.setattr(module, '_MODEL_DIR', model_dir)
    touch(model_dir / 'NN-Mixup-1/model.ckpt.index')
    touch(model_dir / 'NN-Mixup-1/model.ckpt.data-00000-of-00001')

    mlp_candidate = next(candidate for candidate in module.MODEL_CANDIDATES if candidate['name'] == 'NN-Mixup-Random-1')

    assert module._is_candidate_available(mlp_candidate) is True



def test_build_available_models_skips_mlp_when_tensorflow_unavailable(tmp_path, monkeypatch):
    module = load_module()
    install_fake_core_modules(monkeypatch)
    model_dir = tmp_path / 'model'
    monkeypatch.setattr(module, '_MODEL_DIR', model_dir)
    touch(model_dir / 'NN-Mixup-1/model.ckpt.index')
    touch(model_dir / 'NN-Mixup-1/model.ckpt.data-00000-of-00001')

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == 'core.predict.ml_model':
            raise ModuleNotFoundError("No module named 'tensorflow'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr('builtins.__import__', fake_import)

    models = module.build_available_models()

    assert 'NN-Mixup-Random-1' not in [model.model_name for model in models]



def test_run_quietly_returns_result_without_leaking_stdout(capsys):
    module = load_module()

    def noisy():
        print('noise')
        return 'ok'

    assert module.run_quietly(noisy) == 'ok'
    captured = capsys.readouterr()
    assert captured.out == ''



def test_format_results_table_renders_rank_disease_score_columns():
    module = load_module()
    results = [
        ('RD:7367', 1.0107991360691828),
        ('RD:6963', 1.0016198704104353),
    ]

    table = module.format_results_table(results)

    assert 'Rank' in table
    assert 'Disease' in table
    assert 'Score' in table
    assert 'RD:7367' in table
    assert '1.010799' in table



def test_format_results_table_handles_empty_results():
    module = load_module()
    assert module.format_results_table([]) == 'No results.'



def test_parse_hpo_list_splits_comma_separated_values():
    module = load_module()
    assert module.parse_hpo_list('HP:0001913,HP:0008513,HP:0001123') == [
        'HP:0001913',
        'HP:0008513',
        'HP:0001123',
    ]



def test_parse_args_uses_default_topk_and_no_hpo_list():
    module = load_module()
    args = module.parse_args([])
    assert args.topk == 5
    assert args.hpo_list is None



def test_parse_args_accepts_topk_and_hpo_list_overrides():
    module = load_module()
    args = module.parse_args(['--topk', '10', '--hpo-list', 'HP:1,HP:2'])
    assert args.topk == 10
    assert args.hpo_list == 'HP:1,HP:2'



def test_build_ensemble_model_returns_subset_ensemble_when_multiple_models_available(monkeypatch):
    module = load_module()

    class DummyModel:
        def __init__(self, model_name, hpo_reader):
            self.model_name = model_name
            self.hpo_reader = hpo_reader

    class DummyReader:
        pass

    shared_reader = DummyReader()
    models = [DummyModel('ICTODQAcross-Ave-Random', shared_reader), DummyModel('HPOProbMNB-Random', shared_reader)]
    monkeypatch.setattr(module, 'build_available_models', lambda: models)

    class DummyOrderStatisticMultiModel:
        def __init__(self, model_list=None, hpo_reader=None, model_name=None, **kwargs):
            self.model_list = model_list
            self.hpo_reader = hpo_reader
            self.model_name = model_name
            self.kwargs = kwargs

    fake_ensemble_module = types.ModuleType('core.predict.ensemble')
    fake_ensemble_module.OrderStatisticMultiModel = DummyOrderStatisticMultiModel
    monkeypatch.setitem(sys.modules, 'core.predict.ensemble', fake_ensemble_module)

    ensemble = module.build_ensemble_model()

    assert ensemble.model_name == 'Ensemble'
    assert ensemble.model_list == models
    assert ensemble.hpo_reader is shared_reader
    assert ensemble.kwargs['keep_raw_score'] is False



def test_build_ensemble_model_returns_single_model_when_only_one_available(monkeypatch):
    module = load_module()

    dummy_model = object()
    monkeypatch.setattr(module, 'build_available_models', lambda: [dummy_model])

    assert module.build_ensemble_model() is dummy_model



def test_build_ensemble_model_raises_when_no_models_available(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, 'build_available_models', lambda: [])

    with pytest.raises(RuntimeError) as exc_info:
        module.build_ensemble_model()

    assert 'No diagnosis models are available' in str(exc_info.value)



def test_get_available_model_names_matches_built_models(monkeypatch):
    module = load_module()

    class DummyModel:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(module, 'build_available_models', lambda: [DummyModel('A'), DummyModel('B')])

    assert module.get_available_model_names() == ['A', 'B']



def test_get_available_model_names_suppresses_internal_prints(monkeypatch, capsys):
    module = load_module()

    class DummyModel:
        def __init__(self, name):
            self.name = name

    def noisy_build():
        print('training...')
        return [DummyModel('A')]

    monkeypatch.setattr(module, 'build_available_models', noisy_build)

    assert module.get_available_model_names() == ['A']
    captured = capsys.readouterr()
    assert captured.out == ''



def test_describe_available_models_reports_subset(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, 'get_available_model_names', lambda: ['ICTODQAcross-Ave-Random', 'HPOProbMNB-Random'])

    assert module.describe_available_models() == 'Available models: ICTODQAcross-Ave-Random, HPOProbMNB-Random'


def test_describe_available_models_reports_new_baselines(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, 'get_available_model_names', lambda: ['RBPModel', 'GDDPFisherModel'])

    assert 'RBPModel' in module.describe_available_models()
    assert 'GDDPFisherModel' in module.describe_available_models()



def test_build_ensemble_model_uses_expected_wiring(monkeypatch):
    module = load_module()
    fake_types = install_fake_core_modules(monkeypatch)
    monkeypatch.setattr(module, '_is_candidate_available', lambda candidate: True)

    ensemble = module.build_ensemble_model()

    assert ensemble.model_name == 'Ensemble'
    assert ensemble.hpo_reader.keep_dnames == ['OMIM', 'ORPHA', 'CCRD']
    assert ensemble.hpo_reader.rm_no_use_hpo is False
    assert isinstance(ensemble, fake_types['FakeOrderStatisticMultiModel'])

    assert [model.model_name for model in ensemble.model_list] == [
        'ICTODQAcross-Ave-Random',
        'HPOProbMNB-Random',
        'CNB-Random',
        'NN-Mixup-Random-1',
        'MICAModel',
        'MICALinModel',
        'MICAJCModel',
        'MinICModel',
        'RBPModel',
        'GDDPFisherModel',
        'BOQAModel',
    ]

    (
        icto_model,
        hpo_prob_model,
        cnb_model,
        mlp_model,
        mica_model,
        mica_lin_model,
        mica_jc_model,
        min_ic_model,
        rbp_model,
        gddp_model,
        boqa_model,
    ) = ensemble.model_list

    assert icto_model.model_inits[0][0] is fake_types['FakeICTODQAcrossModel']
    assert icto_model.model_inits[0][1] == (icto_model.hpo_reader,)
    assert icto_model.model_inits[0][2] == {
        'sym_mode': 'ave',
        'model_name': 'ICTODQAcross-Ave',
    }

    assert hpo_prob_model.model_inits[0][0] is fake_types['FakeHPOProbMNBModel']
    assert hpo_prob_model.model_inits[0][1] == (hpo_prob_model.hpo_reader,)
    assert hpo_prob_model.model_inits[0][2] == {
        'phe_list_mode': 'PHELIST_REDUCE',
        'p1': 0.65,
        'p2': None,
        'child_to_parent_prob': 'sum',
        'model_name': 'HPOProbMNB',
    }

    assert cnb_model.model_inits[0][0] is fake_types['FakeCNBModel']
    assert cnb_model.model_inits[0][1] == (cnb_model.hpo_reader, 'VEC_TYPE_0_1', 'PHELIST_ANCESTOR')
    assert cnb_model.model_inits[0][2] == {
        'model_name': 'CNB',
        'save_folder': str(module._MODEL_DIR / 'CNBModel'),
    }

    assert mlp_model.model_inits[0][0] is fake_types['FakeLRNeuronModel']
    assert mlp_model.model_inits[0][1] == (mlp_model.hpo_reader, 'VEC_TYPE_0_1')
    assert mlp_model.model_inits[0][2] == {'model_name': 'NN-Mixup-1'}

    assert mica_model.model_name == 'MICAModel'
    assert mica_lin_model.model_name == 'MICALinModel'
    assert mica_jc_model.model_name == 'MICAJCModel'
    assert min_ic_model.model_name == 'MinICModel'
    assert rbp_model.model_name == 'RBPModel'
    assert rbp_model.kwargs == {}
    assert gddp_model.model_name == 'GDDPFisherModel'
    assert gddp_model.kwargs == {}
    assert boqa_model.model_name == 'BOQAModel'
    assert boqa_model.kwargs == {}
