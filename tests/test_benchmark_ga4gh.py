import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark_ga4gh.py'


def load_module():
    spec = importlib.util.spec_from_file_location('benchmark_ga4gh', str(SCRIPT_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyModel:
    name = 'Ensemble'


class DummyComponent:
    def __init__(self, name):
        self.name = name


def test_benchmark_paths_are_project_relative():
    module = load_module()
    root = Path(__file__).resolve().parents[1]

    assert module.PROJECT_ROOT == root
    assert module.GA4GH_DATASET_PATH == root / 'data' / 'inputs' / 'GA4GH.benchmark_patients.json'
    assert module.SUMMARY_PATH == root / 'results' / 'ga4gh_benchmark_summary.json'


def test_select_mme_component_models_filters_extra_models_in_expected_order():
    module = load_module()
    available = [
        DummyComponent('MICAModel'),
        DummyComponent('CNB'),
        DummyComponent('ICTODQAcross-Ave'),
        DummyComponent('NN-Mixup-1'),
        DummyComponent('HPOProbMNB'),
    ]

    selected = module.select_mme_component_models(available)

    assert [model.name for model in selected] == list(module.MME_COMPONENT_MODEL_NAMES)


def test_select_mme_component_models_reports_missing_models():
    module = load_module()
    available = [
        DummyComponent('ICTODQAcross-Ave'),
        DummyComponent('HPOProbMNB'),
        DummyComponent('CNB'),
    ]

    with pytest.raises(RuntimeError, match='NN-Mixup-1'):
        module.select_mme_component_models(available)


def test_convert_dataset_answers_to_rd_codes_maps_source_codes():
    module = load_module()
    rd_reader = object()

    def mapper(dis_list, reader):
        assert reader is rd_reader
        return [f'RD:{dis_code.rsplit(":", 1)[1]}' for dis_code in dis_list]

    dataset = [
        [['HP:1'], ['OMIM:1']],
        [['HP:2'], ['OMIM:2', 'OMIM:3']],
    ]

    assert module.convert_dataset_answers_to_rd_codes(dataset, rd_reader, mapper) == [
        [['HP:1'], ['RD:1']],
        [['HP:2'], ['RD:2', 'RD:3']],
    ]


def test_run_benchmark_uses_single_process_for_tensorflow_ensemble(monkeypatch, tmp_path):
    module = load_module()
    calls = {}

    class DummyTestor:
        RESULT_PATH = 'dummy-results'

        def cal_metric_and_save(self, model, data_names, cpu_use, use_query_many, save_raw_results, logger):
            calls['cpu_use'] = cpu_use
            calls['use_query_many'] = use_query_many
            calls['save_raw_results'] = save_raw_results
            return {'GA4GH': {'Mic.Recall.1': 0.5}}

        def get_dataset_size(self, data_name):
            return 2

    monkeypatch.setattr(module, 'build_benchmark_model', lambda: (DummyModel(), [DummyComponent('ICTODQAcross-Ave')]))
    monkeypatch.setattr(module, 'build_testor', lambda: DummyTestor())
    monkeypatch.setattr(module, 'write_summary', lambda summary: tmp_path / 'summary.json')

    summary = module.run_benchmark()

    assert calls == {
        'cpu_use': 1,
        'use_query_many': False,
        'save_raw_results': True,
    }
    assert summary['top_k_summary']['top1'] == {'count': 1, 'total': 2, 'recall': 0.5}


def test_build_summary_rounds_top_k_counts():
    module = load_module()
    metrics = {
        'Mic.Recall.1': 0.5,
        'Mic.Recall.3': 0.7209,
        'Mic.Recall.5': 0.7674,
        'Mic.RankMedian': 4.0,
    }

    summary = module.build_summary(
        model=DummyModel(),
        component_models=[DummyComponent('ICTODQAcross-Ave'), DummyComponent('HPOProbMNB')],
        dataset_size=43,
        metrics=metrics,
    )

    assert summary['model'] == 'Ensemble'
    assert summary['dataset'] == 'GA4GH'
    assert summary['num_patients'] == 43
    assert summary['component_models'] == ['ICTODQAcross-Ave', 'HPOProbMNB']
    assert summary['top_k_summary']['top3'] == {'count': 31, 'total': 43, 'recall': 0.7209}
    assert summary['metrics'] == metrics
