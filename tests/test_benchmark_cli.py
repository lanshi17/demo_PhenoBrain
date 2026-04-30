import gc
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark.py'


def load_module():
    spec = importlib.util.spec_from_file_location('benchmark_cli', str(SCRIPT_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_defaults_to_all_datasets_all_metrics_and_default_ensemble():
    module = load_module()

    args = module.parse_args([])

    assert args.models is None
    assert args.ensembles == ['HPOP-ICT-CNB-NN']
    assert args.datasets == ['MME', 'GA4GH']
    assert args.metrics == ['top1', 'top3', 'top5', 'top10', 'top30']


def test_parse_args_accepts_comma_separated_overrides():
    module = load_module()

    args = module.parse_args([
        '--model', 'MICAModel,RBPModel',
        '--ensemble', 'none',
        '--dataset', 'MME',
        '--metrics', 'top1,top10',
    ])

    assert args.models == ['MICAModel', 'RBPModel']
    assert args.ensembles == []
    assert args.datasets == ['MME']
    assert args.metrics == ['top1', 'top10']


def test_metric_names_convert_to_model_testor_metric_keys():
    module = load_module()

    assert module.metric_names_to_keys(['top1', 'top30']) == {'Mic.Recall.1', 'Mic.Recall.30'}


def test_parse_args_rejects_unknown_metrics():
    module = load_module()

    with pytest.raises(SystemExit):
        module.parse_args(['--metrics', 'top2'])


class DummyModel:
    def __init__(self, name):
        self.name = name
        self.hpo_reader = object()


def test_select_single_models_defaults_to_all_available_models():
    module = load_module()
    available = [DummyModel('A'), DummyModel('B')]

    assert module.select_single_models(available, None) == available


def test_select_single_models_uses_requested_order():
    module = load_module()
    available = [DummyModel('A'), DummyModel('B')]

    selected = module.select_single_models(available, ['B', 'A'])

    assert [model.name for model in selected] == ['B', 'A']


def test_select_single_models_reports_missing_name():
    module = load_module()

    with pytest.raises(RuntimeError, match='Missing requested model: C'):
        module.select_single_models([DummyModel('A')], ['C'])


def test_build_ensemble_models_uses_named_preset_components(monkeypatch):
    module = load_module()
    available = [
        DummyModel('HPOProbMNB'),
        DummyModel('ICTODQAcross-Ave'),
        DummyModel('CNB'),
        DummyModel('NN-Mixup-1'),
    ]
    captured = {}

    def fake_outer_ensemble(models):
        captured['models'] = models
        model = DummyModel('Ensemble')
        model.name = 'HPOP-ICT-CNB-NN'
        return model

    monkeypatch.setattr(module, 'build_outer_ensemble', fake_outer_ensemble)

    ensembles = module.build_ensemble_models(available, ['HPOP-ICT-CNB-NN'])

    assert [model.name for model in captured['models']] == list(module.ENSEMBLE_PRESETS['HPOP-ICT-CNB-NN'])
    assert [model.name for model in ensembles] == ['HPOP-ICT-CNB-NN']


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def test_load_benchmark_dataset_converts_questions_and_answers(tmp_path):
    module = load_module()
    questions_path = tmp_path / 'questions.json'
    answers_path = tmp_path / 'answers.json'
    write_json(questions_path, [
        {'patient_id': 'p1', 'hpo_terms': [{'hpo_id': 'HP:1'}, {'hpo_id': 'HP:2'}]},
        {'patient_id': 'p2', 'hpo_terms': [{'hpo_id': 'HP:3'}]},
    ])
    write_json(answers_path, [
        {'patient_id': 'p1', 'answers': [{'omim_id': 'MIM:123'}]},
        {'patient_id': 'p2', 'answers': []},
    ])

    dataset = module.load_benchmark_dataset({'questions': questions_path, 'answers': answers_path})

    assert dataset == [[['HP:1', 'HP:2'], ['OMIM:123']]]


def test_load_benchmark_dataset_remaps_obsolete_hpo_ids(tmp_path):
    module = load_module()
    questions_path = tmp_path / 'questions.json'
    answers_path = tmp_path / 'answers.json'
    write_json(questions_path, [
        {'patient_id': 'p1', 'hpo_terms': [{'hpo_id': 'HP:old'}, {'hpo_id': 'HP:current'}]},
    ])
    write_json(answers_path, [
        {'patient_id': 'p1', 'answers': [{'omim_id': 'MIM:123'}]},
    ])

    dataset = module.load_benchmark_dataset(
        {'questions': questions_path, 'answers': answers_path},
        hpo_dict={'HP:current': {}},
        old_to_new_hpo={'HP:old': 'HP:current'},
    )

    assert dataset == [[['HP:current', 'HP:current'], ['OMIM:123']]]


def test_resolve_dataset_specs_uses_requested_order():
    module = load_module()

    specs = module.resolve_dataset_specs(['GA4GH', 'MME'])

    assert list(specs) == ['GA4GH', 'MME']


def test_build_summary_includes_requested_top_k_counts():
    module = load_module()
    model = DummyModel('MICAModel')
    metrics = {'Mic.Recall.1': 0.5, 'Mic.Recall.10': 0.75, 'Mic.RankMedian': 4.0}

    summary = module.build_result_summary(model, 'MME', 4, ['top1', 'top10'], metrics)

    assert summary['model'] == 'MICAModel'
    assert summary['dataset'] == 'MME'
    assert summary['num_patients'] == 4
    assert summary['top_k_summary'] == {
        'top1': {'count': 2, 'total': 4, 'recall': 0.5},
        'top10': {'count': 3, 'total': 4, 'recall': 0.75},
    }
    assert summary['metrics'] == metrics


def test_print_result_summary_renders_metrics(capsys):
    module = load_module()
    summary = {
        'model': 'MICAModel',
        'dataset': 'MME',
        'top_k_summary': {'top1': {'count': 2, 'total': 4, 'recall': 0.5}},
    }

    module.print_result_summary(summary)

    assert 'MME / MICAModel' in capsys.readouterr().out


def test_run_benchmark_runs_each_model_dataset_pair(monkeypatch, tmp_path):
    module = load_module()
    calls = []

    class DummyTestor:
        def __init__(self, dataset_name):
            self.dataset_name = dataset_name

        def cal_metric_and_save(self, model, data_names, metric_set, cpu_use, use_query_many, save_raw_results, logger):
            calls.append((model.name, data_names, metric_set, cpu_use, use_query_many, save_raw_results))
            return {self.dataset_name: {'Mic.Recall.1': 0.5}}

        def get_dataset_size(self, dataset_name):
            return 2

    monkeypatch.setattr(module, 'build_available_models', lambda: [DummyModel('A'), DummyModel('B')])
    monkeypatch.setattr(module, 'build_ensemble_models', lambda available, names: [DummyModel('HPOP-ICT-CNB-NN')])
    monkeypatch.setattr(module, 'build_testor', lambda dataset_name, spec: DummyTestor(dataset_name))
    monkeypatch.setattr(module, 'write_summary', lambda summary, summary_path=module.SUMMARY_PATH: tmp_path / 'summary.json')

    summary = module.run_benchmark(
        requested_models=['A'],
        requested_ensembles=['HPOP-ICT-CNB-NN'],
        requested_datasets=['MME'],
        requested_metrics=['top1'],
    )

    assert [call[0] for call in calls] == ['A', 'HPOP-ICT-CNB-NN']
    assert calls[0][2] == {'Mic.Recall.1'}
    assert calls[0][3:] == (1, False, True)
    assert len(summary['runs']) == 2


def test_write_summary_creates_json(tmp_path):
    module = load_module()
    path = tmp_path / 'summary.json'
    value = {'runs': []}

    assert module.write_summary(value, path) == path
    assert json.loads(path.read_text(encoding='utf-8')) == value


def test_print_models_suppresses_model_discovery_noise(monkeypatch, capsys):
    module = load_module()

    def noisy_model_names():
        print('training...')
        return ['MICAModel']

    monkeypatch.setattr(module, 'get_available_model_names', noisy_model_names)

    module.print_models()

    captured = capsys.readouterr()
    assert 'training...' not in captured.out
    assert 'MICAModel' in captured.out
    assert 'HPOP-ICT-CNB-NN' in captured.out


def test_get_available_model_names_collects_delayed_model_teardown_noise(monkeypatch, capsys):
    module = load_module()

    class NoisyCyclicModel:
        name = 'NN-Mixup-1'

        def __init__(self):
            self.cycle = self

        def __del__(self):
            print('__del__ starts running...')

    monkeypatch.setattr(module, 'build_available_models', lambda: [NoisyCyclicModel()])

    names = module.run_quietly(module.get_available_model_names)
    gc.collect()

    assert names == ['NN-Mixup-1']
    assert '__del__ starts running...' not in capsys.readouterr().out
