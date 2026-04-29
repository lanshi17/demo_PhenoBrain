#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / 'timgroup_disease_diagnosis' / 'codes' / 'core'
SCRIPT_DIR = CORE_DIR / 'core' / 'script'
SUMMARY_PATH = PROJECT_ROOT / 'results' / 'benchmark_summary.json'
DEFAULT_DATASETS = ('MME', 'GA4GH')
DEFAULT_METRICS = ('top1', 'top3', 'top5', 'top10', 'top30')
SUPPORTED_METRICS = DEFAULT_METRICS
DEFAULT_ENSEMBLES = ('HPOP-ICT-CNB-NN',)
ENSEMBLE_PRESETS = {
    'HPOP-ICT-CNB-NN': (
        'HPOProbMNB-Random',
        'ICTODQAcross-Ave-Random',
        'CNB-Random',
        'NN-Mixup-Random-1',
    ),
}
DATASET_SPECS = {
    'MME': {
        'questions': PROJECT_ROOT / 'data' / 'inputs' / 'MME.benchmark_patients.questions.json',
        'answers': PROJECT_ROOT / 'data' / 'inputs' / 'MME.benchmark_patients.answers.json',
    },
    'GA4GH': {
        'questions': PROJECT_ROOT / 'data' / 'GA4GH' / 'GA4GH.benchmark_patients.questions.json',
        'answers': PROJECT_ROOT / 'data' / 'GA4GH' / 'GA4GH.benchmark_patients.answers.json',
    },
}


def parse_csv(value):
    if value is None:
        return None
    return [item.strip() for item in value.split(',') if item.strip()]


def parse_ensembles(value):
    items = parse_csv(value)
    if items is None:
        return list(DEFAULT_ENSEMBLES)
    if [item.lower() for item in items] == ['none']:
        return []
    return items


def metric_names_to_keys(metric_names):
    return {f'Mic.Recall.{metric[3:]}' for metric in metric_names}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Run PhenoBrain benchmark datasets.')
    parser.add_argument('--model', dest='models', type=parse_csv)
    parser.add_argument('--ensemble', dest='ensembles', type=parse_ensembles, default=list(DEFAULT_ENSEMBLES))
    parser.add_argument('--dataset', dest='datasets', type=parse_csv, default=list(DEFAULT_DATASETS))
    parser.add_argument('--metrics', type=parse_csv, default=list(DEFAULT_METRICS))
    parser.add_argument('--list-models', action='store_true')
    args = parser.parse_args(argv)
    for metric in args.metrics:
        if metric not in SUPPORTED_METRICS:
            parser.error(f'Unsupported metric: {metric}')
    return args


def configure_import_paths():
    for path in (CORE_DIR, SCRIPT_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def build_available_models():
    configure_import_paths()
    from example_predict_ensemble import build_available_models as build_models
    return build_models()


def run_quietly(fn, *args, **kwargs):
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return fn(*args, **kwargs)


def get_available_model_names():
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    return [model.name for model in build_available_models()]


def build_outer_ensemble(models):
    configure_import_paths()
    from example_predict_ensemble import _build_outer_ensemble
    return _build_outer_ensemble(models)


def select_single_models(available_models, requested_names):
    if requested_names is None:
        return list(available_models)
    by_name = {model.name: model for model in available_models}
    selected = []
    for name in requested_names:
        if name not in by_name:
            raise RuntimeError(f'Missing requested model: {name}')
        selected.append(by_name[name])
    return selected


def select_component_models(available_models, component_names, ensemble_name):
    by_name = {model.name: model for model in available_models}
    missing = [name for name in component_names if name not in by_name]
    if missing:
        raise RuntimeError(f'Missing component models for {ensemble_name}: ' + ', '.join(missing))
    return [by_name[name] for name in component_names]


def build_ensemble_models(available_models, ensemble_names):
    ensembles = []
    for ensemble_name in ensemble_names:
        if ensemble_name not in ENSEMBLE_PRESETS:
            raise RuntimeError(f'Unknown ensemble preset: {ensemble_name}')
        components = select_component_models(available_models, ENSEMBLE_PRESETS[ensemble_name], ensemble_name)
        ensemble = build_outer_ensemble(components)
        ensemble.name = ensemble_name
        ensembles.append(ensemble)
    return ensembles


def load_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def normalize_omim_id(omim_id):
    if omim_id.startswith('MIM:'):
        return 'OMIM:' + omim_id[4:]
    return omim_id


def load_benchmark_dataset(spec):
    questions = {item['patient_id']: item for item in load_json(spec['questions'])}
    answers = {item['patient_id']: item for item in load_json(spec['answers'])}
    dataset = []
    for patient_id, question in questions.items():
        answer = answers.get(patient_id)
        if answer is None or not answer['answers']:
            continue
        hpo_list = [hpo['hpo_id'] for hpo in question['hpo_terms']]
        disease_list = [normalize_omim_id(item['omim_id']) for item in answer['answers']]
        dataset.append([hpo_list, disease_list])
    return dataset


def resolve_dataset_specs(dataset_names):
    specs = {}
    for dataset_name in dataset_names:
        if dataset_name not in DATASET_SPECS:
            raise RuntimeError(f'Unknown dataset: {dataset_name}')
        specs[dataset_name] = DATASET_SPECS[dataset_name]
    return specs


def convert_dataset_answers_to_rd_codes(dataset, rd_reader, mapper):
    return [[hpo_list, mapper(dis_list, rd_reader)] for hpo_list, dis_list in dataset]


def build_testor(dataset_name, dataset_spec):
    configure_import_paths()
    from core.predict.model_testor import ModelTestor
    from core.reader import HPOIntegratedDatasetReader, source_codes_to_rd_codes
    from core.utils.constant import CUSTOM_DATA

    hpo_reader = HPOIntegratedDatasetReader(keep_dnames=['OMIM', 'ORPHA', 'CCRD'], rm_no_use_hpo=False)
    testor = ModelTestor(eval_data=CUSTOM_DATA, hpo_reader=hpo_reader)
    testor.data_names = [dataset_name]
    dataset = load_benchmark_dataset(dataset_spec)
    testor.data[dataset_name] = convert_dataset_answers_to_rd_codes(
        dataset,
        testor.get_rd_reader(),
        source_codes_to_rd_codes,
    )
    return testor


def top_k_count(recall, total):
    return int(round(recall * total))


def build_result_summary(model, dataset_name, dataset_size, requested_metrics, metrics):
    top_k_summary = {}
    for metric_name in requested_metrics:
        metric_key = f'Mic.Recall.{metric_name[3:]}'
        if metric_key in metrics:
            top_k_summary[metric_name] = {
                'count': top_k_count(metrics[metric_key], dataset_size),
                'total': dataset_size,
                'recall': metrics[metric_key],
            }
    return {
        'model': model.name,
        'dataset': dataset_name,
        'num_patients': dataset_size,
        'metrics': metrics,
        'top_k_summary': top_k_summary,
    }


def print_result_summary(summary):
    print(f"\n{summary['dataset']} / {summary['model']}")
    for metric_name, item in summary['top_k_summary'].items():
        print(f"{metric_name}: {item['count']}/{item['total']} ({item['recall']:.4f})")


def write_summary(summary, summary_path=SUMMARY_PATH):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    return summary_path


def run_benchmark(requested_models, requested_ensembles, requested_datasets, requested_metrics):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    available_models = build_available_models()
    models = select_single_models(available_models, requested_models)
    models.extend(build_ensemble_models(available_models, requested_ensembles))
    dataset_specs = resolve_dataset_specs(requested_datasets)
    metric_set = metric_names_to_keys(requested_metrics)
    runs = []

    for dataset_name, dataset_spec in dataset_specs.items():
        testor = build_testor(dataset_name, dataset_spec)
        for model in models:
            metric_dict = testor.cal_metric_and_save(
                model,
                data_names=[dataset_name],
                metric_set=metric_set,
                cpu_use=1,
                use_query_many=False,
                save_raw_results=True,
                logger=logger,
            )
            result = build_result_summary(
                model,
                dataset_name,
                testor.get_dataset_size(dataset_name),
                requested_metrics,
                metric_dict[dataset_name],
            )
            print_result_summary(result)
            runs.append(result)

    summary = {
        'datasets': requested_datasets,
        'metrics': requested_metrics,
        'models': [model.name for model in models],
        'runs': runs,
    }
    write_summary(summary)
    return summary


def print_models():
    available_model_names = run_quietly(get_available_model_names)
    print('Available single models:')
    for model_name in available_model_names:
        print(f'  {model_name}')
    print('Available ensembles:')
    for ensemble_name in ENSEMBLE_PRESETS:
        print(f'  {ensemble_name}')


def main(argv=None):
    args = parse_args(argv)
    if args.list_models:
        print_models()
        return None
    return run_benchmark(args.models, args.ensembles, args.datasets, args.metrics)


if __name__ == '__main__':
    main()
