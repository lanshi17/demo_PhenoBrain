#!/usr/bin/env python3
"""Run the full GA4GH benchmark with the same metrics used for the MME benchmark."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / 'timgroup_disease_diagnosis' / 'codes' / 'core'
SCRIPT_DIR = CORE_DIR / 'core' / 'script'
DATASET_NAME = 'GA4GH'
GA4GH_DATASET_PATH = PROJECT_ROOT / 'data' / 'inputs' / 'GA4GH.benchmark_patients.json'
SUMMARY_PATH = PROJECT_ROOT / 'results' / 'ga4gh_benchmark_summary.json'
TOP_K_LIST = (1, 3, 5, 10, 30)
BENCHMARK_CPU_USE = 1
MME_COMPONENT_MODEL_NAMES = (
    'ICTODQAcross-Ave-Random',
    'HPOProbMNB-Random',
    'CNB-Random',
    'NN-Mixup-Random-1',
)


def configure_import_paths():
    for path in (CORE_DIR, SCRIPT_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def select_mme_component_models(available_models):
    by_name = {model.name: model for model in available_models}
    missing = [name for name in MME_COMPONENT_MODEL_NAMES if name not in by_name]
    if missing:
        available_names = ', '.join(model.name for model in available_models) or 'none'
        raise RuntimeError(
            'Missing MME benchmark component models: '
            + ', '.join(missing)
            + f'. Available models: {available_names}'
        )
    return [by_name[name] for name in MME_COMPONENT_MODEL_NAMES]


def build_benchmark_model():
    configure_import_paths()
    from example_predict_ensemble import _build_outer_ensemble, build_available_models, describe_available_models

    print('Available models:')
    print(describe_available_models())
    print()

    component_models = select_mme_component_models(build_available_models())
    return _build_outer_ensemble(component_models), component_models


def convert_dataset_answers_to_rd_codes(dataset, rd_reader, mapper):
    return [[hpo_list, mapper(dis_list, rd_reader)] for hpo_list, dis_list in dataset]


def build_testor(dataset_path=GA4GH_DATASET_PATH):
    configure_import_paths()
    from core.predict.model_testor import ModelTestor
    from core.reader import HPOIntegratedDatasetReader, source_codes_to_rd_codes
    from core.utils.constant import CUSTOM_DATA

    hpo_reader = HPOIntegratedDatasetReader(
        keep_dnames=['OMIM', 'ORPHA', 'CCRD'],
        rm_no_use_hpo=False,
    )
    testor = ModelTestor(eval_data=CUSTOM_DATA, hpo_reader=hpo_reader)
    testor.set_custom_data_set(name_to_path={DATASET_NAME: str(dataset_path)}, data_names=[DATASET_NAME])
    testor.load_test_data(data_names=[DATASET_NAME])
    testor.data[DATASET_NAME] = convert_dataset_answers_to_rd_codes(
        testor.data[DATASET_NAME],
        testor.get_rd_reader(),
        source_codes_to_rd_codes,
    )
    return testor


def top_k_count(recall, total):
    return int(round(recall * total))


def build_summary(model, component_models, dataset_size, metrics):
    return {
        'model': model.name,
        'dataset': DATASET_NAME,
        'num_patients': dataset_size,
        'component_models': [model.name for model in component_models],
        'metrics': metrics,
        'top_k_summary': {
            f'top{k}': {
                'count': top_k_count(metrics[f'Mic.Recall.{k}'], dataset_size),
                'total': dataset_size,
                'recall': metrics[f'Mic.Recall.{k}'],
            }
            for k in TOP_K_LIST
            if f'Mic.Recall.{k}' in metrics
        },
    }


def print_summary(summary, results_path):
    print('\n' + '=' * 60)
    print(f"Benchmark Results for {summary['dataset']} ({summary['model']})")
    print('=' * 60)
    for k in TOP_K_LIST:
        item = summary['top_k_summary'].get(f'top{k}')
        if item:
            print(f"top{k}: {item['count']}/{item['total']} ({item['recall']:.4f})")
    rank_median = summary['metrics'].get('Mic.RankMedian')
    if rank_median is not None:
        print(f'\nMedian Rank: {rank_median:.2f}')
    print('\nFull results saved to:')
    print(f'  {results_path}')


def write_summary(summary, summary_path=SUMMARY_PATH):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    return summary_path


def run_benchmark():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    model, component_models = build_benchmark_model()
    print(f'Running benchmark with model: {model.name}')
    print(f'Number of component models: {len(component_models)}')
    print()

    testor = build_testor()
    logger.info('Starting benchmark calculation...')
    metric_dict = testor.cal_metric_and_save(
        model,
        data_names=[DATASET_NAME],
        cpu_use=BENCHMARK_CPU_USE,
        use_query_many=False,
        save_raw_results=True,
        logger=logger,
    )
    metrics = metric_dict[DATASET_NAME]
    summary = build_summary(model, component_models, testor.get_dataset_size(DATASET_NAME), metrics)
    print_summary(summary, testor.RESULT_PATH)
    summary_path = write_summary(summary)
    print(f'\nSummary saved to: {summary_path}')
    return summary


def main():
    run_benchmark()


if __name__ == '__main__':
    main()
