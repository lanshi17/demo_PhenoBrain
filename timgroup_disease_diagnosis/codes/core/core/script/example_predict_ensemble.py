"""Offline subset-Ensemble prediction example.

Usage:
    PYTHONPATH=/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core \
    python /mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py

The script uses the subset of locally available diagnosis models.
`ICTO(A)`, `HPOProb`, `MICA`, `MinIC`, `RBP`, and `GDDP`
are constructed locally; `CNB`, `MLP`, and `BOQA` are included only when
their supporting assets exist under the repository data/model paths.
"""

from __future__ import annotations

import argparse
import io
import shutil
import warnings
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path


_PROJECT_CORE_DIR = Path(__file__).resolve().parents[2]
_MODEL_DIR = _PROJECT_CORE_DIR / 'model'
_REQUIRED_ASSETS = (
    ('ICTODQAcross-Ave', Path('INTEGRATE_CCRD_OMIM_ORPHA/ICTODQAcrossModel/ICTODQAcross-Ave/dis_vec_mat.npz')),
    ('HPOProbMNB', Path('INTEGRATE_CCRD_OMIM_ORPHA/HPOProbMNBModel/HPOProbMNB/dis_hpo_ances_mat.npz')),
    ('CNB', Path('INTEGRATE_CCRD_OMIM_ORPHA/CNBModel/CNB.joblib')),
    ('NN-Mixup-1 checkpoint index', Path('INTEGRATE_CCRD_OMIM_ORPHA/NN-Mixup-1/model.ckpt.index')),
    ('NN-Mixup-1 checkpoint data', Path('INTEGRATE_CCRD_OMIM_ORPHA/NN-Mixup-1/model.ckpt.data-00000-of-00001')),
)

MODEL_CANDIDATES = [
    {'name': 'ICTODQAcross-Ave', 'kind': 'baseline'},
    {'name': 'HPOProbMNB', 'kind': 'baseline'},
    {
        'name': 'CNB',
        'kind': 'optional',
        'required_asset_groups': [[
            Path('INTEGRATE_CCRD_OMIM_ORPHA/CNBModel/CNB.joblib'),
        ], [
            Path('CNBModel/CNB.joblib'),
        ]],
    },
    {
        'name': 'NN-Mixup-1',
        'kind': 'optional',
        'required_asset_groups': [[
            Path('INTEGRATE_CCRD_OMIM_ORPHA/NN-Mixup-1/model.ckpt.index'),
            Path('INTEGRATE_CCRD_OMIM_ORPHA/NN-Mixup-1/model.ckpt.data-00000-of-00001'),
        ], [
            Path('INTEGRATE_CCRD_OMIM_ORPHA/LRNeuronModel/NN-Mixup-1/model.ckpt.index'),
            Path('INTEGRATE_CCRD_OMIM_ORPHA/LRNeuronModel/NN-Mixup-1/model.ckpt.data-00000-of-00001'),
        ], [
            Path('NN-Mixup-1/model.ckpt.index'),
            Path('NN-Mixup-1/model.ckpt.data-00000-of-00001'),
        ]],
    },
    {'name': 'MICAModel', 'kind': 'baseline'},
    {'name': 'MICALinModel', 'kind': 'baseline'},
    {'name': 'MICAJCModel', 'kind': 'baseline'},
    {'name': 'MinICModel', 'kind': 'baseline'},
    {'name': 'RBPModel', 'kind': 'baseline'},
    {'name': 'GDDPFisherModel', 'kind': 'baseline'},
    {
        'name': 'BOQAModel',
        'kind': 'optional',
        'availability_check': '_boqa_assets_available',
    },
]


DEFAULT_SAMPLE_HPO_LIST = [
    'HP:0001913',
    'HP:0008513',
    'HP:0001123',
    'HP:0000365',
    'HP:0002857',
    'HP:0001744',
]



def parse_hpo_list(value):
    return [item.strip() for item in value.split(',') if item.strip()]



def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--hpo-list')
    return parser.parse_args(argv)



def configure_quiet_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=Warning, message='.*InconsistentVersionWarning.*')
    warnings.filterwarnings('ignore', category=Warning, message='.*VisibleDeprecationWarning.*')



def run_quietly(fn, *args, **kwargs):
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return fn(*args, **kwargs)



def _resolve_hpo_raw_path(preferred_relative_path, fallback_relative_path):
    preferred_path = _PROJECT_CORE_DIR / preferred_relative_path
    fallback_path = _PROJECT_CORE_DIR / fallback_relative_path
    if preferred_path.exists():
        return preferred_path
    if fallback_path.exists():
        return fallback_path
    return preferred_path


def _boqa_required_paths():
    return (
        _resolve_hpo_raw_path('data/raw/HPO/2019/Ontology/hp.obo', 'data/raw/HPO/2022/Ontology/hp.obo'),
        _resolve_hpo_raw_path('data/raw/HPO/2019/Annotations/phenotype.hpoa', 'data/raw/HPO/2022/Annotations/phenotype.hpoa'),
        _PROJECT_CORE_DIR / 'core/predict/prob_model/boqa-master/out/artifacts/boqa_jar/boqa.jar',
    )


def _boqa_assets_available():
    return all(path.exists() for path in _boqa_required_paths()) and shutil.which('java') is not None


def _is_candidate_available(candidate):
    availability_check = candidate.get('availability_check')
    if availability_check:
        return globals()[availability_check]()

    required_asset_groups = candidate.get('required_asset_groups')
    if required_asset_groups:
        return any(
            all((_MODEL_DIR / relative_path).exists() for relative_path in group)
            for group in required_asset_groups
        )
    required_assets = candidate.get('required_assets', ())
    return all((_MODEL_DIR / relative_path).exists() for relative_path in required_assets)


def _check_required_assets() -> None:
    missing = [
        (label, _MODEL_DIR / relative_path)
        for label, relative_path in _REQUIRED_ASSETS
        if not (_MODEL_DIR / relative_path).exists()
    ]
    if not missing:
        return

    missing_lines = '\n'.join(f'- {label}: {path}' for label, path in missing)
    raise FileNotFoundError(
        'Missing Ensemble model assets. Download or train the required model files under '
        f'{_MODEL_DIR} before calling build_ensemble_model().\n{missing_lines}'
    )


def build_available_models():
    from core.predict.prob_model import CNBModel, HPOProbMNBModel
    from core.predict.prob_model import BOQAModel
    from core.predict.sim_model import (
        GDDPFisherModel,
        ICTODQAcrossModel,
        MICAJCModel,
        MICALinModel,
        MICAModel,
        MinICModel,
        RBPModel,
    )
    from core.reader import HPOIntegratedDatasetReader
    from core.utils.constant import PHELIST_ANCESTOR, PHELIST_REDUCE, PREDICT_MODE, VEC_TYPE_0_1

    hpo_reader_with_all_hpo = HPOIntegratedDatasetReader(
        keep_dnames=['OMIM', 'ORPHA', 'CCRD'],
        rm_no_use_hpo=False,
    )
    hpo_reader_rm_unused = HPOIntegratedDatasetReader(
        keep_dnames=['OMIM', 'ORPHA', 'CCRD'],
        rm_no_use_hpo=True,
    )

    def get_mlp_kwargs():
        for relative_folder in (
            Path('INTEGRATE_CCRD_OMIM_ORPHA/NN-Mixup-1'),
            Path('INTEGRATE_CCRD_OMIM_ORPHA/LRNeuronModel/NN-Mixup-1'),
            Path('NN-Mixup-1'),
        ):
            folder = _MODEL_DIR / relative_folder
            if (folder / 'model.ckpt.index').exists() and (folder / 'model.ckpt.data-00000-of-00001').exists():
                return {
                    'model_name': 'NN-Mixup-1',
                    'save_folder': str(folder),
                }
        return {
            'model_name': 'NN-Mixup-1',
        }

    def build_mlp_model():
        from core.predict.ml_model import LRNeuronModel

        return LRNeuronModel(
            hpo_reader_rm_unused,
            vec_type=VEC_TYPE_0_1,
            **get_mlp_kwargs(),
        )

    def get_cnb_kwargs():
        flat_cnb_folder = _MODEL_DIR / 'CNBModel'
        if (flat_cnb_folder / 'CNB.joblib').exists():
            return {
                'model_name': 'CNB',
                'save_folder': str(flat_cnb_folder),
            }
        return {
            'model_name': 'CNB',
        }

    builders = {
        'ICTODQAcross-Ave': lambda: ICTODQAcrossModel(
            hpo_reader=hpo_reader_with_all_hpo,
            sym_mode='ave',
            model_name='ICTODQAcross-Ave',
        ),
        'HPOProbMNB': lambda: HPOProbMNBModel(
            hpo_reader=hpo_reader_with_all_hpo,
            phe_list_mode=PHELIST_REDUCE,
            p1=0.65,
            p2=None,
            child_to_parent_prob='sum',
            model_name='HPOProbMNB',
        ),
        'CNB': lambda: CNBModel(
            hpo_reader=hpo_reader_with_all_hpo,
            vec_type=VEC_TYPE_0_1,
            phe_list_mode=PHELIST_ANCESTOR,
            **get_cnb_kwargs(),
        ),
        'NN-Mixup-1': build_mlp_model,
        'MICAModel': lambda: MICAModel(
            hpo_reader=hpo_reader_with_all_hpo,
            model_name='MICAModel',
        ),
        'MICALinModel': lambda: MICALinModel(
            hpo_reader=hpo_reader_with_all_hpo,
            model_name='MICALinModel',
        ),
        'MICAJCModel': lambda: MICAJCModel(
            hpo_reader=hpo_reader_with_all_hpo,
            model_name='MICAJCModel',
        ),
        'MinICModel': lambda: MinICModel(
            hpo_reader=hpo_reader_with_all_hpo,
            model_name='MinICModel',
        ),
        'RBPModel': lambda: RBPModel(
            hpo_reader=hpo_reader_with_all_hpo,
            model_name='RBPModel',
        ),
        'GDDPFisherModel': lambda: GDDPFisherModel(
            hpo_reader=hpo_reader_with_all_hpo,
            model_name='GDDPFisherModel',
        ),
        'BOQAModel': lambda: BOQAModel(
            hpo_reader=hpo_reader_with_all_hpo,
            model_name='BOQAModel',
        ),
    }

    model_list = []
    for candidate in MODEL_CANDIDATES:
        if not _is_candidate_available(candidate):
            continue
        try:
            model_list.append(builders[candidate['name']]())
        except ModuleNotFoundError:
            if candidate['name'] == 'NN-Mixup-1':
                continue
            raise
    return model_list


def _build_outer_ensemble(model_list):
    from core.predict.ensemble import OrderStatisticMultiModel

    return OrderStatisticMultiModel(
        model_list=model_list,
        hpo_reader=model_list[0].hpo_reader,
        model_name='Ensemble',
        keep_raw_score=False,
    )



def build_ensemble_model():
    model_list = build_available_models()
    if not model_list:
        raise RuntimeError('No diagnosis models are available for offline prediction.')
    if len(model_list) == 1:
        return model_list[0]
    return _build_outer_ensemble(model_list)


def get_available_model_names():
    return [model.name for model in run_quietly(build_available_models)]



def describe_available_models():
    names = get_available_model_names()
    if not names:
        return 'Available models: none'
    return 'Available models: ' + ', '.join(names)



def format_results_table(results):
    if not results:
        return 'No results.'

    rows = [(str(i), disease_code, f'{float(score):.6f}') for i, (disease_code, score) in enumerate(results, start=1)]
    headers = ('Rank', 'Disease', 'Score')
    rank_width = max(len(headers[0]), max(len(row[0]) for row in rows))
    disease_width = max(len(headers[1]), max(len(row[1]) for row in rows))
    score_width = max(len(headers[2]), max(len(row[2]) for row in rows))

    lines = [
        f"{headers[0]:<{rank_width}}  {headers[1]:<{disease_width}}  {headers[2]:>{score_width}}"
    ]
    lines.extend(
        f"{rank:<{rank_width}}  {disease:<{disease_width}}  {score:>{score_width}}"
        for rank, disease, score in rows
    )
    return '\n'.join(lines)



def predict_ensemble(hpo_list, topk=10):
    def run_prediction():
        return build_ensemble_model().query(hpo_list, topk)

    return run_quietly(run_prediction)


def resolve_cli_inputs(args):
    hpo_list = parse_hpo_list(args.hpo_list) if args.hpo_list else DEFAULT_SAMPLE_HPO_LIST
    return hpo_list, args.topk



if __name__ == '__main__':
    configure_quiet_warnings()
    args = parse_args()
    hpo_list, topk = resolve_cli_inputs(args)
    print(describe_available_models())
    results = predict_ensemble(hpo_list, topk=topk)
    print(format_results_table(results))
