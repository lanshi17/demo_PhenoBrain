import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'convert_ga4gh_benchmark.py'


def load_module():
    spec = importlib.util.spec_from_file_location('convert_ga4gh_benchmark', str(SCRIPT_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def test_convert_ga4gh_to_dataset_normalizes_mim_and_preserves_omim(tmp_path):
    module = load_module()
    questions_path = tmp_path / 'questions.json'
    answers_path = tmp_path / 'answers.json'
    output_path = tmp_path / 'converted.json'
    write_json(
        questions_path,
        [
            {
                'patient_id': 'p1',
                'hpo_terms': [
                    {'hpo_id': 'HP:0000118', 'hpo_name': 'Phenotypic abnormality'},
                    {'hpo_id': 'HP:0001250', 'hpo_name': 'Seizure'},
                ],
            },
            {
                'patient_id': 'p2',
                'hpo_terms': [{'hpo_id': 'HP:0004322', 'hpo_name': 'Short stature'}],
            },
        ],
    )
    write_json(
        answers_path,
        [
            {'patient_id': 'p1', 'answers': [{'omim_id': 'MIM:123456', 'disease_name': 'A'}]},
            {'patient_id': 'p2', 'answers': [{'omim_id': 'OMIM:654321', 'disease_name': 'B'}]},
        ],
    )

    dataset = module.convert_ga4gh_to_dataset(questions_path, answers_path, output_path)

    assert dataset == [
        [['HP:0000118', 'HP:0001250'], ['OMIM:123456']],
        [['HP:0004322'], ['OMIM:654321']],
    ]
    assert json.loads(output_path.read_text(encoding='utf-8')) == dataset


def test_convert_ga4gh_to_dataset_skips_missing_or_empty_answers(tmp_path):
    module = load_module()
    questions_path = tmp_path / 'questions.json'
    answers_path = tmp_path / 'answers.json'
    output_path = tmp_path / 'converted.json'
    write_json(
        questions_path,
        [
            {'patient_id': 'p1', 'hpo_terms': [{'hpo_id': 'HP:0000118'}]},
            {'patient_id': 'p2', 'hpo_terms': [{'hpo_id': 'HP:0001250'}]},
            {'patient_id': 'p3', 'hpo_terms': [{'hpo_id': 'HP:0004322'}]},
        ],
    )
    write_json(
        answers_path,
        [
            {'patient_id': 'p1', 'answers': [{'omim_id': 'MIM:123456'}]},
            {'patient_id': 'p2', 'answers': []},
        ],
    )

    dataset = module.convert_ga4gh_to_dataset(questions_path, answers_path, output_path)

    assert dataset == [[['HP:0000118'], ['OMIM:123456']]]


def test_checked_in_ga4gh_full_dataset_matches_source_files():
    root = Path(__file__).resolve().parents[1]
    questions = json.loads(
        (root / 'data' / 'GA4GH' / 'GA4GH.benchmark_patients.questions.json').read_text(encoding='utf-8')
    )
    answers = json.loads(
        (root / 'data' / 'GA4GH' / 'GA4GH.benchmark_patients.answers.json').read_text(encoding='utf-8')
    )
    dataset = json.loads(
        (root / 'data' / 'inputs' / 'GA4GH.benchmark_patients.json').read_text(encoding='utf-8')
    )

    assert len(questions) == 384
    assert len(answers) == 384
    assert len(dataset) == 384
    assert all(hpo_list for hpo_list, _ in dataset)
    assert all(dis_list for _, dis_list in dataset)
    assert all(dis_code.startswith('OMIM:') for _, dis_list in dataset for dis_code in dis_list)
