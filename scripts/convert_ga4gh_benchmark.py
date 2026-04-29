#!/usr/bin/env python3
"""Convert GA4GH benchmark questions/answers to the standard PhenoBrain dataset format."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GA4GH_DIR = PROJECT_ROOT / 'data' / 'GA4GH'
QUESTIONS_PATH = GA4GH_DIR / 'GA4GH.benchmark_patients.questions.json'
ANSWERS_PATH = GA4GH_DIR / 'GA4GH.benchmark_patients.answers.json'
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'inputs' / 'GA4GH.benchmark_patients.json'


def load_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def normalize_omim_id(omim_id):
    if omim_id.startswith('MIM:'):
        return 'OMIM:' + omim_id[4:]
    return omim_id


def convert_ga4gh_to_dataset(questions_path=QUESTIONS_PATH, answers_path=ANSWERS_PATH, output_path=OUTPUT_PATH):
    questions = {question['patient_id']: question for question in load_json(questions_path)}
    answers = {answer['patient_id']: answer for answer in load_json(answers_path)}
    dataset = []

    for patient_id, question in questions.items():
        answer = answers.get(patient_id)
        if answer is None:
            print(f'Warning: No answer for patient {patient_id}, skipping')
            continue

        hpo_list = [hpo['hpo_id'] for hpo in question['hpo_terms']]
        dis_list = [normalize_omim_id(item['omim_id']) for item in answer['answers']]
        if not dis_list:
            print(f'Warning: No disease answers for patient {patient_id}, skipping')
            continue

        dataset.append([hpo_list, dis_list])

    print(f'Conversion complete: {len(dataset)} patients out of {len(questions)} total')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(dataset, handle, indent=2)
    print(f'Saved to {output_path}')
    return dataset


if __name__ == '__main__':
    convert_ga4gh_to_dataset()
