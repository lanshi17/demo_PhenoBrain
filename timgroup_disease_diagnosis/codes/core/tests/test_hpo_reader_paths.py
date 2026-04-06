from pathlib import Path

import core.reader.hpo_reader as hpo_reader_module


def test_hpo_reader_falls_back_to_2022_raw_assets(tmp_path, monkeypatch):
    data_path = tmp_path / 'data'
    monkeypatch.setattr(hpo_reader_module, 'DATA_PATH', str(data_path))

    obo_2022 = data_path / 'raw/HPO/2022/Ontology/hp.obo'
    hpoa_2022 = data_path / 'raw/HPO/2022/Annotations/phenotype.hpoa'
    obo_2022.parent.mkdir(parents=True, exist_ok=True)
    hpoa_2022.parent.mkdir(parents=True, exist_ok=True)
    obo_2022.write_text('format-version: 1.2')
    hpoa_2022.write_text('#description')

    reader = hpo_reader_module.HPOReader()

    assert Path(reader.HPO_OBO_PATH) == obo_2022
    assert Path(reader.ANNOTATION_HPOA_PATH) == hpoa_2022
