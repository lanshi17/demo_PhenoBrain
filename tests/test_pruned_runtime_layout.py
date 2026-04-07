from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "timgroup_disease_diagnosis"


def test_pruned_runtime_keeps_only_supported_top_level_extras():
    assert not (DIAG / "Docker").exists()
    assert not (DIAG / "PhenoBrain_Web_API").exists()
    assert not (DIAG / "codes" / "bert_syn_project").exists()
    assert not (DIAG / "example_result").exists()


def test_offline_runtime_entrypoint_still_exists():
    assert (
        DIAG / "codes" / "core" / "core" / "script" / "example_predict_ensemble.py"
    ).exists()


def test_old_research_entrypoints_are_removed():
    script_root = DIAG / "codes" / "core" / "core" / "script"
    assert not (script_root / "test" / "test_optimal_model.py").exists()
    assert not (script_root / "train").exists()
