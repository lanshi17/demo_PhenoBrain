from subprocess import DEVNULL
import subprocess

import pytest

import core.predict.prob_model.boqa_model as boqa_model_module


def test_run_boqa_command_uses_quiet_subprocess(monkeypatch):
    calls = {}

    def fake_run(cmd, stdout=None, stderr=None, check=None):
        calls['cmd'] = cmd
        calls['stdout'] = stdout
        calls['stderr'] = stderr
        calls['check'] = check

    monkeypatch.setattr(boqa_model_module.subprocess, 'run', fake_run)

    boqa_model_module.run_boqa_command(['java', '-jar', 'boqa.jar'])

    assert calls['cmd'] == ['java', '-jar', 'boqa.jar']
    assert calls['stdout'] is DEVNULL
    assert calls['stderr'] is DEVNULL
    assert calls['check'] is True


def test_run_boqa_command_raises_runtime_error_on_nonzero_exit(monkeypatch):
    def fake_run(cmd, stdout=None, stderr=None, check=None):
        raise subprocess.CalledProcessError(returncode=7, cmd=cmd)

    monkeypatch.setattr(boqa_model_module.subprocess, 'run', fake_run)

    with pytest.raises(RuntimeError) as exc_info:
        boqa_model_module.run_boqa_command(['java', '-jar', 'boqa.jar'])

    assert 'boqa.jar' in str(exc_info.value)
    assert '7' in str(exc_info.value)
