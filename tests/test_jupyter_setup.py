from pathlib import Path
import tomllib
import types


ROOT = Path(__file__).resolve().parents[1]


def load_jupyter_server_config():
    namespace = {}

    def get_config():
        return types.SimpleNamespace(
            ServerApp=types.SimpleNamespace(),
            AiExtension=types.SimpleNamespace(),
        )

    namespace["get_config"] = get_config
    exec((ROOT / ".jupyter" / "jupyter_server_config.py").read_text(), namespace)
    return namespace["c"]


def test_jupyter_ai_installs_provider_backends():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    enhanced = data["project"]["optional-dependencies"]["enhanced"]

    assert "jupyter-ai[all]>=2,<3" in enhanced


def test_repo_launcher_uses_repo_local_jupyter_config():
    script = (ROOT / "scripts" / "start_jupyter_lab.sh").read_text()

    assert 'JUPYTER_CONFIG_DIR="${ROOT}/.jupyter"' in script
    assert 'exec uv run jupyter lab "$@"' in script


def test_jupyter_server_config_enables_jupyterfs_metamanager():
    config = load_jupyter_server_config()

    assert (
        config.ServerApp.contents_manager_class
        == "jupyterfs.metamanager.MetaManager"
    )
