"""`run --slurm`: the generated job scripts call the CLI the way it parses (SlurmRunner)."""

import shlex
from pathlib import Path

import pytest
from click.testing import CliRunner

from eval3d.cli import main
from eval3d.config import SlurmConfig, load_config
from eval3d.runner.base import Job
from eval3d.runner.slurm import SlurmRunner

CONFIG = Path(__file__).resolve().parents[1] / "config" / "martine.yaml"


def _runner(tmp_path, setup_script=None):
    config = load_config(CONFIG)
    config.execution.slurm = SlurmConfig(account="acc", partition="part", setup_script=setup_script)
    config.paths.output_root = tmp_path
    return SlurmRunner(config)


def _command_line(script: Path) -> str:
    (line,) = [l for l in script.read_text().splitlines() if "eval-pipeline" in l]
    return line


@pytest.mark.parametrize("stage", ["cleanup", "evaluate"])
def test_script_command_is_parsed_by_the_cli(tmp_path, monkeypatch, stage):
    script = _runner(tmp_path)._generate_script(Job("rock", "my_method", stage, str(CONFIG)))
    args = shlex.split(_command_line(script).split("eval-pipeline ", 1)[1])
    assert args[:2] == ["-c", str(CONFIG)] and args[2] == stage

    seen = []
    monkeypatch.setattr("eval3d.pipeline.cleanup.cleanup_mesh", lambda config, obj, method, force: seen.append((obj, method)))
    monkeypatch.setattr("eval3d.pipeline.evaluate.evaluate", lambda config, obj, method, force: seen.append((obj, method)))
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert seen == [("rock", "my_method")]


def test_script_logs_go_to_an_existing_absolute_directory(tmp_path):
    runner = _runner(tmp_path)
    text = runner._generate_script(Job("rock", "my_method", "cleanup", str(CONFIG))).read_text()
    assert runner.logs_dir == (tmp_path / "slurm_logs").resolve() and runner.logs_dir.is_dir()
    assert f"#SBATCH --output={runner.logs_dir}/%x_%j.out" in text
    assert "#SBATCH --job-name=cleanup_rock_my_method" in text


def test_script_sources_the_setup_script_and_the_virtualenv(tmp_path):
    runner = _runner(tmp_path, setup_script="/my dir/setup_env.local.sh")
    runner.venv_path = Path("/venv")
    line = _command_line(runner._generate_script(Job("rock", "my_method", "evaluate", str(CONFIG))))
    assert line.startswith("source '/my dir/setup_env.local.sh' && ")
    assert "source /venv/bin/activate && eval-pipeline -c " in line


def test_script_without_virtualenv_has_no_activation(tmp_path):
    runner = _runner(tmp_path)
    runner.venv_path = None
    line = _command_line(runner._generate_script(Job("rock", "my_method", "cleanup", str(CONFIG))))
    assert "activate" not in line and line.startswith("export PYOPENGL_PLATFORM=osmesa && eval-pipeline -c ")


def test_missing_config_path_is_refused(tmp_path):
    with pytest.raises(ValueError, match="config file"):
        _runner(tmp_path)._generate_script(Job("rock", "my_method", "cleanup", ""))
