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
    (line,) = [text for text in script.read_text().splitlines() if "eval-pipeline" in text]
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


def _variant_runner(tmp_path):
    config = load_config(CONFIG)
    config.execution.slurm = SlurmConfig(account="acc", partition="part")
    config.paths.output_root = tmp_path
    config._config_path = CONFIG
    config.apply_views(str(tmp_path / "sets" / "{object}" / "front3.sfm"), "front3")
    return SlurmRunner(config)


def test_views_script_passes_the_view_set_before_the_subcommand(tmp_path):
    runner = _variant_runner(tmp_path)
    script = runner._generate_script(Job("rock", "my_method", "evaluate", str(CONFIG)))
    args = shlex.split(_command_line(script).split("eval-pipeline ", 1)[1])
    assert args[:7] == ["-c", str(CONFIG), "--views", str(tmp_path / "sets" / "{object}" / "front3.sfm"),
                        "--variant", "front3", "evaluate"]
    assert script.parent == tmp_path / "slurm_scripts@front3" and runner.logs_dir == (tmp_path / "slurm_logs@front3").resolve()
    assert "#SBATCH --job-name=evaluate@front3_rock_my_method" in script.read_text()


def test_views_gt_is_a_job_the_evaluations_wait_for(tmp_path, monkeypatch):
    runner = _variant_runner(tmp_path)
    submitted = []

    def fake_sbatch(cmd, **kwargs):
        submitted.append(cmd)
        return type("R", (), {"stdout": f"{100 + len(submitted)}\n"})()

    monkeypatch.setattr("eval3d.runner.slurm.subprocess.run", fake_sbatch)
    monkeypatch.setattr(runner, "wait", lambda job_ids: {j: True for j in job_ids})
    results = runner.run_pipeline(runner.config, objects=["rock"], methods=["a", "b"])

    scripts = [Path(cmd[-1]).name for cmd in submitted]
    assert scripts == ["cleanup_rock_a.sh", "cleanup_rock_b.sh", "variant_gt_rock.sh", "evaluate_rock_a.sh", "evaluate_rock_b.sh"]
    assert "--dependency=afterok:101:103" in submitted[3] and "--dependency=afterok:102:103" in submitted[4]
    assert set(results) == {"rock/a/cleanup", "rock/b/cleanup", "rock/variant_gt", "rock/a/evaluate", "rock/b/evaluate"}
    gt_line = _command_line(tmp_path / "slurm_scripts@front3" / "variant_gt_rock.sh")
    assert gt_line.endswith("--variant front3 preprocess-gt --object rock")


def test_default_mode_submits_no_gt_job(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    runner.config._config_path = CONFIG
    submitted = []
    monkeypatch.setattr("eval3d.runner.slurm.subprocess.run",
                        lambda cmd, **kwargs: submitted.append(cmd) or type("R", (), {"stdout": f"{len(submitted)}\n"})())
    monkeypatch.setattr(runner, "wait", lambda job_ids: {j: True for j in job_ids})
    runner.run_pipeline(runner.config, objects=["rock"], methods=["a"])
    assert [Path(cmd[-1]).name for cmd in submitted] == ["cleanup_rock_a.sh", "evaluate_rock_a.sh"]


def test_run_slurm_views_derives_no_gt_in_the_submitting_process(tmp_path, monkeypatch):
    def no_gt(*args, **kwargs):
        raise AssertionError("the GT of the view set must be a job with --slurm")

    monkeypatch.setattr("eval3d.pipeline.variant_gt.derive_variant_gt", no_gt)
    monkeypatch.setattr(SlurmRunner, "__init__", lambda self, config: None)
    monkeypatch.setattr(SlurmRunner, "run_pipeline", lambda self, config, objects=None, methods=None, stages=None: {})
    result = CliRunner().invoke(main, ["-c", str(CONFIG), "run", "--slurm", "-o", "rock", "-m", "my_method",
                                       "--views", str(tmp_path / "front3.sfm")])
    assert result.exit_code == 0, result.output
