import json
from pathlib import Path
import subprocess
import sys

import pytest

from derivopt.cli import main
from derivopt.config import ExperimentConfig
from derivopt.data import PDEBenchDataset, write_fixture
from derivopt.preparation import prepare


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_fixture_and_inspection_commands(tmp_path, capsys, monkeypatch):
    path = tmp_path / "advection.h5"
    assert main(["fixture", "--family", "advection", "--output", str(path),
                 "--trajectories", "5", "--steps", "6", "--size", "32"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["manufactured_fixture"] is True
    assert output["trajectory_shape"] == [6, 1, 32]
    def forbidden(self, index):
        raise AssertionError("Inspection must use a bounded window, not a full trajectory")
    monkeypatch.setattr(PDEBenchDataset, "__getitem__", forbidden)
    assert main(["inspect-data", "--family", "advection", "--data", str(path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["trajectory_count"] == 5
    assert output["shape"] == [2, 1, 32]
    assert output["trajectory_length"] == 6
    assert output["sampled_frame_count"] == 2
    assert output["statistics_scope"] == "sampled_frames"
    assert output["times"]["count"] == 2
    assert output["metadata"]["time_window"]["stop"] == 2
    assert output["channels"] == ["u"]
    assert output["metadata"]["file_attributes"]["derivopt_fixture"]


def test_non_square_fixture_and_multi_file_inspection(tmp_path, capsys):
    paths = [tmp_path / "rd_a.h5", tmp_path / "rd_b.h5"]
    for path in paths:
        main(["fixture", "--family", "diffusion_reaction", "--output", str(path), "--size", "8", "12"])
    capsys.readouterr()
    main(["inspect-data", "--family", "diffusion_reaction", "--data", *(str(path) for path in paths), "--index", "5"])
    result = json.loads(capsys.readouterr().out)
    assert result["trajectory_count"] == 8
    assert result["shape"] == [2, 2, 8, 12]
    assert result["trajectory_length"] == 5
    assert result["sampled_frame_count"] == 2


@pytest.mark.parametrize("frames", [2, 4, 6])
def test_inspection_requested_frames_are_bounded(tmp_path, capsys, frames):
    path = write_fixture(tmp_path / "data.h5", "advection", steps=6, spatial_size=16)
    main(["inspect-data", "--family", "advection", "--data", str(path), "--frames", str(frames)])
    result = json.loads(capsys.readouterr().out)
    assert result["trajectory_length"] == 6
    assert result["sampled_frame_count"] == frames
    assert result["shape"] == [frames, 1, 16]
    assert result["times"]["count"] == frames
    assert result["metadata"]["time_window"]["stop"] == frames


@pytest.mark.parametrize("frames", [0, 1, 7])
def test_inspection_invalid_frames_fail_before_state_read(tmp_path, capsys, monkeypatch, frames):
    path = write_fixture(tmp_path / "data.h5", "advection", steps=6, spatial_size=16)
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid frame count must fail before reading states")
    monkeypatch.setattr(PDEBenchDataset, "read_window", forbidden)
    with pytest.raises(SystemExit) as error:
        main(["inspect-data", "--family", "advection", "--data", str(path), "--frames", str(frames)])
    assert error.value.code == 2
    assert "--frames must be between 2 and the trajectory length (6)" in capsys.readouterr().err


def test_inspection_native_ins_reads_only_two_frames(tmp_path, capsys, monkeypatch):
    path = write_fixture(tmp_path / "ins.h5", "incompressible_ns", steps=6, spatial_size=(8, 12))
    read_window = PDEBenchDataset.read_window
    windows = []
    def bounded(self, index, start=0, stop=None):
        windows.append((index, start, stop))
        assert (index, start, stop) == (1, 0, 2)
        return read_window(self, index, start, stop)
    monkeypatch.setattr(PDEBenchDataset, "read_window", bounded)
    main(["inspect-data", "--family", "incompressible_ns", "--data", str(path), "--index", "1"])
    result = json.loads(capsys.readouterr().out)
    assert windows == [(1, 0, 2)]
    assert result["shape"] == [2, 2, 8, 12]
    assert result["trajectory_length"] == 6
    assert result["sampled_frame_count"] == 2


def test_main_matrix_count_and_manifest(tmp_path, capsys):
    path = tmp_path / "main_matrix.json"
    main(["matrix", "--output", str(path), "--smoke"])
    output = json.loads(capsys.readouterr().out)
    assert output["count"] == 1764
    manifest = json.loads(path.read_text())
    assert len(manifest["configurations"]) == 1764
    for config in manifest["configurations"]:
        assert ExperimentConfig(**config).validate().smoke
    unique = {(row["family"], row["backbone"], row["budget_ratio"], row["retain_frac"], row["method"])
              for row in manifest["configurations"]}
    assert len(unique) == 1764
    original = path.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(["matrix", "--output", str(path)])
    assert error.value.code == 2
    assert path.read_bytes() == original


def test_matrix_multiple_seeds_keep_identical_data_partitions(tmp_path, capsys):
    path = tmp_path / "paired.json"
    main(["matrix", "--smoke", "--seeds", "1", "3", "--output", str(path)])
    summary = json.loads(capsys.readouterr().out)
    assert summary["count"] == 3528 and summary["seed_count"] == 2
    manifest = json.loads(path.read_text())
    assert {row["seed"] for row in manifest["configurations"]} == {1, 3}
    assert {row["split_seed"] for row in manifest["configurations"]} == {0}
    with pytest.raises(SystemExit) as error:
        main(["matrix", "--seeds", "1", "1"])
    assert error.value.code == 2


def test_all_example_configs_validate_and_small_preparation_runs(tmp_path):
    for path in CONFIGS.glob("*.json"):
        ExperimentConfig.from_file(path)
    config = ExperimentConfig.from_file(CONFIGS / "smoke_advection_fno.json")
    path = write_fixture(tmp_path / "advection.h5", config.family, spatial_size=32)
    prepared = prepare(PDEBenchDataset(path, config.family), config)
    assert prepared.optimum.exact
    assert prepared.provenance["calibration_split"] == "train"
    assert prepared.geometry.resampler.coarse_shape == (4,)
    for method in ("primitive", "best_single", "derivbase", "derivopt"):
        assert prepared.design_for(method).field_bits_per_site <= 8


def test_yaml_config_and_unknown_keys_fail_clearly(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("family: advection\nbackbone: fno\nmethod: primitive\ntrain_steps: 2\n")
    config = ExperimentConfig.from_file(path)
    assert config.family == "advection" and config.train_steps == 2
    path.write_text("unknown_option: 1\n")
    with pytest.raises(TypeError):
        ExperimentConfig.from_file(path)


def test_calibrate_command_saves_reusable_preparation(tmp_path, capsys):
    from derivopt.io import load_torch
    from derivopt.preparation import PreparedState
    data = write_fixture(tmp_path / "advection.h5", "advection", spatial_size=32)
    output = tmp_path / "prepared.pt.gz"
    assert main(["calibrate", "--config", str(CONFIGS / "smoke_advection_fno.json"),
                 "--data", str(data), "--output", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["prepared"] == str(output)
    restored = PreparedState.from_state_dict(load_torch(output))
    assert restored.optimum.exact and restored.provenance["calibration_split"] == "train"
    input_output = tmp_path / "input-analysis"
    main(["inputs", "--prepared", str(output), "--data", str(data), "--output", str(input_output),
          "--max-states", "1", "--methods", "primitive", "derivopt"])
    inputs = json.loads(capsys.readouterr().out)
    assert inputs["predictor_used"] is False
    assert inputs["state_count"] == 1 and inputs["frame_protocol"] == "trajectory_t0"
    assert set(inputs["methods"]) == {"primitive", "derivopt"}
    assert all((input_output / name).is_file() for name in
               ("input_metrics.json", "input_fields.pt.gz", "calibration_curves.json"))


def test_calibrate_fold_command_excludes_only_training_trajectories(tmp_path, capsys):
    from derivopt.io import load_torch
    from derivopt.preparation import PreparedState
    data = write_fixture(tmp_path / "advection.h5", "advection", trajectories=16, spatial_size=32)
    output = tmp_path / "fold.pt.gz"
    main(["calibrate", "--config", str(CONFIGS / "smoke_advection_fno.json"), "--data", str(data),
          "--output", str(output), "--fold", "2", "--folds", "5"])
    capsys.readouterr()
    restored = PreparedState.from_state_dict(load_torch(output))
    assert restored.provenance["calibration_fold"] == 2
    assert restored.provenance["calibration_folds"] == 5
    used = set(restored.provenance["calibration_trajectory_ids"])
    excluded = {index for i, index in enumerate(restored.partitions["train"]) if i % 5 == 2}
    assert used and not used.intersection(excluded)
    assert not used.intersection(restored.partitions["val"] + restored.partitions["test"])


def test_predictor_free_inputs_command_forwards_exact_interface(tmp_path, monkeypatch, capsys):
    import types
    module = types.ModuleType("derivopt.input_analysis")
    received = {}

    def evaluate_inputs(prepared, data, output, **kwargs):
        received.update(prepared=prepared, data=data, output=output, **kwargs)
        return {"status": "completed"}

    module.evaluate_inputs = evaluate_inputs
    monkeypatch.setitem(sys.modules, "derivopt.input_analysis", module)
    main(["inputs", "--prepared", str(tmp_path / "prepared.pt.gz"), "--data", str(tmp_path / "data.h5"),
          "--output", str(tmp_path / "inputs"), "--max-states", "3", "--methods", "primitive", "derivopt"])
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert received["methods"] == ("primitive", "derivopt")
    assert received["split"] == "test" and received["max_states"] == 3
    assert received["prepared"] == tmp_path / "prepared.pt.gz"


def test_invalid_cli_input_and_existing_fixture_fail_cleanly(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["fixture", "--family", "advection", "--output", str(tmp_path / "bad.h5"), "--size", "8", "8"])
    assert error.value.code == 2
    assert "--size" in capsys.readouterr().err
    path = write_fixture(tmp_path / "existing.h5", "advection")
    with pytest.raises(SystemExit) as error:
        main(["fixture", "--family", "advection", "--output", str(path)])
    assert error.value.code == 2


def test_python_module_entry_point_outside_project(tmp_path):
    result = subprocess.run([sys.executable, "-m", "derivopt", "--help"], cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    assert "inspect-data" in result.stdout and "reevaluate" in result.stdout
