"""Fixed physical time steps are checked at input and every simulation boundary."""

from dataclasses import replace

import h5py
import numpy as np
import pytest
import torch

from derivopt.config import ExperimentConfig
from derivopt.data import PDEBenchDataset, fixed_time_step, write_fixture
from derivopt.geometry import Geometry
from derivopt.io import dataset_identity, load_torch, save_json, save_torch
from derivopt.preparation import PreparedState, prepare
from derivopt.runner import build_simulator, evaluate_checkpoint, evaluate_simulator, train


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fixed_time_step_accepts_native_roundoff_and_rejects_nonuniform(dtype):
    times = np.linspace(0, 1, 21, dtype=dtype)
    assert fixed_time_step(times) == pytest.approx(.05)
    assert fixed_time_step(times.astype(np.float64), source_dtype=str(times.dtype)) == pytest.approx(.05)
    times[7] += .001
    with pytest.raises(ValueError, match="Nonuniform time"):
        fixed_time_step(times)


@pytest.mark.parametrize("times", [[0.], [0., 0.], [0., float("nan")], [0., float("inf")]])
def test_fixed_time_step_requires_two_finite_increasing_times(times):
    with pytest.raises(ValueError, match="time|Time"):
        fixed_time_step(torch.tensor(times))


def test_loader_rejects_nonuniform_raw_times_before_stride(tmp_path):
    source = write_fixture(tmp_path / "data.h5", "advection", spatial_size=16)
    with h5py.File(source, "r+") as handle:
        handle["t-coordinate"][1] += .01
    with pytest.raises(ValueError, match="Nonuniform time"):
        PDEBenchDataset(source, "advection", time_stride=2)[0]


def test_geometry_preserves_dt_and_checks_every_sample_and_old_checkpoint(tmp_path):
    source = write_fixture(tmp_path / "data.h5", "advection", spatial_size=16)
    sample = PDEBenchDataset(source, "advection")[0]
    assert sample.metadata["time_coordinate_dtype"] == "float32"
    geometry = Geometry.from_sample(sample, .5)
    assert geometry.config["time_step"] == pytest.approx(.1)
    geometry.validate_sample(replace(sample, times=sample.times + 3.))
    rounded_offset = (sample.times.to(torch.float32) + 100).to(torch.float64)
    geometry.validate_sample(replace(sample, times=rounded_offset))
    with pytest.raises(ValueError, match="time step differs"):
        geometry.validate_sample(replace(sample, times=sample.times * 2))
    restored = Geometry.from_state_dict(geometry.state_dict())
    assert restored.config["time_step"] == geometry.config["time_step"]
    restored.validate_sample(sample)
    legacy = geometry.state_dict()
    legacy["format_version"] = 2
    with pytest.raises(ValueError, match="regenerate preparation"):
        Geometry.from_state_dict(legacy)
    missing = geometry.state_dict()
    missing["config"] = dict(missing["config"])
    del missing["config"]["time_step"]
    with pytest.raises(ValueError, match="fixed time step"):
        Geometry.from_state_dict(missing)


def test_float32_large_origin_does_not_hide_changed_time_step(tmp_path):
    source = write_fixture(tmp_path / "data.h5", "advection", spatial_size=16)
    sample = PDEBenchDataset(source, "advection")[0]
    def axis(step):
        return torch.from_numpy((1000 + np.arange(5) * step).astype(np.float32)).to(torch.float64)
    original = replace(sample, times=axis(.1))
    geometry = Geometry.from_sample(original, .5)
    geometry.validate_sample(replace(sample, times=axis(.1)))
    with pytest.raises(ValueError, match="time step differs"):
        geometry.validate_sample(replace(sample, times=axis(.1004)))
    assert geometry.config["time_step_tolerance"] == pytest.approx(np.spacing(np.float32(1000)) / 4)
    assert type(geometry.config["time_step_tolerance"]) is float
    saved = tmp_path / "large-origin.pt.gz"
    save_torch(geometry.state_dict(), saved)
    restored = Geometry.from_state_dict(load_torch(saved))
    restored.validate_sample(original)
    with pytest.raises(ValueError, match="time step differs"):
        restored.validate_sample(replace(sample, times=axis(.1004)))


def grouped_case(tmp_path, *, changed_index=None):
    source = write_fixture(tmp_path / "reaction.h5", "diffusion_reaction", spatial_size=8)
    if changed_index is not None:
        with h5py.File(source, "r+") as handle:
            times = handle[f"{changed_index:04d}/grid/t"]
            times[...] = times[...] * 2
    manifest = {"version": 1, "train": [{"file": source.name, "indices": [0, 1]}],
                "val": [{"file": source.name, "indices": [2]}],
                "test": [{"file": source.name, "indices": [3]}]}
    path = tmp_path / "partitions.json"
    save_json(manifest, path)
    config = ExperimentConfig.small(family="diffusion_reaction", retain_frac=.5,
                                    calibration_states=2, train_steps=1, batch_size=16)
    return source, path, config


def test_calibration_rejects_a_different_time_step(tmp_path):
    source, path, config = grouped_case(tmp_path, changed_index=1)
    with pytest.raises(ValueError, match="time step differs"):
        prepare(PDEBenchDataset(source, config.family), replace(config, calibration_states=6), partitions=path)


@pytest.mark.parametrize("index", [1, 2, 3])
def test_training_checks_noncalibrated_train_validation_and_test_time_steps(tmp_path, index):
    source, path, config = grouped_case(tmp_path, changed_index=index)
    dataset = PDEBenchDataset(source, config.family)
    prepared = prepare(dataset, config, partitions=path)
    assert all(identity.endswith("::0000") for identity in prepared.provenance["calibration_trajectory_ids"])
    with pytest.raises(ValueError, match="time step differs"):
        train(source, config, tmp_path / "run", prepared=prepared)


@pytest.mark.parametrize("split,index", [("val", 2), ("test", 3)])
def test_checkpoint_evaluator_checks_prepared_time_step(tmp_path, split, index):
    source, path, config = grouped_case(tmp_path, changed_index=index)
    dataset = PDEBenchDataset(source, config.family)
    prepared = prepare(dataset, config, partitions=path)
    simulator = build_simulator(prepared, config)
    checkpoint = tmp_path / "model.pt.gz"
    save_torch({"simulator": simulator.checkpoint_state(), "experiment_config": config.to_dict(),
                "dataset_identity": dataset_identity(dataset), "partitions": prepared.partitions,
                "completed_steps": 0}, checkpoint)
    with pytest.raises(ValueError, match="time step differs"):
        evaluate_checkpoint(checkpoint, source, tmp_path / "evaluate", split=split)


def test_fixed_time_step_saved_in_preparation_and_training_checkpoint(tmp_path):
    source, path, config = grouped_case(tmp_path)
    dataset = PDEBenchDataset(source, config.family)
    prepared = prepare(dataset, config, partitions=path)
    assert prepared.provenance["time_step"] == pytest.approx(.1)
    legacy = prepared.state_dict()
    legacy["format_version"] = 1
    with pytest.raises(ValueError, match="regenerate preparation"):
        PreparedState.from_state_dict(legacy)
    run = tmp_path / "run"
    train(source, config, run, prepared=prepared)
    state = load_torch(run / "best.pt.gz")
    assert state["prepared"]["geometry"]["config"]["time_step"] == pytest.approx(.1)
    assert state["simulator"]["geometry"]["config"]["time_step"] == pytest.approx(.1)
    result = evaluate_checkpoint(run / "best.pt.gz", source, tmp_path / "eval")
    assert result["summary"]["trajectory_count"] == 1
