"""Explicit file-local splits survive calibration, training and checkpoint reuse."""

import copy
from dataclasses import replace
import json
import shutil

import pytest
import torch

from derivopt.cli import main
from derivopt.config import ExperimentConfig
from derivopt.data import PDEBenchDataset, partition_trajectories, split_trajectories, write_fixture
from derivopt.io import load_torch, save_json, save_torch
from derivopt.input_analysis import evaluate_inputs
from derivopt.preparation import PreparedState, prepare
from derivopt.runner import evaluate_checkpoint, train


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def sources(tmp_path):
    files = [write_fixture(tmp_path / name, "advection", trajectories=2, spatial_size=32)
             for name in ("official-train.h5", "official-test.h5")]
    manifest = {"version": 1,
                "train": [{"file": files[0].name, "indices": [0]}],
                "val": [{"file": files[0].name, "indices": [1]}],
                "test": [{"file": files[1].name, "indices": [0, 1]}]}
    manifest_path = tmp_path / "partitions.json"
    save_json(manifest, manifest_path)
    return files, manifest, manifest_path


def test_relative_manifest_uses_file_local_order_not_combined_order(sources):
    files, _, path = sources
    first = partition_trajectories(PDEBenchDataset(files, "advection"), path)
    reversed_files = partition_trajectories(PDEBenchDataset(files[::-1], "advection"), path)
    for name in ("train", "val", "test"):
        assert first[name].trajectory_ids == reversed_files[name].trajectory_ids
        assert first[name].split == name
    assert first["train"].trajectory_ids == (f"{files[0]}::0",)
    assert first["test"].trajectory_ids == (f"{files[1]}::0", f"{files[1]}::1")


def test_manifest_bundle_is_relocatable(sources, tmp_path):
    files, manifest, path = sources
    moved = tmp_path / "moved"
    moved.mkdir()
    for source in (*files, path):
        shutil.copy2(source, moved / source.name)
    dataset = PDEBenchDataset([moved / file.name for file in files], "advection")
    splits = partition_trajectories(dataset, moved / path.name)
    assert splits["train"].trajectory_ids == (f"{moved / files[0].name}::0",)
    assert json.loads((moved / path.name).read_text()) == manifest


@pytest.mark.parametrize("damage,message", [
    ("overlap", "Overlapping"), ("duplicate", "Overlapping"),
    ("omission", "omits"), ("unknown_file", "Unknown"), ("unknown_index", "Unknown"),
    ("negative", "nonnegative"), ("boolean", "nonnegative"),
    ("empty", "nonempty"), ("extra_key", "exactly"), ("version", "version 1"),
])
def test_manifest_rejects_invalid_partitions(sources, monkeypatch, tmp_path, damage, message):
    files, manifest, _ = sources
    monkeypatch.chdir(tmp_path)
    value = copy.deepcopy(manifest)
    if damage == "overlap":
        value["val"][0]["indices"] = [0]
    elif damage == "duplicate":
        value["train"][0]["indices"] = [0, 0]
    elif damage == "omission":
        value["test"][0]["indices"] = [0]
    elif damage == "unknown_file":
        value["test"][0]["file"] = "missing.h5"
    elif damage == "unknown_index":
        value["test"][0]["indices"] = [0, 9]
    elif damage == "negative":
        value["test"][0]["indices"] = [-1]
    elif damage == "boolean":
        value["test"][0]["indices"] = [True]
    elif damage == "empty":
        value["val"] = []
    elif damage == "extra_key":
        value["validation"] = []
    else:
        value["version"] = 0
    with pytest.raises(ValueError, match=message):
        partition_trajectories(PDEBenchDataset(files, "advection"), value)


def test_file_indices_refer_to_original_file_not_selected_view(sources, monkeypatch, tmp_path):
    files, _, _ = sources
    monkeypatch.chdir(tmp_path)
    dataset = PDEBenchDataset(files, "advection", indices=[3, 1, 0])
    manifest = {"version": 1,
                "train": [{"file": files[0].name, "indices": [0]}],
                "val": [{"file": files[0].name, "indices": [1]}],
                "test": [{"file": files[1].name, "indices": [1]}]}
    assert partition_trajectories(dataset, manifest)["test"].trajectory_ids == (f"{files[1]}::1",)


def test_labeled_view_cannot_be_silently_repartitioned(sources):
    files, _, path = sources
    dataset = PDEBenchDataset(files, "advection").subset([0, 1, 2], split="train")
    config = ExperimentConfig.small(family="advection")
    for operation in (lambda: split_trajectories(dataset), lambda: prepare(dataset, config),
                      lambda: prepare(dataset, config, partitions=path)):
        with pytest.raises(ValueError, match="already labeled"):
            operation()


def test_prepare_train_eval_resume_preserve_manifest(sources, tmp_path):
    files, manifest, path = sources
    config = ExperimentConfig.small(family="advection", train_steps=1, retain_frac=.5)
    dataset = PDEBenchDataset(files, config.family)
    prepared = prepare(dataset, config, partitions=path)
    assert prepared.provenance["partition_source"] == "explicit_manifest"
    assert set(prepared.provenance["calibration_trajectory_ids"]) == {f"{files[0]}::0"}
    expected = prepared.partitions
    prepared_path = tmp_path / "prepared.pt.gz"
    save_torch(prepared.state_dict(), prepared_path)
    assert PreparedState.from_state_dict(load_torch(prepared_path)).partitions == expected
    # A different random-split seed cannot change the explicit partitions.
    other = prepare(dataset, replace(config, split_seed=99), partitions=path)
    assert other.partitions == expected
    run = tmp_path / "train"
    train(files, config, run, partitions=path)
    checkpoint = load_torch(run / "last.pt.gz")
    assert checkpoint["partitions"] == expected
    evaluated = evaluate_checkpoint(run / "best.pt.gz", files[::-1], tmp_path / "eval")
    assert [row["trajectory_id"] for row in evaluated["rows"]] == expected["test"]
    resumed = train(files[::-1], config, tmp_path / "resume", partitions=path, resume_path=run / "last.pt.gz")
    assert resumed["completed_steps"] == 2
    train(files[::-1], config, tmp_path / "reuse", prepared_path=prepared_path, partitions=path)
    inputs = evaluate_inputs(prepared_path, files[::-1], tmp_path / "inputs", methods=("primitive",))
    assert [row["trajectory_id"] for row in inputs["state_identities"]] == expected["test"]
    changed = copy.deepcopy(manifest)
    changed["train"], changed["val"] = changed["val"], changed["train"]
    changed_path = tmp_path / "changed.json"
    save_json(changed, changed_path)
    for kwargs in ({"prepared_path": prepared_path}, {"resume_path": run / "last.pt.gz"}):
        with pytest.raises(ValueError, match="differs from fixed"):
            train(files, config, tmp_path / "rejected", partitions=changed_path, **kwargs)


def test_cli_calibrate_and_train_accept_partition_manifest(sources, tmp_path, capsys):
    files, _, path = sources
    config = ExperimentConfig.small(family="advection", train_steps=1, retain_frac=.5)
    config_path = tmp_path / "config.json"
    save_json(config.to_dict(), config_path)
    prepared_path, run = tmp_path / "prepared.pt.gz", tmp_path / "run"
    common = ["--config", str(config_path), "--data", *(str(file) for file in files), "--partitions", str(path)]
    assert main(["calibrate", *common, "--output", str(prepared_path)]) == 0
    capsys.readouterr()
    assert main(["train", *common, "--prepared", str(prepared_path), "--output", str(run)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["provenance"]["partition_source"] == "explicit_manifest"
    assert load_torch(run / "last.pt.gz")["partitions"]["test"] == [f"{files[1]}::0", f"{files[1]}::1"]
