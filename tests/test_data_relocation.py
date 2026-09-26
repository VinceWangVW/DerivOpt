"""Explicit copy relocation preserves fixed preparation and strict default checks."""
import copy
from dataclasses import replace
import json
import os
import shutil

import pytest
import torch

from derivopt.cli import main
from derivopt.config import ExperimentConfig
from derivopt.data import PDEBenchDataset, write_fixture
from derivopt.io import dataset_identity, load_torch, save_json, save_torch, validate_dataset_identity
from derivopt.preparation import prepare
from derivopt.runner import _validate_relocated_geometry, evaluate_checkpoint, train


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def copied(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    stamp = source.stat().st_mtime_ns + 1_000_000_000
    os.utime(destination, ns=(stamp, stamp))
    return destination


def test_copy_mapping_evaluate_prepared_train_and_resume(tmp_path, monkeypatch, capsys):
    source = write_fixture(tmp_path / "old" / "advection.h5", "advection", spatial_size=32)
    config = ExperimentConfig.small(family="advection", retain_frac=.5, train_steps=1)
    prepared = prepare(PDEBenchDataset(source, config.family), config)
    original_provenance, original_partitions = copy.deepcopy(prepared.provenance), copy.deepcopy(prepared.partitions)
    prepared_path = tmp_path / "prepared.pt.gz"
    save_torch(prepared.state_dict(), prepared_path)
    train(source, config, tmp_path / "first", prepared=prepared)
    train(source, replace(config, train_steps=2), tmp_path / "continuous", prepared=prepared)
    checkpoint = tmp_path / "first" / "best.pt.gz"
    checkpoint_bytes = checkpoint.read_bytes()
    baseline = load_torch(tmp_path / "first" / "predictions.pt.gz")
    destination = copied(source, tmp_path / "new" / "renamed.h5")
    mapping = {str(source): str(destination)}
    map_path = tmp_path / "data-map.json"
    save_json(mapping, map_path)
    # Migration does not require access to the original source location.
    source.rename(tmp_path / "old" / "original-not-at-saved-path.h5")
    with pytest.raises(ValueError, match="Source dataset identity changed"):
        evaluate_checkpoint(checkpoint, destination, tmp_path / "strict-rejection")

    assert main(["evaluate", "--checkpoint", str(checkpoint), "--data", str(destination),
                 "--data-map", str(map_path), "--output", str(tmp_path / "mapped-eval")]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["data_relocation"]["files"] == mapping
    assert printed["data_relocation"]["content_verified"] is False
    mapped = load_torch(tmp_path / "mapped-eval" / "predictions.pt.gz")
    for old, new in zip(baseline["predictions"], mapped["predictions"]):
        torch.testing.assert_close(old["predicted"], new["predicted"], rtol=0, atol=0)
        assert new["trajectory_id"] == old["trajectory_id"].replace(str(source), str(destination), 1)
    assert checkpoint.read_bytes() == checkpoint_bytes

    def forbidden(*args, **kwargs):
        raise AssertionError("A relocated prepared/resume run must not recalibrate or repartition")
    monkeypatch.setattr("derivopt.runner.prepare", forbidden)
    reused = train(destination, config, tmp_path / "reused", prepared=prepared, data_map=mapping)
    assert prepared.provenance == original_provenance and prepared.partitions == original_partitions
    assert reused["provenance"]["data_relocations"][0]["source_identity"] == original_provenance["dataset_identity"]
    config_path = tmp_path / "config.json"
    save_json(config.to_dict(), config_path)
    assert main(["train", "--config", str(config_path), "--data", str(destination),
                 "--resume", str(tmp_path / "first" / "last.pt.gz"), "--data-map", str(map_path),
                 "--output", str(tmp_path / "resumed")]) == 0
    capsys.readouterr()
    resumed = load_torch(tmp_path / "resumed" / "last.pt.gz")
    continuous = load_torch(tmp_path / "continuous" / "last.pt.gz")
    for name, value in continuous["simulator"]["module_state"].items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, resumed["simulator"]["module_state"][name], rtol=0, atol=0)
    assert resumed["partitions"] == resumed["prepared"]["partitions"]
    assert resumed["dataset_identity"] == resumed["prepared"]["provenance"]["dataset_identity"]
    assert resumed["dataset_identity"] == dataset_identity(PDEBenchDataset(destination, config.family))
    # Once saved at the new location, strict evaluation/resume needs no mapping.
    evaluate_checkpoint(tmp_path / "resumed" / "best.pt.gz", destination, tmp_path / "strict-new-eval")
    again = train(destination, config, tmp_path / "strict-new-resume", resume_path=tmp_path / "resumed" / "last.pt.gz")
    assert again["completed_steps"] == 3
    from_path = train(destination, config, tmp_path / "prepared-path", prepared_path=prepared_path, data_map=map_path)
    assert from_path["completed_steps"] == 1
    assert load_torch(prepared_path)["partitions"] == original_partitions


def test_partial_mapping_keeps_unmapped_files_strict_and_order_independent(tmp_path):
    a = write_fixture(tmp_path / "a.h5", "advection", spatial_size=16)
    b = write_fixture(tmp_path / "b.h5", "advection", spatial_size=16)
    expected = dataset_identity(PDEBenchDataset([a, b], "advection"))
    moved = copied(a, tmp_path / "copy" / "a.h5")
    current = dataset_identity(PDEBenchDataset([b, moved], "advection"))
    mapping = {str(a): str(moved)}
    assert validate_dataset_identity(expected, current, data_map=mapping) == mapping
    assert expected["files"][0]["path"] == str(a)
    timestamp = b.stat().st_mtime_ns + 1_000_000_000
    os.utime(b, ns=(timestamp, timestamp))
    with pytest.raises(ValueError, match="Source dataset identity changed"):
        validate_dataset_identity(expected, dataset_identity(PDEBenchDataset([moved, b], "advection")), data_map=mapping)
    with pytest.raises(ValueError, match="one-to-one"):
        validate_dataset_identity(expected, current, data_map={str(a): str(moved), str(b): str(moved)})


@pytest.mark.parametrize("field", ["size_bytes", "trajectory_ids", "family", "time_stride"])
def test_mapping_does_not_waive_nonlocation_identity(tmp_path, field):
    source = write_fixture(tmp_path / "source.h5", "advection", spatial_size=16)
    expected = dataset_identity(PDEBenchDataset(source, "advection"))
    destination = copied(source, tmp_path / "copy.h5")
    current = dataset_identity(PDEBenchDataset(destination, "advection"))
    if field == "size_bytes":
        current["files"][0][field] += 1
    elif field == "trajectory_ids":
        current[field][0] += "other"
    elif field == "family":
        current[field] = "burgers"
    else:
        current[field] = 2
    with pytest.raises(ValueError, match="Source dataset identity changed"):
        validate_dataset_identity(expected, current, data_map={str(source): str(destination)})


def test_mapping_requires_explicit_absolute_known_changed_paths(tmp_path):
    source = write_fixture(tmp_path / "source.h5", "advection", spatial_size=16)
    identity = dataset_identity(PDEBenchDataset(source, "advection"))
    for mapping in ({}, [], {"source.h5": str(source)}, {str(source): "relative.h5"},
                    {str(tmp_path / "unknown.h5"): str(source)}, {str(source): str(source)}):
        with pytest.raises(ValueError, match="data_map"):
            validate_dataset_identity(identity, identity, data_map=mapping)
    with pytest.raises(ValueError, match="existing prepared"):
        train(source, ExperimentConfig.small(family="advection"), tmp_path / "new", data_map={str(source): str(source)})


def test_mapped_schema_uses_all_indexed_shapes_without_reading_states(tmp_path, monkeypatch):
    source = write_fixture(tmp_path / "source.h5", "advection", spatial_size=32)
    prepared = prepare(PDEBenchDataset(source, "advection"), ExperimentConfig.small(family="advection"))
    different = write_fixture(tmp_path / "wrong-shape.h5", "advection", spatial_size=16)
    dataset = PDEBenchDataset(different, "advection")
    def forbidden(*args, **kwargs):
        raise AssertionError("Relocation schema checks must be metadata-only")
    monkeypatch.setattr(PDEBenchDataset, "read_window", forbidden)
    with pytest.raises(ValueError, match="schema/shape"):
        _validate_relocated_geometry(dataset, prepared.geometry)
