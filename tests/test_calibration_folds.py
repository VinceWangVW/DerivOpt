import pytest

from derivopt.config import ExperimentConfig
from derivopt.data import PDEBenchDataset, split_trajectories, write_fixture
from derivopt.preparation import prepare


def test_five_folds_calibrate_only_on_allowed_training_trajectories(tmp_path):
    path = write_fixture(tmp_path/"folds.h5", "advection", trajectories=12, steps=3, spatial_size=32)
    dataset = PDEBenchDataset(path, "advection")
    splits = split_trajectories(dataset)
    train_ids = splits["train"].trajectory_ids
    heldout = set(splits["val"].trajectory_ids) | set(splits["test"].trajectory_ids)
    for fold in range(5):
        prepared = prepare(dataset, ExperimentConfig.small(family="advection", calibration_states=64),
                           calibration_fold=fold, folds=5)
        used = set(prepared.provenance["calibration_trajectory_ids"])
        allowed = {value for index, value in enumerate(train_ids) if index % 5 != fold}
        assert used == allowed and not (used & heldout)
        assert prepared.provenance["calibration_fold"] == fold
        assert set(prepared.partitions["train"]) == set(train_ids)


def test_undersized_training_folds_fail(tmp_path):
    path = write_fixture(tmp_path/"small.h5", "advection", spatial_size=32)
    with pytest.raises(ValueError, match="folds"):
        prepare(PDEBenchDataset(path, "advection"), ExperimentConfig.small(family="advection"), calibration_fold=0)
