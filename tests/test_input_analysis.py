import json

import pytest
import torch

from derivopt.config import ExperimentConfig
from derivopt.data import PDEBenchDataset, write_fixture
from derivopt.input_analysis import evaluate_inputs
from derivopt.io import load_torch, save_torch
from derivopt.preparation import prepare
from derivopt.simulation import BudgetedSimulator


def test_input_analysis_never_calls_predictor(tmp_path, monkeypatch):
    path = write_fixture(tmp_path/"data.h5", "advection", spatial_size=128)
    prepared = prepare(PDEBenchDataset(path, "advection"), ExperimentConfig.small(family="advection"))
    saved = tmp_path/"prepared.pt.gz"
    save_torch(prepared.state_dict(), saved)
    def fail(*args, **kwargs):
        raise AssertionError("input analysis called time-step predictor")
    monkeypatch.setattr(BudgetedSimulator, "step", fail)
    result = evaluate_inputs(saved, path, tmp_path/"analysis", max_states=3)
    assert not result["predictor_used"] and result["state_count"] == 1
    assert result["frame_protocol"] == "trajectory_t0"
    assert all(row["frame"] == 0 for row in result["state_identities"])
    assert set(result["methods"]) == {"primitive", "best_single", "derivbase", "derivopt"}
    payload = load_torch(tmp_path/"analysis"/"input_fields.pt.gz")
    assert payload["target"].shape == (1, 1, 128)
    assert all(torch.isfinite(value).all() for value in payload["decoded"].values())
    assert json.loads((tmp_path/"analysis"/"calibration_curves.json").read_text())["curves"]
    with pytest.raises(FileExistsError):
        evaluate_inputs(saved, path, tmp_path/"analysis")


def test_untrained_latent_is_not_an_input_control(tmp_path):
    with pytest.raises(ValueError, match="explicit-state"):
        evaluate_inputs("unused", "unused", tmp_path, methods=("latent_hyper",))
