"""Small end-to-end runs, source identity, safe resume and output protection."""

from dataclasses import replace
import json
import math

import h5py
import pytest
import torch

from derivopt.config import ExperimentConfig
from derivopt.data import PDEBenchDataset, split_trajectories, write_fixture
from derivopt.io import dataset_identity, load_torch, save_json, save_torch
from derivopt.preparation import PreparedState, prepare
from derivopt.protocol import MetricProtocol
from derivopt.runner import _loss, build_simulator, environment, evaluate_checkpoint, evaluate_simulator, reevaluate, train
from derivopt.simulation import BudgetedSimulator


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def source(tmp_path):
    return write_fixture(tmp_path / "advection.h5", "advection", spatial_size=32, trajectories=4, steps=5)


def small_config(**kwargs):
    return ExperimentConfig.small(family="advection", retain_frac=0.5, **kwargs)


def test_prepare_train_eval_reevaluate_and_exact_best_predictions(source, tmp_path):
    config = small_config()
    dataset = PDEBenchDataset(source, config.family)
    prepared = prepare(dataset, config)
    prepared_path = tmp_path / "prepared.pt.gz"
    save_torch(prepared.state_dict(), prepared_path)
    loaded = PreparedState.from_state_dict(load_torch(prepared_path))
    assert loaded.provenance["dataset_identity"] == dataset_identity(dataset)
    run_dir = tmp_path / "run"
    result = train(source, config, run_dir, prepared_path=prepared_path)
    assert result["completed_steps"] == 2
    assert result["predictor_parameter_updated"] and result["nonzero_finite_gradient"]
    assert result["evaluation_checkpoint"] == "best"
    assert math.isfinite(result["evaluation"]["summary"]["nrmse"])
    assert all(row["supervision_steps"] == 1 for row in result["history"])
    for name in ("best.pt.gz", "last.pt.gz", "results.json", "predictions.pt.gz"):
        assert (run_dir / name).is_file()
    best = load_torch(run_dir / "best.pt.gz")
    last = load_torch(run_dir / "last.pt.gz")
    assert type(best["environment"]["torch"]) is str
    assert best["checkpoint_role"] == "best" and last["checkpoint_role"] == "last"
    assert result["evaluation_completed_steps"] == best["completed_steps"]
    evaluate_checkpoint(run_dir / "best.pt.gz", source, tmp_path / "eval", max_trajectories=1)
    original = load_torch(run_dir / "predictions.pt.gz")
    repeated = load_torch(tmp_path / "eval" / "predictions.pt.gz")
    for left, right in zip(original["predictions"], repeated["predictions"]):
        torch.testing.assert_close(left["predicted"], right["predicted"], rtol=0, atol=0)
        assert left["payload_bits"] == right["payload_bits"]
    revised = reevaluate(run_dir / "predictions.pt.gz", tmp_path / "reeval", MetricProtocol(tau_q=2.0, tau_out=0.2))
    for old, new in zip(result["evaluation"]["rows"], revised["rows"]):
        torch.testing.assert_close(old["timesteps"]["fine_rel"], new["timesteps"]["fine_rel"], equal_nan=True, rtol=0, atol=0)
    assert (tmp_path / "reeval" / "evaluation.json").is_file()


def test_resume_matches_continuous_updates_without_recalibration(source, tmp_path, monkeypatch):
    config = small_config(train_steps=2)
    first_dir, resumed_dir, continuous_dir = (tmp_path / name for name in ("first", "resumed", "continuous"))
    train(source, config, first_dir)
    continuous = train(source, replace(config, train_steps=3), continuous_dir)

    def forbidden_recalibration(*args, **kwargs):
        raise AssertionError("resume must not recalibrate")

    monkeypatch.setattr("derivopt.runner.prepare", forbidden_recalibration)
    resumed = train(source, replace(config, train_steps=1), resumed_dir, resume_path=first_dir / "last.pt.gz")
    assert resumed["completed_steps"] == continuous["completed_steps"] == 3
    assert [row["step"] for row in resumed["history"]] == [1, 2, 3]
    assert resumed["executed_steps"] == 1
    continued_last = load_torch(continuous_dir / "last.pt.gz")
    resumed_last = load_torch(resumed_dir / "last.pt.gz")
    for key, left in continued_last["simulator"]["module_state"].items():
        right = resumed_last["simulator"]["module_state"][key]
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        else:
            assert left == right
    assert resumed["best_validation_nrmse"] == continuous["best_validation_nrmse"]
    original_predictions = load_torch(continuous_dir / "predictions.pt.gz")["predictions"]
    resumed_predictions = load_torch(resumed_dir / "predictions.pt.gz")["predictions"]
    torch.testing.assert_close(original_predictions[0]["predicted"], resumed_predictions[0]["predicted"], rtol=0, atol=0)


def test_rolloutmulti_really_trains_three_feedback_steps(source, tmp_path):
    config = small_config(method="rolloutmulti", rollout_supervision=3)
    result = train(source, config, tmp_path / "rolloutmulti")
    assert all(row["supervision_steps"] == 3 for row in result["history"])
    assert all(len(row["payload_bits"]) == 3 * config.batch_size for row in result["history"])
    assert result["predictor_parameter_updated"]


def test_latent_training_uses_real_records_and_saves_decodable_model(source, tmp_path):
    config = small_config(method="latent_hyper")
    result = train(source, config, tmp_path / "latent")
    assert result["selected_design"]["kind"] == "learned_hyperprior"
    assert result["selected_design"]["explicit_fields"] is False
    assert all(bits == 16 * 8 for row in result["history"] for bits in row["payload_bits"])
    checkpoint = load_torch(tmp_path / "latent" / "best.pt.gz")
    model = BudgetedSimulator.from_checkpoint_state(checkpoint["simulator"])
    sample = PDEBenchDataset(source, "advection")[0].states[:1]
    _, info = model.step(sample, actual_codec=True)
    assert info["ledgers"][0].hyper_bits > 0 and info["ledgers"][0].primary_bits > 0


def test_explicit_rollout_loss_excludes_codec_diagnostic(source):
    config = small_config(method="rolloutmulti", latent_rate_weight=999.0, latent_reconstruction_weight=999.0)
    prepared = prepare(PDEBenchDataset(source, "advection"), config)
    model = build_simulator(prepared, config)
    samples = PDEBenchDataset(source, "advection")[0].states
    initial, targets = samples[:1], samples[1:4].unsqueeze(0)
    observed, _ = _loss(model, initial, targets, config)
    current, terms = initial, []
    model.reset_state()
    scale = model.primitive_std.reshape(1, 1, 1)
    for target in targets.unbind(1):
        current, _ = model.step(current)
        terms.append(((current - target) / scale).square().mean())
    torch.testing.assert_close(observed, torch.stack(terms).mean(), rtol=0, atol=0)


@pytest.mark.parametrize("change", [{"batch_size": 2}, {"learning_rate": 0.002}, {"rollout_supervision": 2}, {"boundary_override": "periodic"}])
def test_resume_rejects_changed_training_configuration(source, tmp_path, change):
    config = small_config(train_steps=1)
    train(source, config, tmp_path / "original")
    with pytest.raises(ValueError, match="resume configuration changed"):
        train(source, replace(config, **change), tmp_path / "resumed", resume_path=tmp_path / "original" / "last.pt.gz")


def test_modified_source_rejected_for_prepared_resume_and_evaluation(source, tmp_path):
    config = small_config(train_steps=1)
    prepared = prepare(PDEBenchDataset(source, "advection"), config)
    train(source, config, tmp_path / "original", prepared=prepared)
    with h5py.File(source, "r+") as handle:
        handle["tensor"][0, 0, 0] += 0.001
    with pytest.raises(ValueError, match="Source dataset identity changed"):
        train(source, config, tmp_path / "bad-prepared", prepared=prepared)
    with pytest.raises(ValueError, match="Source dataset identity changed"):
        train(source, config, tmp_path / "bad-resume", resume_path=tmp_path / "original" / "last.pt.gz")
    with pytest.raises(ValueError, match="Source dataset identity changed"):
        evaluate_checkpoint(tmp_path / "original" / "best.pt.gz", source, tmp_path / "bad-eval")


def test_existing_outputs_are_not_overwritten(source, tmp_path):
    config = small_config(train_steps=1)
    output = tmp_path / "existing"
    output.mkdir()
    protected = output / "user.json"
    save_json({"keep": True}, protected)
    before = protected.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        train(source, config, output)
    assert protected.read_bytes() == before
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_json({"keep": False}, protected)
    torch_path = tmp_path / "fixed.pt.gz"
    save_torch({"value": torch.tensor(1)}, torch_path)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_torch({"value": torch.tensor(2)}, torch_path)
    assert load_torch(torch_path)["value"].item() == 1


def test_environment_is_weights_only_loadable(tmp_path):
    path = tmp_path / "environment.pt.gz"
    save_torch(environment(), path)
    assert load_torch(path)["torch"] == str(torch.__version__)


@pytest.mark.parametrize("parameter", ["D", "sol"])
def test_noncalibrated_training_trajectory_cannot_change_robin_parameters(tmp_path, parameter):
    source = write_fixture(tmp_path / "sorption.h5", "diffusion_sorption", trajectories=6, spatial_size=16)
    original = PDEBenchDataset(source, "diffusion_sorption")
    training = split_trajectories(original)["train"]
    altered_id = training[-1].metadata["trajectory_id"]
    altered_group = training[-1].metadata["trajectory"]
    with h5py.File(source, "r+") as handle:
        settings = json.loads(handle[altered_group].attrs["config"])
        settings["sim"][parameter] *= 2.0
        handle[altered_group].attrs["config"] = json.dumps(settings)
    config = ExperimentConfig.small(
        family="diffusion_sorption", retain_frac=0.5, calibration_states=2,
        train_steps=1, batch_size=32,
    )
    prepared = prepare(PDEBenchDataset(source, config.family), config)
    assert altered_id not in prepared.provenance["calibration_trajectory_ids"]
    with pytest.raises(ValueError, match="boundary parameters/lifting"):
        train(source, config, tmp_path / "bad-robin", prepared=prepared)


@pytest.mark.parametrize("split", ["val", "test"])
@pytest.mark.parametrize("change", ["shifted_domain", "nonuniform"])
def test_eval_checks_each_trajectory_coordinate_grid(tmp_path, split, change):
    source = write_fixture(tmp_path / "reaction.h5", "diffusion_reaction", spatial_size=8)
    original = PDEBenchDataset(source, "diffusion_reaction")
    altered_group = split_trajectories(original)[split][0].metadata["trajectory"]
    with h5py.File(source, "r+") as handle:
        values = handle[f"{altered_group}/grid/x"][:]
        if change == "shifted_domain":
            values += 0.25
        else:
            values[3] += 0.01
        handle[f"{altered_group}/grid/x"][:] = values
    dataset = PDEBenchDataset(source, "diffusion_reaction")
    config = ExperimentConfig.small(family="diffusion_reaction", retain_frac=0.5)
    prepared = prepare(dataset, config)
    model = build_simulator(prepared, config)
    with pytest.raises(ValueError, match="physical coordinates/domain"):
        evaluate_simulator(model, prepared.reopen_splits(dataset)[split], horizon=1)
