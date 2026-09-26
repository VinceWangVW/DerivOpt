import io
import json
import time

import pytest
import torch
import h5py

from derivopt.config import ExperimentConfig
from derivopt.data import FAMILY_CHANNELS, FAMILY_NDIM, PDEBenchDataset, split_trajectories, write_fixture
from derivopt.fields import shared_candidates
from derivopt.preparation import PreparedState, prepare
from derivopt.protocol import MAIN_METHODS


def fixture_dataset(tmp_path, family, *, trajectories=6, spatial_size=None):
    size = spatial_size if spatial_size is not None else (128 if FAMILY_NDIM[family] == 1 else 32)
    path = write_fixture(tmp_path / f"{family}.h5", family, trajectories=trajectories, steps=5, spatial_size=size)
    return PDEBenchDataset(path, family)


@pytest.mark.parametrize("family", tuple(FAMILY_CHANNELS))
def test_primary_canonical_preparation_and_all_main_controls(tmp_path, family):
    dataset = fixture_dataset(tmp_path, family)
    started = time.monotonic()
    config = ExperimentConfig.small(family=family)
    state = prepare(dataset, config)
    print(f"\nprepare primary {family}: {time.monotonic()-started:.3f}s, {state.optimum.visited_nodes} nodes", flush=True)
    assert state.optimum.exact
    assert state.geometry.basis.shape == ((128,) if FAMILY_NDIM[family] == 1 else (32, 32))
    groups = {name: set(ids) for name, ids in state.partitions.items()}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
    assert set(state.provenance["calibration_trajectory_ids"]).issubset(groups["train"])
    assert state.provenance["calibration_state_count"] == config.calibration_states
    assert state.calibration.metadata["split"] == "train"
    for method in (*MAIN_METHODS, "derivopt_archmulti"):
        design = state.design_for(method)
        assert design.exact
        assert design.field_bits_per_site <= state.calibration.metadata["budget_per_site"]
        assert torch.isfinite(torch.tensor(design.risk))
    base = state.design_for("derivbase")
    assert [value > 0 for value in base.bits] == [value > 0 for value in state.optimum.bits]
    active = [value for value in base.bits if value]
    assert len(set(active)) == 1
    reopened = state.reopen_splits(dataset)
    assert all(tuple(state.partitions[name]) == split.trajectory_ids for name, split in reopened.items())


def test_cns_full_budget_has_actual_128bit_single_derived_option(tmp_path):
    dataset = fixture_dataset(tmp_path, "compressible_ns")
    config = ExperimentConfig.small(family="compressible_ns", budget_ratio=1.0)
    started = time.monotonic()
    state = prepare(dataset, config)
    print(f"\nprepare primary compressible_ns R1: {time.monotonic()-started:.3f}s, {state.optimum.visited_nodes} nodes", flush=True)
    assert state.calibration.metadata["budget_per_site"] == 128
    assert max(state.calibration.precisions[0]) == 32
    assert max(state.calibration.precisions[1]) == 128
    single = state.design_for("best_single")
    assert single.bits == (0, 128, 0, 0)
    assert single.field_bits_per_site == 128
    residual = state.geometry.to_residual(dataset[0].states[:1])
    decoded = state.calibration.reconstruct(residual, single.bits)
    assert decoded.shape == residual.shape and torch.isfinite(decoded).all()


def test_prepared_checkpoint_roundtrip_and_train_only_statistics(tmp_path):
    dataset = fixture_dataset(tmp_path, "diffusion_sorption")
    state = prepare(dataset, ExperimentConfig.small(family="diffusion_sorption"))
    saved = io.BytesIO()
    torch.save(state.state_dict(), saved)
    saved.seek(0)
    restored = PreparedState.from_state_dict(torch.load(saved, weights_only=True))
    assert restored.optimum.bits == state.optimum.bits
    assert restored.partitions == state.partitions
    assert restored.geometry.config == state.geometry.config
    residual = state.geometry.to_residual(dataset[0].states[:1])
    torch.testing.assert_close(restored.calibration.reconstruct(residual, restored.optimum.bits),
                               state.calibration.reconstruct(residual, state.optimum.bits))
    # Independently derive predictor normalization from the recorded TRAIN
    # calibration frame identities; it must not depend on validation/test.
    samples = {sample.metadata["trajectory_id"]: sample for sample in state.reopen_splits(dataset)["train"]}
    counters = {}
    frames = []
    for identity in state.provenance["calibration_trajectory_ids"]:
        position = counters.get(identity, 0)
        frames.append(samples[identity].states[position])
        counters[identity] = position + 1
    coarse = state.geometry.resampler.coarsen(state.geometry.to_residual(torch.stack(frames)))
    axes = (0, *range(2, coarse.ndim))
    torch.testing.assert_close(state.mean, coarse.mean(axes))
    torch.testing.assert_close(state.std, coarse.std(axes, correction=0).clamp_min(1e-6))


@pytest.mark.parametrize("family,count", [("advection", 9), ("burgers", 9), ("diffusion_sorption", 9),
                                          ("diffusion_reaction", 20), ("radial_dam_break", 9),
                                          ("incompressible_ns", 22), ("compressible_ns", 42)])
def test_shared_candidate_counts_preserve_full_library(tmp_path, family, count):
    from derivopt.geometry import Geometry
    dataset = fixture_dataset(tmp_path, family)
    geometry = Geometry.from_sample(dataset[0], 0.125)
    vectors = ((0, 1),) if family == "incompressible_ns" else (((1, 2),) if family == "compressible_ns" else ())
    candidates = shared_candidates(geometry.basis, FAMILY_CHANNELS[family],
                                   retained_cutoff=geometry.resampler.retained_cutoff, vector_groups=vectors)
    assert len(candidates) == count
    assert sum(candidate.primitive for candidate in candidates) == len(FAMILY_CHANNELS[family])
    assert all(candidate.n_components == 1 for candidate in candidates)


def test_mixed_boundary_parameters_are_rejected_not_silently_reused(tmp_path):
    dataset = fixture_dataset(tmp_path, "diffusion_sorption")
    second_training = split_trajectories(dataset, seed=0)["train"][1]
    with h5py.File(dataset.paths[0], "r+") as handle:
        group = handle[str(second_training.metadata["trajectory"])]
        source_config = json.loads(group.attrs["config"])
        source_config["sim"]["sol"] = 2.0
        group.attrs["config"] = json.dumps(source_config)
    with pytest.raises(ValueError, match="boundary parameters"):
        prepare(dataset, ExperimentConfig.small(family="diffusion_sorption"))


@pytest.mark.parametrize("family", list(FAMILY_CHANNELS))
def test_shared_canonical_exact_preparation_with_resource_guard(tmp_path, family):
    # Keep the canonical budget/retain fractions and all 42 CNS candidates.
    # Use a rectangular CNS fixture to bound the cost of exact search.
    dataset = fixture_dataset(tmp_path, family, spatial_size=(32, 16) if family == "compressible_ns" else None)
    # Including the radial Nyquist boundary adds observations and
    # increases this complete CNS library's exact-certification workload.
    max_nodes = 2000000 if family == "compressible_ns" else 20000
    state = prepare(dataset, ExperimentConfig.small(family=family, library="shared", selector_max_nodes=max_nodes))
    print(f"\nprepare shared {family}: {state.optimum.visited_nodes} nodes", flush=True)
    assert state.optimum.exact
    assert state.optimum.visited_nodes <= max_nodes
    assert len(state.calibration.candidates) == (42 if family == "compressible_ns" else (22 if family == "incompressible_ns" else (20 if family == "diffusion_reaction" else 9)))
    assert state.geometry.resampler.expressible_mask.sum() > 1  # Include non-DC modes.
    assert state.design_for("primitive").field_bits_per_site == state.calibration.metadata["budget_per_site"]
    for method in (*MAIN_METHODS, "derivopt_archmulti"):
        design = state.design_for(method)
        assert design.exact
        assert design.field_bits_per_site <= state.calibration.metadata["budget_per_site"]
