"""Candidate meanings, retained-band information and Small Expert workflows."""

import io
import json
import math

import h5py
import pytest
import torch

from derivopt.calibration import CalibratedState, calibrate
from derivopt.config import ExperimentConfig
from derivopt.data import FAMILY_CHANNELS, FAMILY_NDIM, PDEBenchDataset, write_fixture
from derivopt.fields import FieldCandidate, primary_candidates, shared_candidates, small_expert_candidates
from derivopt.geometry import Geometry
from derivopt.io import load_torch, save_json
from derivopt.operators import SpectralBasis, SpectralResampler
from derivopt.preparation import prepare
from derivopt.runner import evaluate_checkpoint, train


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


COUNTS = dict(zip(FAMILY_CHANNELS, (5, 5, 4, 8, 4, 6, 9)))
PRIMARY_COUNTS = dict(zip(FAMILY_CHANNELS, (3, 3, 3, 7, 3, 3, 7)))


@pytest.mark.parametrize("family", FAMILY_CHANNELS)
def test_small_expert_has_all_41_scalar_channels_and_preserves_primary(tmp_path, family):
    path = write_fixture(tmp_path / "data.h5", family, spatial_size=16)
    geometry = Geometry.from_sample(PDEBenchDataset(path, family).read_window(0, 0, 2), .5)
    primary = primary_candidates(family, geometry.basis, FAMILY_CHANNELS[family])
    small = small_expert_candidates(family, geometry.basis, FAMILY_CHANNELS[family], known_lift=geometry.lifting)
    assert sum(field.n_components for field in primary) == PRIMARY_COUNTS[family]
    assert len(small) == COUNTS[family]
    assert all(field.n_components == 1 for field in small)
    assert sum(field.primitive for field in small) == len(FAMILY_CHANNELS[family])
    assert len({field.name for field in small}) == len(small)
    assert sum(COUNTS.values()) == 41


def test_burgers_p4_and_dam_break_p3_survive_canonical_coarsening():
    periodic = SpectralBasis((128,), (2 * math.pi,))
    coarse = SpectralResampler(periodic, (16,))
    field = next(item for item in shared_candidates(periodic, ("u",), retained_cutoff=coarse.retained_cutoff)
                 if item.name == "band_4_u")
    x = torch.arange(128, dtype=torch.float64) * 2 * math.pi / 128
    state = torch.cos(6 * x)[None, None]
    torch.testing.assert_close(field.evaluate(state, periodic), state, atol=1e-13, rtol=1e-13)
    assert float(coarse.coarsen(field.evaluate(state, periodic)).norm()) > 1
    neumann = SpectralBasis((32, 32), (1., 1.), "neumann")
    coarse = SpectralResampler(neumann, (4, 4))
    field = next(item for item in shared_candidates(neumann, ("h",), retained_cutoff=coarse.retained_cutoff)
                 if item.name == "band_3_h")
    coefficients = torch.zeros(1, 1, 32, 32, dtype=torch.float64)
    coefficients[0, 0, 2, 1] = 8
    state = neumann.synthesis(coefficients)
    torch.testing.assert_close(field.evaluate(state, neumann), state, atol=1e-13, rtol=1e-13)
    assert float(coarse.coarsen(field.evaluate(state, neumann)).norm()) > .1


def test_shared_bands_partition_retained_radius_and_require_explicit_cutoff():
    basis = SpectralBasis((64,), (2 * math.pi,))
    coarse = SpectralResampler(basis, (16,))
    fields = shared_candidates(basis, ("u",), retained_cutoff=coarse.retained_cutoff)
    masks = torch.stack([field.matrix[:, 0, 0].real.bool() for field in fields if field.name.startswith("band_")])
    expected = basis.eigenvalues.sqrt().flatten() <= coarse.retained_cutoff
    assert torch.equal(masks.sum(0), expected.long())
    assert masks[3, 8]  # Final P4 endpoint includes frequency / cutoff == 1.
    with pytest.raises(TypeError, match="retained_cutoff"):
        shared_candidates(basis, ("u",))
    for cutoff in (0, -1, float("nan")):
        with pytest.raises(ValueError, match="retained_cutoff"):
            shared_candidates(basis, ("u",), retained_cutoff=cutoff)


def test_burgers_flux_is_nonlinear_and_scaling_survives_checkpoint():
    basis = SpectralBasis((64,), (2 * math.pi,))
    x = torch.arange(64, dtype=torch.float64) * 2 * math.pi / 64
    state = (.3 + torch.sin(x))[None, None].requires_grad_()
    fields = {field.name: field for field in small_expert_candidates("burgers", basis)}
    torch.testing.assert_close(fields["flux"].evaluate(state, basis), state.square() / 2)
    torch.testing.assert_close(fields["flux_derivative"].evaluate(state, basis), state * torch.cos(x), atol=2e-14, rtol=2e-14)
    scaled = fields["flux"].scaled(torch.tensor([2.], dtype=torch.float64))
    scaled = scaled.with_effective_matrix(torch.ones_like(scaled.matrix))
    # The fitted response is not substituted for the nonlinear physical field.
    torch.testing.assert_close(scaled.evaluate(state, basis), state.square() / 4)
    scaled.evaluate(state, basis).sum().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    buffer = io.BytesIO()
    torch.save(scaled.state_dict(), buffer)
    buffer.seek(0)
    restored = FieldCandidate.from_state_dict(torch.load(buffer, weights_only=True))
    assert restored.kind == "flux"
    torch.testing.assert_close(restored.evaluate(state, basis), scaled.evaluate(state, basis), rtol=0, atol=0)
    with pytest.raises(ValueError, match="not a diagonal"):
        restored.apply_coefficients(basis.analysis(state))


def test_sorption_raw_gradient_includes_known_lift():
    basis = SpectralBasis((32,), (1.,), "robin", robin_coefficients=(math.inf, 2.))
    x = (torch.arange(32, dtype=torch.float64) + .5) / 32
    residual = x.square()[None, None].requires_grad_()
    lift = (2 - x)[None, None]
    gradient = next(item for item in small_expert_candidates("diffusion_sorption", basis, known_lift=lift)
                    if item.name == "raw_gradient")
    actual = gradient.evaluate(residual, basis)
    torch.testing.assert_close(actual, (2*x - 1)[None, None], rtol=0, atol=1e-13)
    actual.square().sum().backward()
    assert residual.grad is not None and torch.isfinite(residual.grad).all()
    restored = FieldCandidate.from_state_dict(gradient.state_dict())
    torch.testing.assert_close(restored.evaluate(residual, basis), actual)


def test_dam_break_radial_gradient_uses_domain_centre():
    basis = SpectralBasis((17, 19), (2., 3.), "neumann")
    x, y = torch.meshgrid((torch.arange(17, dtype=torch.float64)+.5)*2/17 - 1,
                         (torch.arange(19, dtype=torch.float64)+.5)*3/19 - 1.5, indexing="ij")
    state = (x.square()+y.square())[None, None].requires_grad_()
    candidate = next(item for item in small_expert_candidates("radial_dam_break", basis)
                     if item.name == "radial_gradient")
    result = candidate.evaluate(state, basis)
    torch.testing.assert_close(result, (2*(x.square()+y.square()).sqrt())[None, None], atol=2e-14, rtol=2e-14)
    result.sum().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()


def test_nonperiodic_vector_derivatives_do_not_wrap_and_poisson_is_boundary_adapted():
    basis = SpectralBasis((16, 18), (2., 3.), "robin", robin_coefficients=((math.inf, math.inf),)*2)
    x, y = torch.meshgrid((torch.arange(16, dtype=torch.float64)+.5)*2/16,
                         (torch.arange(18, dtype=torch.float64)+.5)*3/18, indexing="ij")
    state = torch.stack((x.square()+y, x*y+2*y.square()))[None]
    fields = {field.name: field for field in small_expert_candidates("incompressible_ns", basis)}
    torch.testing.assert_close(fields["vorticity"].evaluate(state, basis), (y-1)[None, None], atol=1e-13, rtol=1e-13)
    psi = fields["streamfunction"].evaluate(state, basis)
    torch.testing.assert_close(basis.spectral_power(psi, 1), (y-1)[None, None], atol=1e-12, rtol=1e-12)
    shared = {field.name: field for field in shared_candidates(basis, ("vx", "vy"),
              retained_cutoff=SpectralResampler(basis, (8, 9)).retained_cutoff, vector_groups=((0, 1),))}
    torch.testing.assert_close(shared["divergence_0"].evaluate(state, basis), (3*x+4*y)[None, None], atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(shared["curl_0_0"].evaluate(state, basis), (y-1)[None, None], atol=1e-13, rtol=1e-13)


def test_empirical_flux_calibration_keeps_real_evaluator_after_roundtrip():
    torch.manual_seed(18)
    basis = SpectralBasis((32,), (2 * math.pi,))
    resampler = SpectralResampler(basis, (16,))
    training = torch.randn(12, 1, 32, dtype=torch.float64) + 1
    raw = small_expert_candidates("burgers", basis)[-2:]
    model = calibrate(training, basis, resampler, raw, (2, 4), shellwise=False)
    assert all(candidate.requires_empirical_response for candidate in model.candidates)
    assert all(candidate.matrix.abs().sum() > 0 for candidate in model.candidates)
    for candidate, original, scale in zip(model.candidates, raw, model.rms_scales):
        torch.testing.assert_close(candidate.evaluate(training, basis), original.evaluate(training, basis)/scale.reshape(1, 1, 1))
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = CalibratedState.from_state_dict(torch.load(buffer, weights_only=True))
    assert [field.kind for field in restored.candidates] == ["flux", "flux_derivative"]
    torch.testing.assert_close(restored.reconstruct(training[:2], (2, 4)), model.reconstruct(training[:2], (2, 4)), rtol=0, atol=0)


@pytest.mark.parametrize("family", FAMILY_CHANNELS)
def test_small_expert_preparation_training_and_evaluation(tmp_path, family):
    size = 128 if FAMILY_NDIM[family] == 1 else ((32, 16) if family == "compressible_ns" else 32)
    source = write_fixture(tmp_path / "data.h5", family, spatial_size=size)
    config = ExperimentConfig.small(family=family, library="small_expert", train_steps=2, selector_max_nodes=20000)
    result = train(source, config, tmp_path / "run")
    assert result["predictor_parameter_updated"] and result["nonzero_finite_gradient"]
    assert result["provenance"]["candidate_scalar_channels"] == COUNTS[family]
    checkpoint = load_torch(tmp_path / "run" / "last.pt.gz")
    assert checkpoint["prepared"]["config"]["library"] == "small_expert"
    evaluated = evaluate_checkpoint(tmp_path / "run" / "best.pt.gz", source, tmp_path / "evaluation")
    assert math.isfinite(evaluated["summary"]["nrmse"])


def test_small_expert_shared_comparison_keeps_nonlibrary_settings_matched(tmp_path):
    source = write_fixture(tmp_path / "burgers.h5", "burgers", spatial_size=128)
    saved = {}
    configurations = []
    for library in ("small_expert", "shared"):
        config = ExperimentConfig.small(family="burgers", library=library, seed=17, train_steps=2,
                                        selector_max_nodes=20000)
        config_path = tmp_path / f"{library}.json"
        save_json(config.to_dict(), config_path)
        result = train(source, ExperimentConfig.from_file(config_path), tmp_path / library)
        assert result["predictor_parameter_updated"]
        saved[library] = load_torch(tmp_path / library / "last.pt.gz")
        configurations.append({key: value for key, value in result["config"].items() if key != "library"})
    assert configurations[0] == configurations[1]
    small, shared = saved["small_expert"], saved["shared"]
    assert small["partitions"] == shared["partitions"]
    assert small["provenance"]["calibration_trajectory_ids"] == shared["provenance"]["calibration_trajectory_ids"]
    assert small["provenance"]["candidate_scalar_channels"] == 5
    assert shared["provenance"]["candidate_scalar_channels"] == 9
    torch.testing.assert_close(small["prepared"]["mean"], shared["prepared"]["mean"], rtol=0, atol=0)
    torch.testing.assert_close(small["prepared"]["std"], shared["prepared"]["std"], rtol=0, atol=0)


@pytest.mark.parametrize("library", ["primary", "small_expert", "shared"])
def test_nonperiodic_ins_complete_library_workflow(tmp_path, library):
    # The rectangular 8x4 case keeps three nonconstant Dirichlet modes and all
    # 22 shared channels at R=.25 with a bounded exact-search workload.
    source = write_fixture(tmp_path / "ins.h5", "incompressible_ns", spatial_size=(8, 4))
    with h5py.File(source, "r+") as handle:
        settings = json.loads(handle.attrs["config"])
        settings["velocity_extrapolation"] = "ZERO"
        handle.attrs["config"] = json.dumps(settings)
    config = ExperimentConfig.small(family="incompressible_ns", library=library, retain_frac=.5,
                                    train_steps=1, selector_max_nodes=200000)
    prepared = prepare(PDEBenchDataset(source, config.family), config)
    assert not prepared.geometry.basis.is_periodic
    assert int(prepared.geometry.resampler.expressible_mask.sum()) == 3
    assert prepared.calibration.metadata["divergence_free"] is False
    assert any(field.requires_empirical_response for field in prepared.calibration.candidates)
    result = train(source, config, tmp_path / "run", prepared=prepared)
    assert result["predictor_parameter_updated"] and math.isfinite(result["evaluation"]["summary"]["nrmse"])


def test_prepare_reads_bounded_windows_not_full_trajectories(tmp_path, monkeypatch):
    source = write_fixture(tmp_path / "data.h5", "advection", trajectories=4, steps=20, spatial_size=32)
    dataset = PDEBenchDataset(source, "advection")
    original = PDEBenchDataset.read_window
    windows = []
    def bounded(self, index, start=0, stop=None):
        windows.append((start, stop))
        return original(self, index, start, stop)
    def forbidden(self, index):
        raise AssertionError("Preparation must not request an entire trajectory")
    monkeypatch.setattr(PDEBenchDataset, "read_window", bounded)
    monkeypatch.setattr(PDEBenchDataset, "__getitem__", forbidden)
    prepared = prepare(dataset, ExperimentConfig.small(family="advection", calibration_states=3))
    assert prepared.provenance["calibration_state_count"] == 3
    assert windows == [(0, 2), (0, 3)]
