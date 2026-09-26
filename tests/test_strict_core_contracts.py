"""Independent channel, basis and discrete-code checks."""
import math

import pytest
import torch

from derivopt.calibration import calibrate
from derivopt.fields import primary_candidates
from derivopt.geometry import Geometry
from derivopt.operators import SpectralBasis, SpectralResampler
from derivopt.quantization import ScalarQuantizer
from derivopt.selectors import DesignProblem
from derivopt.simulation import BudgetedSimulator


@pytest.mark.parametrize("other_boundary", ["periodic", "robin"])
def test_calibration_rejects_same_shape_different_boundary_operator(other_boundary):
    basis = SpectralBasis((8,), (2.0,), "neumann")
    kwargs = {"robin_coefficients": (1.0, 2.0)} if other_boundary == "robin" else {}
    other = SpectralBasis((8,), (2.0,), other_boundary, **kwargs)
    resampler = SpectralResampler(other, (4,))
    training = torch.randn(4, 1, 8, dtype=torch.float64)
    with pytest.raises(ValueError, match="boundaries"):
        calibrate(training, basis, resampler, primary_candidates("advection", basis), (2,))


def test_calibration_rejects_changed_robin_coefficients():
    basis = SpectralBasis((8,), (2.0,), "robin", robin_coefficients=(1.0, 2.0))
    other = SpectralBasis((8,), (2.0,), "robin", robin_coefficients=(2.0, 2.0))
    training = torch.randn(4, 1, 8, dtype=torch.float64)
    with pytest.raises(ValueError, match="robin_coefficients"):
        calibrate(training, basis, SpectralResampler(other, (4,)),
                  primary_candidates("diffusion_sorption", basis), (2,))


@pytest.mark.parametrize("bits", [1, 8, 32, 33, 128])
def test_decoder_rejects_noninteger_quantization_codes(bits):
    quantizer = ScalarQuantizer(torch.zeros(1), torch.ones(1), torch.ones(1))
    with pytest.raises(ValueError, match="invalid scalar quantizer code"):
        quantizer.from_codes(torch.tensor([[[0.5]]]), bits)


def test_physical_target_risk_is_invariant_to_nonorthogonal_latent_coordinates():
    # This includes the non-isometric scaling in a Helmholtz potential map;
    # retaining its target map is essential, not just rotating the prior.
    torch.manual_seed(405)
    modes, channels = 5, 4
    root = torch.randn(modes, channels, channels, dtype=torch.complex128)
    prior = root @ root.mH + torch.eye(channels)[None]
    response = torch.randn(modes, channels, channels, dtype=torch.complex128)
    precision = response.mH @ response
    rotation, _ = torch.linalg.qr(torch.randn(modes, channels, channels, dtype=torch.complex128))
    scales = torch.tensor([1.0, 2.0, 4.0, 0.5], dtype=torch.float64)
    target = rotation * scales[None, None, :]
    inverse = torch.linalg.inv(target)
    physical_prior = target @ prior @ target.mH
    physical_precision = inverse.mH @ precision @ inverse
    latent = DesignProblem(prior, [{3: precision}], [1], ["field"], target=target)
    physical = DesignProblem(physical_prior, [{3: physical_precision}], [1], ["field"])
    expected = target @ torch.linalg.inv(torch.linalg.inv(prior) + precision) @ target.mH
    torch.testing.assert_close(physical.posterior((3,)), expected, atol=2e-12, rtol=2e-12)
    assert latent.risk((3,)) == pytest.approx(physical.risk((3,)), rel=2e-12)
    assert physical.risk((3,)) == pytest.approx(float(expected.diagonal(dim1=-2, dim2=-1).sum().real), rel=2e-12)


def test_ns_rank_one_prior_matches_scalar_closed_form():
    wave = torch.tensor([[1.0, 2.0], [3.0, 1.0]], dtype=torch.float64)
    transverse = torch.stack((-wave[:, 1], wave[:, 0]), dim=-1)
    transverse = transverse / wave.square().sum(-1).sqrt()[:, None]
    projector = transverse[:, :, None] * transverse[:, None, :]
    spectrum = torch.tensor([2.0, 3.0], dtype=torch.float64)
    primitive_precision = torch.eye(2, dtype=torch.float64)[None].expand(2, -1, -1) * 4
    curl = torch.stack((-wave[:, 1], wave[:, 0]), dim=-1)[:, None, :]
    vortical_precision = (curl.mT @ curl) / 0.5
    problem = DesignProblem(spectrum[:, None, None] * projector,
                            [{2: primitive_precision}, {4: vortical_precision}], [2, 1], ["u", "omega"])
    expected = 1 / (spectrum.reciprocal() + 4 + wave.square().sum(-1) / 0.5)
    torch.testing.assert_close(problem.posterior((2, 4)), expected[:, None, None] * projector,
                               atol=2e-15, rtol=2e-14)
    assert problem.risk((2, 4)) == pytest.approx(float(expected.sum()), rel=2e-14)


def test_calibrated_ns_transfer_does_not_hide_boundary_sampling_residual():
    basis = SpectralBasis((32, 32), (2 * math.pi, 2 * math.pi))
    resampler = SpectralResampler(basis, (8, 8))
    x = torch.arange(32, dtype=torch.float64) * 2 * math.pi / 32
    phase = torch.linspace(0, 2 * math.pi, 9, dtype=torch.float64)[:-1]
    velocity = torch.zeros(8, 2, 32, 32, dtype=torch.float64)
    velocity[:, 1] = torch.sin(4 * x[None, :, None] + phase[:, None, None])
    calibration = calibrate(velocity, basis, resampler,
                            primary_candidates("incompressible_ns", basis), (32,),
                            divergence_free=True, shellwise=False)
    # A Nyquist sine cannot be distinguished from zero by these coarse nodes.
    # Its residual remains nonzero even when scalar quantization is very fine.
    index = 4 * 32
    assert calibration.noises[0][32][index] > 1.0
    assert calibration.noises[1][32][index] > 1.0


@pytest.mark.parametrize("override", [None, False, True])
@pytest.mark.parametrize("method", ["primitive", "archmulti", "derivopt_archmulti"])
def test_convlstm_simulator_retains_explicit_recurrent_configuration(override, method):
    basis = SpectralBasis((16,), (2 * math.pi,))
    resampler = SpectralResampler(basis, (8,))
    training = torch.randn(4, 1, 16, dtype=torch.float64)
    calibration = calibrate(training, basis, resampler,
                            primary_candidates("advection", basis), (2,))
    geometry = Geometry(basis, resampler, torch.zeros(1, 1, 16), {"family": "advection"})
    options = {"width": 4, "depth": 1}
    if override is not None:
        options["stateful"] = override
    simulator = BudgetedSimulator(calibration, (2, 0, 0), geometry,
                                  backbone="ConvLSTM", method=method, model_kwargs=options)
    expected = True if override is None else override
    predictor = getattr(simulator.predictor, "fine_branch", simulator.predictor)
    assert predictor.stateful is expected
    assert simulator.config["model_kwargs"]["stateful"] is expected
    assert simulator.config["arch_recurrence"] == "fine_branch_only"
    restored = BudgetedSimulator.from_checkpoint_state(simulator.checkpoint_state())
    assert getattr(restored.predictor, "fine_branch", restored.predictor).stateful is expected
    if override is False:
        checkpoint = simulator.checkpoint_state()
        checkpoint["config"]["model_kwargs"].pop("stateful")
        checkpoint["config"].pop("arch_recurrence")
        restored_legacy = BudgetedSimulator.from_checkpoint_state(checkpoint)
        assert getattr(restored_legacy.predictor, "fine_branch", restored_legacy.predictor).stateful is False
    elif method != "primitive":
        checkpoint = simulator.checkpoint_state()
        checkpoint["config"].pop("arch_recurrence")
        with pytest.raises(ValueError, match="two persistent branches"):
            BudgetedSimulator.from_checkpoint_state(checkpoint)
