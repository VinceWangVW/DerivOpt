import io
import json
import math
from dataclasses import replace

import pytest
import torch

from derivopt.data import PDEBenchDataset, write_fixture
from derivopt.geometry import Geometry, from_sample


def fixture(tmp_path, family="diffusion_sorption", spatial_size=12):
    path = write_fixture(tmp_path / f"{family}.h5", family, spatial_size=spatial_size)
    return PDEBenchDataset(path, family)[0]


def test_diffusion_sorption_known_boundary_lifting(tmp_path):
    sample = fixture(tmp_path)
    geometry = from_sample(sample, 0.5)
    assert geometry.basis.boundaries == ("robin",)
    assert geometry.basis.robin_coefficients == ((math.inf, 2000.0),)
    assert geometry.resampler.coarse_shape == (6,)
    assert geometry.lifting.shape == (1, 1, 12)
    params = geometry.config["boundary_parameters"]
    diffusion, sol, length = params["D"], params["sol"], params["x_right"] - params["x_left"]
    derivative = -sol / (length + diffusion)
    lift_at_left = sol * (length + diffusion) / (length + diffusion)
    lift_at_right = sol * diffusion / (length + diffusion)
    assert lift_at_left == pytest.approx(sol)
    assert lift_at_right + diffusion * derivative == pytest.approx(0)
    expected = sol * (length + diffusion - (sample.coords[0] - params["x_left"])) / (length + diffusion)
    torch.testing.assert_close(geometry.lifting[0, 0], expected)


def test_lift_roundtrip_dtype_batch_time_and_gradient(tmp_path):
    sample = fixture(tmp_path)
    geometry = Geometry.from_sample(sample, 0.5)
    physical = sample.states.unsqueeze(0).repeat(2, 1, 1, 1).requires_grad_(True)
    residual = geometry.to_residual(physical)
    assert residual.dtype == physical.dtype
    torch.testing.assert_close(geometry.to_physical(residual), physical)
    residual.sum().backward()
    torch.testing.assert_close(physical.grad, torch.ones_like(physical))
    with pytest.raises(ValueError, match="trailing"):
        geometry.to_residual(torch.zeros(1, 2, 12))


def test_parameters_cannot_be_silently_guessed(tmp_path):
    sample = fixture(tmp_path)
    no_config = replace(sample, metadata={})
    with pytest.raises(ValueError, match="D, sol, x_left, x_right"):
        Geometry.from_sample(no_config, 0.5)
    geometry = Geometry.from_sample(no_config, 0.5, boundary_parameters={"D": 0.1, "sol": 2, "x_left": 0, "x_right": 1})
    assert geometry.basis.robin_coefficients == ((math.inf, 10.0),)
    with pytest.raises(ValueError, match="cell-centred"):
        Geometry.from_sample(no_config, 0.5, boundary_parameters={"D": 0.1, "sol": 2, "x_left": 0, "x_right": 2})


def test_ins_source_zero_boundary_uses_no_slip_basis(tmp_path):
    sample = fixture(tmp_path, "incompressible_ns")
    assert Geometry.from_sample(sample, 0.5).basis.boundaries == ("periodic", "periodic")
    metadata = {**sample.metadata, "simulation_config": {"velocity_extrapolation": "ZERO"}}
    zero_sample = replace(sample, metadata=metadata)
    no_slip = Geometry.from_sample(zero_sample, 0.5)
    assert no_slip.basis.boundaries == ("robin", "robin")
    assert no_slip.basis.robin_coefficients == ((math.inf, math.inf),)*2
    assert no_slip.config["lifting"] == "homogeneous_no_slip"
    declared_override = Geometry.from_sample(zero_sample, 0.5, boundary_override="periodic")
    assert declared_override.config["boundary_source"] == "explicit_override"
    assert declared_override.config["source_boundary_declarations"] == ("zero",)
    assert Geometry.from_sample(sample, 0.5, boundary_override="neumann").basis.boundaries == ("neumann",)*2


def test_ins_file_attribute_boundary_is_read_even_with_explicit_coordinates(tmp_path):
    sample = fixture(tmp_path, "incompressible_ns")
    metadata = {"file_attributes": {"config": json.dumps({"velocity_extrapolation": "ZERO"})}}
    assert Geometry.from_sample(replace(sample, metadata=metadata), 0.5).basis.boundaries == ("robin",)*2


def test_cns_transmissive_boundary_and_dr_declared_domain(tmp_path):
    cns = fixture(tmp_path, "compressible_ns")
    cns = replace(cns, metadata={**cns.metadata, "simulation_config": {"bc": "trans"}})
    assert Geometry.from_sample(cns, .5).basis.boundaries == ("neumann",)*2
    dr = fixture(tmp_path, "diffusion_reaction")
    # A changed physical domain cannot be hidden by labelling a shifted axis
    # as cell-centred. The reader must interpolate values as well as coordinates.
    simulation = {"x_left": -1., "x_right": 1., "y_bottom": -1., "y_top": 1.}
    bad = replace(dr, metadata={**dr.metadata, "simulation_config": simulation})
    with pytest.raises(ValueError, match="declared physical domain"):
        Geometry.from_sample(bad, .5)


@pytest.mark.parametrize("family,expected", [("advection", "periodic"), ("burgers", "periodic"), ("diffusion_reaction", "neumann"), ("radial_dam_break", "neumann"), ("compressible_ns", "periodic")])
def test_family_defaults_and_zero_lift(tmp_path, family, expected):
    sample = fixture(tmp_path, family)
    geometry = Geometry.from_sample(sample, 0.5)
    assert all(value == expected for value in geometry.basis.boundaries)
    assert torch.all(geometry.lifting == 0)
    torch.testing.assert_close(geometry.to_residual(sample.states), sample.states)


def test_nonuniform_rejected_and_retained_grid_not_silently_clipped(tmp_path):
    sample = fixture(tmp_path, "advection", spatial_size=10)
    coordinates = sample.coords[0].clone()
    coordinates[3] += 0.01
    with pytest.raises(ValueError, match="uniform"):
        Geometry.from_sample(replace(sample, coords=(coordinates,)), 0.5)
    with pytest.raises(ValueError, match="between 2"):
        Geometry.from_sample(sample, 0.1)
    geometry = Geometry.from_sample(sample, 0.37)
    assert geometry.resampler.coarse_shape == (3,)
    assert geometry.config["retain_frac_requested"] == (0.37,)
    assert geometry.config["retain_frac_realized"] == (0.3,)
    with pytest.raises(ValueError, match="integer"):
        Geometry.from_sample(sample, 0.5, coarse_shape=(3.5,))


def test_checkpoint_tensors_only_roundtrip(tmp_path):
    sample = fixture(tmp_path)
    geometry = Geometry.from_sample(sample, 0.5)
    buffer = io.BytesIO()
    torch.save(geometry.state_dict(), buffer)
    buffer.seek(0)
    restored = Geometry.from_state_dict(torch.load(buffer, weights_only=True))
    assert restored.config == geometry.config
    torch.testing.assert_close(restored.to_residual(sample.states), geometry.to_residual(sample.states))
    torch.testing.assert_close(restored.basis.eigenvalues, geometry.basis.eigenvalues)


def test_sample_validation_checks_coordinates_without_rebuilding_basis(tmp_path, monkeypatch):
    sample = fixture(tmp_path, "diffusion_reaction")
    geometry = Geometry.from_sample(sample, 0.5)

    def forbidden(*args, **kwargs):
        raise AssertionError("sample validation must not construct a spectral basis")

    monkeypatch.setattr("derivopt.geometry.SpectralBasis", forbidden)
    geometry.validate_sample(sample)
    shifted = (sample.coords[0] + 0.2, sample.coords[1])
    with pytest.raises(ValueError, match="physical coordinates/domain"):
        geometry.validate_sample(replace(sample, coords=shifted))
    distorted = sample.coords[0].clone()
    distorted[3] += 0.01
    with pytest.raises(ValueError, match="physical coordinates/domain"):
        geometry.validate_sample(replace(sample, coords=(distorted, sample.coords[1])))


def test_sample_validation_distinguishes_source_parameters_from_overrides(tmp_path):
    sample = fixture(tmp_path)
    geometry = Geometry.from_sample(sample, 0.5)
    changed = dict(sample.metadata["simulation_config"])
    changed["sol"] = 2.0
    changed_sample = replace(sample, metadata={**sample.metadata, "simulation_config": changed})
    with pytest.raises(ValueError, match="boundary parameters/lifting"):
        geometry.validate_sample(changed_sample)
    # An explicitly fixed parameter is a declared experiment assumption; it
    # must not be confused with an original source-derived default.
    fixed = Geometry.from_sample(sample, 0.5, boundary_parameters={"sol": 1.0})
    fixed.validate_sample(changed_sample)


def test_geometry_v1_cannot_silently_guess_coordinate_origin(tmp_path):
    geometry = Geometry.from_sample(fixture(tmp_path), 0.5)
    state = geometry.state_dict()
    state["format_version"] = 1
    with pytest.raises(ValueError, match="regenerate preparation"):
        Geometry.from_state_dict(state)
