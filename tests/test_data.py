import h5py
import json
import numpy as np
import pytest
import torch

from derivopt.data import (
    ChannelStandardizer, DataFormatError, FAMILY_CHANNELS, FAMILY_NDIM,
    PDEBenchDataset, TrajectoryWindows, split_trajectories, write_fixture,
    _sorption_fixture_field,
)


@pytest.mark.parametrize("family", FAMILY_CHANNELS)
def test_disk_fixture_through_production_loader(tmp_path, family):
    path = write_fixture(tmp_path / f"{family}.h5", family, spatial_size=8)
    dataset = PDEBenchDataset(path, family)
    assert len(dataset) == 4
    sample = dataset[0]
    assert sample.states.shape == (5, len(FAMILY_CHANNELS[family]), *((8,) * FAMILY_NDIM[family]))
    assert sample.states.dtype == torch.float32
    assert torch.isfinite(sample.states).all()
    assert sample.states.std() > 0
    assert sample.metadata["file_attributes"]["derivopt_fixture"] is True
    assert sample.metadata["channel_names"] == FAMILY_CHANNELS[family]
    assert sample.times.shape == (5,)
    assert not torch.equal(sample.states[0], sample.states[-1])
    if family == "compressible_ns":
        assert torch.all(sample.states[:, (0, 3)] > 0)


def test_cfd_channel_order_and_coordinates(tmp_path):
    path = write_fixture(tmp_path / "cfd.hdf5", "compressible_ns", spatial_size=(6, 8))
    with h5py.File(path, "r+") as handle:
        for i, key in enumerate(FAMILY_CHANNELS["compressible_ns"]):
            handle[key][...] = i + 1
    sample = PDEBenchDataset(path, "compressible_ns")[0]
    assert torch.equal(sample.states[:, :, 0, 0], torch.tensor([1., 2., 3., 4.]).expand(5, 4))
    assert tuple(c.numel() for c in sample.coords) == (6, 8)
    assert sample.coords[0][0] == pytest.approx(1 / 12)


def test_reaction_writer_axis_order_non_square(tmp_path):
    path = write_fixture(tmp_path / "rd.h5", "diffusion_reaction", spatial_size=(6, 10))
    sample = PDEBenchDataset(path, "diffusion_reaction")[0]
    assert sample.states.shape == (5, 2, 6, 10)
    with h5py.File(path) as handle:
        assert handle["0000/data"].shape == (5, 10, 6, 2)
        expected = torch.from_numpy(handle["0000/data"][()]).permute(0, 3, 2, 1)
    assert torch.equal(sample.states, expected)
    with pytest.raises(DataFormatError, match="coordinates"):
        PDEBenchDataset(path, "diffusion_reaction", reaction_axis_order="xy")[0]


def test_split_window_and_train_only_statistics(tmp_path):
    path = write_fixture(tmp_path / "scalar.hdf5", "advection", trajectories=10, steps=7)
    dataset = PDEBenchDataset(path, "advection")
    splits = split_trajectories(dataset, seed=17)
    all_ids = [set(view.trajectory_ids) for view in splits.values()]
    assert not (all_ids[0] & all_ids[1] or all_ids[0] & all_ids[2] or all_ids[1] & all_ids[2])
    assert set.union(*all_ids) == set(dataset.trajectory_ids)
    assert splits["train"].trajectory_ids == split_trajectories(dataset, seed=17)["train"].trajectory_ids
    with pytest.raises(ValueError, match="train split"):
        ChannelStandardizer.fit(splits["val"])
    with pytest.raises(ValueError, match="train split"):
        ChannelStandardizer.fit(dataset)
    scaler = ChannelStandardizer.fit(splits["train"])
    full_train = torch.cat([s.states for s in splits["train"]])
    transformed = scaler.transform(full_train)
    assert abs(transformed.mean()) < 1e-6
    assert transformed.std(correction=0) == pytest.approx(1, abs=1e-6)
    assert torch.allclose(scaler.inverse(transformed), full_train, atol=2e-7)
    scaler.save(tmp_path / "scaler.json")
    loaded = ChannelStandardizer.load(tmp_path / "scaler.json")
    assert loaded.state_dict() == scaler.state_dict()
    windows = TrajectoryWindows(splits["train"], history=2, horizon=3, stride=2)
    assert len(windows) == len(splits["train"]) * 2
    item = windows[1]
    source = splits["train"][0]
    assert item["start"] == 2
    assert torch.equal(item["inputs"], source.states[2:4])
    assert torch.equal(item["targets"], source.states[4:7])
    assert item["trajectory_id"] in all_ids[0]


def test_nontrain_outliers_do_not_change_statistics(tmp_path):
    path = write_fixture(tmp_path / "scalar.hdf5", "burgers")
    dataset = PDEBenchDataset(path, "burgers")
    train = dataset.subset([0, 1], split="train")
    expected = ChannelStandardizer.fit(train)
    with h5py.File(path, "r+") as handle:
        handle["tensor"][2:] = 1e12
    actual = ChannelStandardizer.fit(train)
    assert actual.state_dict() == expected.state_dict()


def test_ins_geometry_and_missing_velocity_rejected(tmp_path):
    path = write_fixture(tmp_path / "ns.h5", "incompressible_ns", spatial_size=(6, 8))
    sample = PDEBenchDataset(path, "incompressible_ns")[0]
    assert torch.equal(sample.coords[0], torch.arange(6, dtype=torch.float64) + 0.5)
    assert sample.metadata["simulation_config"]["velocity_extrapolation"] == "PERIODIC"
    invalid = tmp_path / "vorticity.h5"
    with h5py.File(invalid, "w") as handle:
        handle.create_dataset("vorticity", data=np.zeros((2, 5, 6, 8)))
    with pytest.raises(DataFormatError, match="vorticity-only"):
        PDEBenchDataset(invalid, "incompressible_ns")


def test_strides_and_unused_nle_time_coordinate(tmp_path):
    path = write_fixture(tmp_path / "scalar.h5", "advection", spatial_size=12, steps=7)
    with h5py.File(path, "r+") as handle:
        previous = handle["t-coordinate"][()]
        del handle["t-coordinate"]
        handle.create_dataset("t-coordinate", data=np.append(previous, .5))
    sample = PDEBenchDataset(path, "advection", spatial_stride=2, time_stride=2)[0]
    assert sample.states.shape == (4, 1, 6)
    assert sample.metadata["unused_trailing_time_coordinate"]


@pytest.mark.parametrize("damage, message", [("nan", "Non-finite"), ("time", "strictly increasing"),
                                             ("coord", "coordinates")])
def test_bad_inputs_fail_clearly(tmp_path, damage, message):
    path = write_fixture(tmp_path / "broken.h5", "advection")
    with h5py.File(path, "r+") as handle:
        if damage == "nan":
            handle["tensor"][0, 0, 0] = float("nan")
        elif damage == "time":
            handle["t-coordinate"][2] = handle["t-coordinate"][1]
        else:
            del handle["x-coordinate"]
            handle.create_dataset("x-coordinate", data=np.arange(3))
    with pytest.raises(DataFormatError, match=message):
        PDEBenchDataset(path, "advection")[0]


def test_lazy_dataset_reads_updated_trajectory_and_multiple_files(tmp_path):
    paths = [write_fixture(tmp_path / f"part{i}.h5", "burgers", trajectories=2) for i in range(2)]
    dataset = PDEBenchDataset(paths, "burgers")
    assert len(dataset) == 4
    with h5py.File(paths[1], "r+") as handle:
        handle["tensor"][1] = 42
    assert torch.all(dataset[3].states == 42)
    assert not any(isinstance(v, h5py.File) for v in dataset.__dict__.values())
    with pytest.raises(FileExistsError):
        write_fixture(paths[0], "burgers")


def test_constant_channel_standardizer_is_finite(tmp_path):
    path = write_fixture(tmp_path / "constant.h5", "advection")
    with h5py.File(path, "r+") as handle:
        handle["tensor"][...] = 3
    dataset = PDEBenchDataset(path, "advection").subset([0, 1], split="train")
    scaler = ChannelStandardizer.fit(dataset)
    assert torch.all(scaler.transform(dataset[0].states) == 0)
    assert torch.equal(scaler.inverse(scaler.transform(dataset[0].states)), dataset[0].states)


def test_sorption_fixture_manufactured_boundary_and_native_disk_values(tmp_path):
    diffusion, step = 5e-4, 1e-5
    for time in (0., .15, .4):
        for phase in (-.2, .1):
            value = lambda x: _sorption_fixture_field(np.array(x), time, phase, .25)
            assert value(0.) == pytest.approx(1.)
            derivative_right = (value(1. + step) - value(1. - step)) / (2 * step)
            assert value(1.) + diffusion * derivative_right == pytest.approx(0., abs=1e-10)
    path = write_fixture(tmp_path / "sorption.h5", "diffusion_sorption", spatial_size=12, seed=13)
    sample = PDEBenchDataset(path, "diffusion_sorption")[0]
    phase = np.random.default_rng(13).uniform(-.2, .2)
    expected = _sorption_fixture_field(sample.coords[0].numpy(), 0., phase, .2)
    torch.testing.assert_close(sample.states[0, 0], torch.tensor(expected, dtype=torch.float32))
    assert "not a PDE solution" in sample.metadata["trajectory_attributes"]["manufactured_boundary"]


@pytest.mark.parametrize("family", FAMILY_CHANNELS)
def test_time_window_reads_requested_hdf5_slice_and_preserves_values(tmp_path, monkeypatch, family):
    path = write_fixture(tmp_path / "window.h5", family, spatial_size=8, steps=11)
    dataset = PDEBenchDataset(path, family, time_stride=2).subset([1], split="train")
    full = dataset[0]
    reads = []
    original = h5py.Dataset.__getitem__

    def traced_read(array, selection):
        if array.name.endswith(("/data", "/tensor", "/velocity", "/density", "/pressure", "/Vx", "/Vy")):
            reads.append(selection)
        return original(array, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", traced_read)
    window = dataset.read_window(0, 1, 4)
    torch.testing.assert_close(window.states, full.states[1:4], rtol=0, atol=0)
    torch.testing.assert_close(window.times, full.times[1:4], rtol=0, atol=0)
    assert window.metadata["trajectory_id"] == full.metadata["trajectory_id"]
    assert window.metadata["split"] == "train"
    assert reads
    for selection in reads:
        time_slice = selection[0] if family in {"diffusion_sorption", "diffusion_reaction", "radial_dam_break"} else selection[1]
        assert (time_slice.start, time_slice.stop, time_slice.step) == (2, 8, 2)


def test_window_reads_no_unrequested_states_but_validates_complete_time_grid(tmp_path):
    path = write_fixture(tmp_path / "partial.h5", "advection", steps=8)
    dataset = PDEBenchDataset(path, "advection")
    with h5py.File(path, "r+") as handle:
        handle["tensor"][0, 0, 0] = np.nan
    assert torch.isfinite(dataset.read_window(0, 2, 5).states).all()
    with pytest.raises(DataFormatError, match="Non-finite"):
        dataset[0]
    with h5py.File(path, "r+") as handle:
        handle["t-coordinate"][-1] += .01
    with pytest.raises(DataFormatError, match="Nonuniform"):
        dataset.read_window(0, 2, 5)


def test_trajectory_windows_uses_lazy_api_without_full_trajectory_read(tmp_path, monkeypatch):
    path = write_fixture(tmp_path / "windows.h5", "burgers", steps=9)
    dataset = PDEBenchDataset(path, "burgers")
    full = dataset[0]

    def forbidden_full_read(self, index):
        raise AssertionError("Full trajectory read is forbidden for window access")

    monkeypatch.setattr(PDEBenchDataset, "__getitem__", forbidden_full_read)
    window = TrajectoryWindows(dataset, history=2, horizon=3, stride=2)[1]
    assert window["start"] == 2
    torch.testing.assert_close(window["inputs"], full.states[2:4])
    torch.testing.assert_close(window["targets"], full.states[4:7])


@pytest.mark.parametrize("start,stop", [(-1,3), (0,1), (2,2), (0,6), (1.5,4), (0,True)])
def test_invalid_time_windows_fail_before_reading_states(tmp_path, start, stop):
    dataset = PDEBenchDataset(write_fixture(tmp_path / "invalid_window.h5", "advection"), "advection")
    with pytest.raises(ValueError, match="Time window"):
        dataset.read_window(0, start, stop)


def test_ins_extracted_real_window_preserves_source_latest_index(tmp_path):
    path = write_fixture(tmp_path / "ns_subset.h5", "incompressible_ns", spatial_size=8, steps=5)
    provenance = {"format_version": 1, "source_frame_count": 1000, "source_frame_start": 0,
                  "source_frame_stop": 5, "source_frame_stride": 1, "source_latestIndex": 999,
                  "source_trajectory_indices": [0, 1, 2, 3]}
    with h5py.File(path, "r+") as handle:
        handle.attrs["latestIndex"] = 999
        handle.attrs["derivopt_source_subset"] = json.dumps(provenance)
    dataset = PDEBenchDataset(path, "incompressible_ns")
    assert dataset.read_window(0, 1, 4).metadata["file_attributes"]["latestIndex"] == 999
    with h5py.File(path, "r+") as handle:
        provenance["source_frame_stop"] = 6
        handle.attrs["derivopt_source_subset"] = json.dumps(provenance)
    with pytest.raises(DataFormatError, match="source-subset provenance"):
        PDEBenchDataset(path, "incompressible_ns")


def _reaction_phase_file(path, *, size=32, phase=-.125, smooth=False):
    spacing = 2/size
    axis = -1+(np.arange(size)+.5+phase)*spacing
    x, y = np.meshgrid(axis, axis, indexing="xy")
    field = np.cos(np.pi*(x+1))*np.cos(np.pi*(y+1)) if smooth else x+2*y
    state = np.stack((field, 3*field), axis=-1)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("0000")
        group.attrs["config"] = json.dumps({"sim": {"x_left": -1, "x_right": 1,
                                                     "y_bottom": -1, "y_top": 1}})
        group.create_dataset("data", data=np.stack((state, 2*state)).astype(np.float32))
        group.create_dataset("grid/x", data=axis)
        group.create_dataset("grid/y", data=axis)
        group.create_dataset("grid/t", data=[0., .1])
    return axis, state


def test_reaction_phase_alignment_interpolates_values_and_keeps_source_coordinates(tmp_path):
    axis, raw = _reaction_phase_file(tmp_path/"phase.h5")
    sample = PDEBenchDataset(tmp_path/"phase.h5", "diffusion_reaction")[0]
    expected_axis = -1+(torch.arange(32, dtype=torch.float64)+.5)*2/32
    torch.testing.assert_close(sample.coords[0], expected_axis)
    x, y = torch.meshgrid(*sample.coords, indexing="ij")
    # Linear interpolation is exact away from the Neumann extension at the edge.
    torch.testing.assert_close(sample.states[0, 0, :-1, :-1].double(), (x+2*y)[:-1, :-1], atol=1e-6, rtol=0)
    assert sample.metadata["source_axes"]["x"] == axis.tolist()
    assert sample.metadata["coordinate_preprocessing"]["source_phase_in_stored_cells"] == [-.125, -.125]
    assert sample.metadata["coordinate_preprocessing"]["applied"]
    assert not np.array_equal(sample.states[0, 0].numpy(), raw[..., 0].T)
    with h5py.File(tmp_path/"phase.h5") as handle:
        np.testing.assert_array_equal(handle["0000/grid/x"][...], axis)
        np.testing.assert_array_equal(handle["0000/data"][0], raw.astype(np.float32))


@pytest.mark.parametrize("phase", [-.125, .125])
def test_reaction_phase_alignment_smooth_neumann_field(tmp_path, phase):
    _reaction_phase_file(tmp_path/"smooth.h5", size=128, phase=phase, smooth=True)
    sample = PDEBenchDataset(tmp_path/"smooth.h5", "diffusion_reaction")[0]
    x, y = torch.meshgrid(*sample.coords, indexing="ij")
    truth = torch.cos(torch.pi*(x+1))*torch.cos(torch.pi*(y+1))
    assert (sample.states[0, 0]-truth).abs().max() < 8e-4


@pytest.mark.parametrize("damage", ["large_phase", "cropped", "missing_bounds"])
def test_reaction_unknown_or_cropped_geometry_is_rejected(tmp_path, damage):
    _reaction_phase_file(tmp_path/"bad_phase.h5", phase=.4 if damage == "large_phase" else -.125)
    with h5py.File(tmp_path/"bad_phase.h5", "r+") as handle:
        if damage == "cropped":
            handle["0000/grid/x"][...] *= .5
        elif damage == "missing_bounds":
            del handle["0000"].attrs["config"]
    with pytest.raises(DataFormatError, match="(coordinate phase|cropped|declared)"):
        PDEBenchDataset(tmp_path/"bad_phase.h5", "diffusion_reaction")[0]


def test_native_unbatched_cns_preserves_schema_and_declares_official_boundary(tmp_path):
    original = write_fixture(tmp_path/"batched.h5", "compressible_ns", spatial_size=(6, 8), steps=7)
    path = tmp_path/"2D_shock.hdf5"
    with h5py.File(original) as source, h5py.File(path, "w") as target:
        for name in (*FAMILY_CHANNELS["compressible_ns"], "x-coordinate", "y-coordinate", "t-coordinate"):
            target.create_dataset(name, data=source[name][0] if name in FAMILY_CHANNELS["compressible_ns"] else source[name][...])
    dataset = PDEBenchDataset(path, "compressible_ns")
    assert len(dataset) == 1
    sample = dataset.read_window(0, 2, 5)
    torch.testing.assert_close(sample.states, PDEBenchDataset(original, "compressible_ns")[0].states[2:5])
    assert sample.metadata["layout"] == "cfd_single"
    assert sample.metadata["boundary"] == "trans"
    with h5py.File(path) as handle:
        assert handle["density"].ndim == 3


def test_cns_source_filename_provenance_is_strict(tmp_path):
    path = write_fixture(tmp_path/"my_trans_data.hdf5", "compressible_ns", spatial_size=8)
    assert "boundary" not in PDEBenchDataset(path, "compressible_ns")[0].metadata
    filename = "2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
    with h5py.File(path, "r+") as handle:
        handle.attrs["derivopt_source_subset"] = json.dumps({"source_filename": filename})
    metadata = PDEBenchDataset(path, "compressible_ns")[0].metadata
    assert metadata["boundary"] == "periodic"
    assert metadata["source_filename"] == filename
