import pytest

from derivopt.config import ExperimentConfig, paired_seed_configs
from derivopt.data import PDEBenchDataset, write_fixture
from derivopt.preparation import prepare
from derivopt.protocol import CANONICAL, select_canonical_from_pilot
from derivopt.reporting import configuration_macro, paired_seed_summary


def record(family, method, seed, value):
    return dict(family=family, backbone="fno", budget_ratio=.25, retain_frac=.125,
                method=method, seed=seed, nrmse=value)


def test_paired_seed_configs_preserve_regime_and_split():
    base = ExperimentConfig.small(family="advection", seed=99, split_seed=17)
    configurations = list(paired_seed_configs(base, seeds=(2, 4, 6)))
    assert len(configurations) == 6
    assert {(item.seed, item.method) for item in configurations} == {
        (seed, method) for seed in (2, 4, 6) for method in ("derivopt", "archmulti")}
    assert all((item.family, item.budget_ratio, item.retain_frac, item.split_seed)
               == ("advection", .25, .125, 17) for item in configurations)
    assert base.seed == 99 and base.method == "derivopt"


def test_paired_seed_summary_rejects_unequal_seed_sets_across_cells():
    rows = [record("advection", method, seed, .1)
            for method in ("a", "b") for seed in (0, 1)]
    rows += [record("burgers", method, 0, .2) for method in ("a", "b")]
    with pytest.raises(ValueError, match="same paired seed set"):
        paired_seed_summary(rows, "nrmse", "a", "b")


def test_macro_reports_observed_coverage_without_inventing_missing_cells():
    rows = [record("advection", "a", 0, .1), record("burgers", "a", 0, .3),
            record("advection", "b", 0, .2)]
    report = configuration_macro(rows, "nrmse")
    assert report["a"]["mean"] == pytest.approx(.2)
    assert report["a"]["configuration_count"] == 2
    assert report["b"]["configuration_count"] == 1
    assert report["b"]["configurations"][0]["configuration"]["family"] == "advection"


def test_five_calibration_folds_never_use_validation_or_test(tmp_path):
    path = write_fixture(tmp_path / "advection.h5", "advection", trajectories=16, steps=3, spatial_size=32)
    dataset = PDEBenchDataset(path, "advection")
    config = ExperimentConfig.small(family="advection", calibration_states=4, split_seed=12)
    reference_partitions = None
    for fold in range(5):
        prepared = prepare(dataset, config, calibration_fold=fold, folds=5)
        if reference_partitions is None:
            reference_partitions = prepared.partitions
        assert prepared.partitions == reference_partitions
        used = set(prepared.provenance["calibration_trajectory_ids"])
        held_out = {index for i, index in enumerate(prepared.partitions["train"]) if i % 5 == fold}
        assert used and used.issubset(prepared.partitions["train"])
        assert not used.intersection(held_out)
        assert not used.intersection(prepared.partitions["val"] + prepared.partitions["test"])
        assert prepared.provenance["calibration_fold"] == fold


def test_canonical_pilot_rule_filters_failures_and_never_accepts_test_selection():
    def eligible(ratio, retain):
        return dict(budget_ratio=ratio, retain_frac=retain, primitive_finite_fraction=1.,
                    primitive_imputed_nrmse=.1, primitive_imputed_horizon=.05,
                    derivbase_imputed_nrmse=.12, archmulti_imputed_nrmse=.2)

    rows = [eligible(.5, .125), eligible(.25, .25), eligible(*CANONICAL)]
    assert select_canonical_from_pilot(rows, split="train") == CANONICAL
    with pytest.raises(ValueError, match="training-only"):
        select_canonical_from_pilot(rows, split="test")
    rows[-1]["primitive_finite_fraction"] = .9
    assert select_canonical_from_pilot(rows, split="train") == (.5, .125)


def report_both(rows):
    return configuration_macro(rows, "nrmse"), paired_seed_summary(rows, "nrmse", "a", "b")


@pytest.mark.parametrize("field,first,second", [("library", "primary", "shared"), ("split_seed", 0, 1),
                                              ("library", "primary", None), ("split_seed", 0, None)])
def test_reporting_refuses_mixed_protocols(field, first, second):
    rows = [dict(record("advection", "a", 0, .1), **{field: first}),
            dict(record("advection", "b", 0, .2), **{field: second})]
    for summarize in (configuration_macro, lambda rows, metric: paired_seed_summary(rows, metric, "a", "b")):
        with pytest.raises(ValueError, match=field):
            summarize(rows, "nrmse")


@pytest.mark.parametrize("difference", ["dataset", "trajectory", "missing", "multiplicity"])
def test_reporting_rejects_unmatched_actual_test_sets(difference):
    rows = [dict(record("advection", method, 0, .1), library="primary", split_seed=0,
                 dataset_identity={"file": "data.h5", "revision": 1}, trajectory_id=identity)
            for method in ("a", "b") for identity in ("one", "two")]
    if difference == "dataset":
        for row in rows[2:]:
            row["dataset_identity"] = {"file": "other.h5", "revision": 1}
    elif difference == "trajectory":
        rows[-1]["trajectory_id"] = "three"
    elif difference == "missing":
        for row in rows[2:]:
            row.pop("trajectory_id")
    else:
        rows.append(dict(rows[-1]))
    for summarize in (configuration_macro, lambda rows, metric: paired_seed_summary(rows, metric, "a", "b")):
        with pytest.raises(ValueError, match="test identities"):
            summarize(rows, "nrmse")


def test_matched_test_sets_are_verified_per_family_without_false_cross_family_mismatch():
    rows = [dict(record(family, method, seed, .1), library="primary", split_seed=0,
                 dataset_identity={"file": family + ".h5"}, trajectory_id=family + "_test")
            for family in ("advection", "burgers") for method in ("a", "b") for seed in (0, 1)]
    macro, paired = report_both(rows)
    assert macro["a"]["protocol"]["test_identity_verified"]
    assert paired["protocol"]["test_identity_status_by_family"] == {"advection": "verified", "burgers": "verified"}


def test_legacy_metrics_remain_usable_but_identity_is_explicitly_unknown():
    macro, paired = report_both([record("advection", "a", 0, .1), record("advection", "b", 0, .2)])
    assert macro["a"]["mean"] == .1
    assert not paired["protocol"]["test_identity_verified"]
    assert paired["protocol"]["test_identity_status_by_family"] == {"advection": "unknown"}
    assert paired["protocol"]["library"] is None
    rows = [dict(record("advection", method, 0, .1), trajectory_id="one") for method in ("a", "b")]
    assert report_both(rows)[1]["protocol"]["test_identity_status_by_family"] == {"advection": "trajectory_ids_only"}
