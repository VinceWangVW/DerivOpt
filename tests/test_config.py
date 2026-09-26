from derivopt.config import ExperimentConfig, main_matrix, paired_seed_configs
from derivopt.protocol import select_canonical_from_pilot
import pytest


def test_full_main_matrix_and_shared_selector_cells():
    configs = list(main_matrix(smoke=True))
    assert len(configs) == 1764
    assert len({(c.family, c.budget_ratio, c.retain_frac) for c in configs}) == 63


def test_paired_seeds_preserve_data_split():
    configs = list(paired_seed_configs(ExperimentConfig.small(), [1, 2, 3, 4, 5]))
    assert len(configs) == 10 and {c.split_seed for c in configs} == {0}


def test_pilot_selects_storage_first_and_rejects_test_split():
    common = dict(primitive_finite_fraction=.99, primitive_imputed_nrmse=.1, primitive_imputed_horizon=.05,
                  derivbase_imputed_nrmse=.1, archmulti_imputed_nrmse=.2)
    rows = [dict(common, budget_ratio=.25, retain_frac=.125), dict(common, budget_ratio=.5, retain_frac=.125)]
    assert select_canonical_from_pilot(rows, split="train") == (.25, .125)
    with pytest.raises(ValueError):
        select_canonical_from_pilot(rows, split="test")


@pytest.mark.parametrize("name", ["budget_ratio", "retain_frac", "learning_rate", "latent_rate_weight", "latent_reconstruction_weight"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_experiment_numbers_are_rejected(name, value):
    with pytest.raises(ValueError, match="finite"):
        ExperimentConfig(**{name: value}).validate()


@pytest.mark.parametrize("name,value", [("train_steps", True), ("seed", -1), ("split_seed", float("nan")),
                                      ("selector_max_nodes", float("inf"))])
def test_integer_configuration_fields_reject_invalid_values(name, value):
    with pytest.raises(ValueError, match="integer"):
        ExperimentConfig(**{name: value}).validate()


def test_nested_model_numbers_are_finite_but_dirichlet_infinity_is_allowed():
    with pytest.raises(ValueError, match="model_kwargs.dropout must be finite"):
        ExperimentConfig(model_kwargs={"dropout": float("nan")}).validate()
    with pytest.raises(ValueError, match="latent_kwargs.steps"):
        ExperimentConfig(latent_kwargs={"steps": [1., float("inf")]}).validate()
    ExperimentConfig(boundary_parameters={"robin_coefficients": [[float("inf"), float("inf")]]}).validate()
