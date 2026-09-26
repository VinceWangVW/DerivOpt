import pytest
import torch

from derivopt.baselines import ArchMulti
from derivopt.config import ExperimentConfig
from derivopt.models import make_model


@pytest.mark.parametrize("method", ["primitive", "archmulti", "derivopt_archmulti"])
def test_canonical_recurrent_memory_is_512_kib_for_all_interfaces(method):
    config = ExperimentConfig(backbone="convlstm", method=method)
    def factory(inputs, outputs):
        return make_model("convlstm", 2, inputs, outputs, **config.model_kwargs, stateful=True)
    model = (ArchMulti(factory, 2, 2, 2, config.model_kwargs["width"])
             if "archmulti" in method else factory(2, 2))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model(torch.zeros(1, 2, 32, 32))
        state_bytes = sum(t.numel() * t.element_size()
                          for module in model.modules() if getattr(module, "_state", None) is not None
                          for pair in module._state for t in pair)
        assert state_bytes == 512 * 1024
        model.reset_state()
        assert all(getattr(module, "_state", None) is None for module in model.modules())
    finally:
        torch.set_num_threads(previous_threads)
