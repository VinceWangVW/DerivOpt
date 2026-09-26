"""Latent codec round-trip, training-gradient and serialized-rate tests."""

import io
import random

import pytest
import torch

from derivopt.entropy import decode_symbols, encode_symbols
from derivopt.latent import HyperpriorCodec


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_arithmetic_coder_roundtrip_varying_cdfs():
    rng = random.Random(19)
    tables = [(0, 1, 15, 20), (0, 9, 10, 40), (0, 1, 2, 65536)]
    for count in (1, 2, 3, 31, 1000):
        symbols = [rng.randrange(3) for _ in range(count)]
        cdfs = [tables[rng.randrange(len(tables))] for _ in range(count)]
        payload = encode_symbols(symbols, cdfs)
        assert decode_symbols(payload, cdfs) == symbols


def test_arithmetic_coder_uses_the_supplied_probability_model():
    symbols = [1] * 1000
    likely = (0, 1, 65535, 65536)
    unlikely = (0, 32767, 32768, 65536)
    assert len(encode_symbols(symbols, [likely] * len(symbols))) < 4
    assert len(encode_symbols(symbols, [unlikely] * len(symbols))) > 1000


def test_arithmetic_coder_rejects_invalid_cdfs_and_noncanonical_stream():
    with pytest.raises(ValueError, match="strictly increasing"):
        encode_symbols([0], [(0, 0, 4)])
    with pytest.raises(ValueError, match="alphabet"):
        encode_symbols([2], [(0, 2, 4)])
    payload = encode_symbols([1] * 10, [(0, 1, 2)] * 10)
    with pytest.raises(ValueError, match="Noncanonical"):
        decode_symbols(payload + b"\x00", [(0, 1, 2)] * 10)


def test_hyperlatents_change_primary_cdf_and_cdf_masses_are_valid():
    torch.manual_seed(43)
    codec = HyperpriorCodec(2, 1, latent_channels=4, width=6, shape=(16, 16))
    zero = torch.zeros(1, codec.hyper_channels, *codec.hyper_shape)
    shifted = torch.full_like(zero, 10.0)
    _, zero_indices = codec._conditional_scales(zero, codec.quant_steps[2])
    _, shifted_indices = codec._conditional_scales(shifted, codec.quant_steps[2])
    assert not torch.equal(zero_indices, shifted_indices)
    assert (codec.frequency_table > 0).all()
    assert (codec.frequency_table.sum(1) == 65536).all()
    assert torch.equal(codec.cdf_table[:, 1:] - codec.cdf_table[:, :-1], codec.frequency_table)


@pytest.mark.parametrize("shape", [(17,), (13, 9)])
def test_hyperprior_training_updates_all_networks(shape):
    torch.manual_seed(81)
    model = HyperpriorCodec(len(shape), channels=2, latent_channels=3, hyper_channels=2, width=6, shape=shape)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.randn(2, 2, *shape)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction, aux = model(inputs)
        assert prediction.shape == inputs.shape
        assert aux["rate_bits"].shape == (inputs.shape[0],)
        loss = aux["reconstruction"] + 0.01 * aux["rate"]
        assert torch.isfinite(loss)
        loss.backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        optimizer.step()
    for prefix in ("encoder", "decoder", "hyperencoder", "hyperdecoder", "hyper_log_scale"):
        assert any(not torch.equal(value, initial[name]) for name, value in model.named_parameters() if name.startswith(prefix)), prefix
    assert model.architecture_metadata["configuration"] == "explicit_scale_hyperprior_settings"


@pytest.mark.parametrize("shape", [(1,), (17,), (1, 1), (13, 9)])
def test_real_stream_matches_rounded_forward_and_loaded_checkpoint(shape):
    torch.manual_seed(7)
    settings = dict(spatial_dim=len(shape), channels=2, latent_channels=3, width=6, shape=shape)
    codec = HyperpriorCodec(**settings).eval()
    inputs = torch.randn(1, 2, *shape)
    payload, ledger = codec.encode(inputs, cap_bytes=128)
    assert isinstance(payload, bytes) and len(payload) == 128
    ledger.validate(payload)
    assert ledger.primary_bits > 0 and ledger.hyper_bits > 0 and ledger.header_bits > 0
    assert ledger.field_bits == 0
    info = codec.record_info(payload)
    assert ledger.header_bits == info["header_bytes"] * 8
    assert ledger.hyper_bits == info["hyper_bytes"] * 8
    assert ledger.primary_bits == info["primary_bytes"] * 8
    assert ledger.padding_bits == info["padding_bytes"] * 8
    with torch.no_grad():
        expected, aux = codec(inputs, step_index=info["step_index"])
    decoded = codec.decode(payload)
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
    # Estimated entropy excludes framing/termination/padding, unlike the ledger.
    assert aux["rate"].item() < ledger.total_bits
    checkpoint = io.BytesIO()
    torch.save(codec.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = HyperpriorCodec(**settings).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    del inputs, codec
    torch.testing.assert_close(restored.decode(payload), expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(8,), (8, 8)])
def test_small_hard_cap_is_actually_satisfied(shape):
    torch.manual_seed(1)
    codec = HyperpriorCodec(len(shape), channels=1, latent_channels=2, width=4, shape=shape)
    inputs = torch.zeros(1, 1, *shape)
    for cap in (16, 32):
        payload, ledger = codec.encode(inputs, cap)
        assert len(payload) == cap
        assert ledger.total_bits == ledger.cap_bits == cap * 8
        assert torch.isfinite(codec.decode(payload)).all()


def test_rate_control_changes_step_and_charges_index():
    torch.manual_seed(4)
    codec = HyperpriorCodec(2, 1, latent_channels=4, width=8, shape=(32, 32))
    inputs = torch.randn(1, 1, 32, 32)
    loose, _ = codec.encode(inputs, cap_bytes=2048)
    tight, ledger = codec.encode(inputs, cap_bytes=16)
    loose_info, tight_info = codec.record_info(loose), codec.record_info(tight)
    assert loose_info["step_index"] < tight_info["step_index"]
    assert ledger.header_bits >= 9 * 8  # includes quantization-step index
    expected, _ = codec(inputs, step_index=tight_info["step_index"])
    torch.testing.assert_close(codec.decode(tight), expected, rtol=0, atol=0)


def test_unattainable_cap_raises_instead_of_bypassing_coding():
    codec = HyperpriorCodec(2, 1, latent_channels=4, width=4, shape=(32, 32), quant_steps=(0.25,))
    with pytest.raises(ValueError, match="at least"):
        codec.encode(torch.zeros(1, 1, 32, 32), cap_bytes=4)
    with pytest.raises(ValueError, match="No configured quantization step"):
        codec.encode(torch.zeros(1, 1, 32, 32), cap_bytes=11)


def test_corrupted_payload_header_and_capacity_padding_are_rejected():
    codec = HyperpriorCodec(1, 1, latent_channels=2, width=4, shape=(8,))
    payload, _ = codec.encode(torch.zeros(1, 1, 8), 64)
    damaged = bytearray(payload)
    damaged[2] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        codec.decode(bytes(damaged))
    damaged = bytearray(payload)
    damaged[codec.record_info(payload)["header_bytes"]] ^= 128
    with pytest.raises(ValueError, match="checksum"):
        codec.decode(bytes(damaged))
    damaged = bytearray(payload)
    damaged[-1] = 1
    with pytest.raises(ValueError, match="padding"):
        codec.decode(bytes(damaged))
    with pytest.raises(ValueError, match="truncated"):
        codec.decode(payload[:5])


def test_quantization_is_clipped_identically_in_train_and_eval():
    codec = HyperpriorCodec(1, 1, latent_channels=2, width=4, shape=(8,), symbol_limit=3)
    inputs = torch.full((1, 1, 8), 1e5)
    codec.train()
    train_result, train_aux = codec(inputs, step_index=0)
    codec.eval()
    eval_result, eval_aux = codec(inputs, step_index=0)
    torch.testing.assert_close(train_result, eval_result, rtol=0, atol=0)
    assert train_aux["latent_symbols"].abs().max() <= 3
    assert train_aux["hyperlatent_symbols"].abs().max() <= 3
    torch.testing.assert_close(train_aux["rate_bits"], eval_aux["rate_bits"], rtol=0, atol=0)


def test_checkpoint_configuration_must_match_decoder():
    codec = HyperpriorCodec(1, 1, latent_channels=2, width=4, shape=(8,))
    other = HyperpriorCodec(1, 1, latent_channels=2, width=4, shape=(12,))
    with pytest.raises(ValueError, match="checkpoint configuration"):
        other.load_state_dict(codec.state_dict())


def test_encode_is_single_state_and_finite():
    codec = HyperpriorCodec(1, 1, latent_channels=2, width=4, shape=(8,))
    with pytest.raises(ValueError, match="exactly one"):
        codec.encode(torch.zeros(2, 1, 8), 32)
    with pytest.raises(ValueError, match="finite"):
        codec.encode(torch.full((1, 1, 8), float("nan")), 32)
