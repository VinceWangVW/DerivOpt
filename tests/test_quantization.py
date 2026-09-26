import pytest
import torch

from derivopt.budget import PayloadBudget
from derivopt.quantization import ExplicitRecordCodec, ScalarQuantizer, pack_unsigned, unpack_unsigned
from derivopt.widecodes import WideCodes, quantize_unit_values, round_ratio_even


def test_canonical_budget_and_real_record():
    budget = PayloadBudget(1024, 2, .25)
    assert budget.cap_bytes == 2048 and budget.field_bits_per_site == 16
    x = torch.randn(4, 2, 32, 32)
    q = ScalarQuantizer.fit(x, split="train")
    codec = ExplicitRecordCodec([q], [8], [(2, 32, 32)], budget.cap_bytes)
    payload, ledger = codec.encode([x[0]])
    assert len(payload) == 2048 and ledger.total_bits == 16384
    torch.testing.assert_close(codec.decode(payload)[0], q(x[:1], 8)[0])


@pytest.mark.parametrize("width", [1, 2, 3, 5, 8, 16, 32])
def test_unsigned_roundtrip(width):
    values = torch.tensor([0, 1, (1 << width)-1], dtype=torch.int64)
    payload = pack_unsigned([values], [width], 20)
    assert torch.equal(unpack_unsigned(payload, [(3,)], [width])[0], values)


def test_padding_and_invalid_payload():
    with pytest.raises(ValueError, match="exceeds"):
        pack_unsigned([torch.arange(4)], [3], 1)
    with pytest.raises(ValueError, match="padding"):
        unpack_unsigned(bytes([255]), [(1,)], [1])
    with pytest.raises(ValueError, match="truncated"):
        unpack_unsigned(bytes([0]), [(2,)], [8])
    with pytest.raises(ValueError, match="training split"):
        ScalarQuantizer.fit(torch.randn(2, 1, 8), split="test")


def test_ste_has_real_quantized_forward_and_gradient():
    train = torch.randn(4, 1, 8)
    q = ScalarQuantizer.fit(train, split="train")
    x = train[:1].clone().requires_grad_()
    out = q(x, 2, straight_through=True)
    torch.testing.assert_close(out, q(x.detach(), 2))
    out.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


@pytest.mark.parametrize("width", [33, 53, 64, 65, 128, 256])
def test_wide_unsigned_roundtrip_preserves_every_bit(width):
    maximum = (1 << width) - 1
    values = WideCodes((0, 1, (1 << (width - 1)) + 5, maximum), (2, 2))
    payload = pack_unsigned([values], [width])
    assert len(payload) == (4 * width + 7) // 8
    decoded = unpack_unsigned(payload, [(2, 2)], [width])[0]
    assert isinstance(decoded, WideCodes)
    assert decoded.values == values.values and decoded.shape == (2, 2)
    if width == 128:
        assert payload[32:48] == values.values[2].to_bytes(16, "little")
        assert payload[48:64] == bytes([255]) * 16


def test_mixed_nonaligned_wide_fields_and_padding():
    first = torch.tensor([1, 7, 2])
    wide = WideCodes(((1 << 127) + 9, (1 << 128) - 1), (2,))
    last = torch.tensor([15, 1])
    payload = pack_unsigned([first, wide, last], [3, 128, 7], 40)
    decoded = unpack_unsigned(payload, [(3,), (2,), (2,)], [3, 128, 7])
    assert torch.equal(decoded[0], first)
    assert decoded[1].values == wide.values
    assert torch.equal(decoded[2], last)
    assert len(payload) == 40
    corrupted = payload[:-1] + bytes([1])
    with pytest.raises(ValueError, match="padding"):
        unpack_unsigned(corrupted, [(3,), (2,), (2,)], [3, 128, 7])
    with pytest.raises(ValueError, match="truncated"):
        unpack_unsigned(payload[:30], [(3,), (2,), (2,)], [3, 128, 7])


def test_exact_arbitrary_integer_nearest_even():
    from fractions import Fraction
    assert round_ratio_even(1, 2) == 0
    assert round_ratio_even(3, 2) == 2
    assert round_ratio_even(5, 2) == 2
    units = [0., .5, .125, 1., float.fromhex("0x1.0000000000001p-1")]
    for width in (33, 64, 128, 256):
        expected = tuple(round(Fraction.from_float(value) * ((1 << width) - 1)) for value in units)
        assert quantize_unit_values(units, width) == expected


@pytest.mark.parametrize("width", [33, 64, 128, 256])
def test_high_bit_scalar_quantizer_no_overflow_and_ste(width):
    q = ScalarQuantizer(torch.zeros(1), torch.ones(1), torch.ones(1))
    inputs = torch.tensor([[[-1., -.75, 0., .125, 1.]]], dtype=torch.float64, requires_grad=True)
    codes = q.codes(inputs, width)
    assert isinstance(codes, WideCodes)
    assert codes.values[0] == 0
    assert codes.values[-1] == (1 << width) - 1
    assert codes.values[2] == 1 << (width - 1)  # midpoint rounds to even
    decoded = q.from_codes(codes, width, dtype=torch.float64)
    torch.testing.assert_close(decoded, inputs, atol=2.0 / ((1 << width) - 1), rtol=0)
    forwarded = q(inputs, width, straight_through=True)
    torch.testing.assert_close(forwarded, decoded, atol=0, rtol=0)
    forwarded.sum().backward()
    assert torch.equal(inputs.grad, torch.ones_like(inputs))


def test_cns_full_budget_single_derived_128_bit_real_record():
    budget = PayloadBudget(retained_sites=120, primitive_components=4, budget_ratio=1.)
    assert budget.field_bits_per_site == 128
    training = torch.randn(3, 1, 12, 10, dtype=torch.float64)
    quantizer = ScalarQuantizer.fit(training, split="train")
    codec = ExplicitRecordCodec([quantizer], [128], [(1, 12, 10)], budget.cap_bytes)
    payload, ledger = codec.encode([training[0]])
    assert len(payload) == 120 * 16
    assert ledger.field_bits == 120 * 128
    assert ledger.padding_bits == 0
    assert ledger.total_bits == budget.cap_bits == len(payload)*8
    torch.testing.assert_close(codec.decode(payload, dtype=torch.float64)[0],
                               quantizer(training[:1], 128)[0], rtol=0, atol=0)


def test_wide_shape_device_operations_and_range_validation():
    value = WideCodes((0, 1, 2), (1, 3, 1))
    assert value.squeeze(0).unsqueeze(0).shape == value.shape
    assert value.squeeze().shape == (3,)
    assert value.unsqueeze(-1).shape == (1, 3, 1, 1)
    assert value.to("cpu").values == value.values
    with pytest.raises(ValueError, match="alphabet"):
        pack_unsigned([WideCodes((1 << 128,), (1,))], [128])
    with pytest.raises(ValueError, match="exceeds"):
        pack_unsigned([WideCodes((1,), (1,))], [128], 15)
