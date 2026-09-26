"""Configurable 1D/2D predictor backbones for DerivOpt.

Every predictor accepts ``[batch, channels, *space]`` and predicts one
frame on the same grid. This module intentionally does not define ArchMulti or
RolloutMulti: those are experimental methods, not aliases for a backbone.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _spatial_tuple(value: int | Sequence[int], dim: int, name: str) -> tuple[int, ...]:
    values = (value,) * dim if isinstance(value, int) else tuple(value)
    if len(values) != dim or any(not isinstance(v, int) or v < 1 for v in values):
        raise ValueError(f"{name} must contain {dim} positive integer(s), got {value!r}")
    return values


def _conv(dim: int) -> type[nn.Conv1d] | type[nn.Conv2d]:
    if dim not in (1, 2):
        raise ValueError("spatial_dim must be 1 or 2")
    return nn.Conv1d if dim == 1 else nn.Conv2d


class _Predictor(nn.Module):
    def __init__(self, spatial_dim: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        _conv(spatial_dim)
        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be positive")
        self.spatial_dim = spatial_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.architecture_metadata: dict[str, Any] = {}

    def _validate(self, x: Tensor) -> None:
        expected = self.spatial_dim + 2
        if x.ndim != expected or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected [B,{self.in_channels},*space] with {expected} dimensions; "
                f"got {tuple(x.shape)}"
            )
        if any(size < 1 for size in x.shape):
            raise ValueError("Batch, channel and spatial dimensions must be nonempty")
        if not torch.is_floating_point(x):
            raise TypeError("Predictor inputs must be real floating-point tensors")

    def _record(self, name: str, **settings: Any) -> None:
        self.architecture_metadata = {
            "name": name,
            "configuration": "explicit_constructor_settings",
            "spatial_dim": self.spatial_dim,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            **settings,
        }


class _ConvBlock(nn.Module):
    def __init__(self, dim: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        conv = _conv(dim)
        self.layers = nn.Sequential(
            conv(in_channels, out_channels, 3, padding=1),
            nn.GELU(),
            conv(out_channels, out_channels, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class UNet(_Predictor):
    """Encoder/decoder with learned downsampling and concatenated skip paths."""

    def __init__(
        self, spatial_dim: int, in_channels: int, out_channels: int,
        width: int = 16, depth: int = 2,
    ) -> None:
        super().__init__(spatial_dim, in_channels, out_channels)
        if width < 1 or depth < 1:
            raise ValueError("UNet width and depth must be positive")
        conv = _conv(spatial_dim)
        widths = [width * 2**level for level in range(depth + 1)]
        self.stem = _ConvBlock(spatial_dim, in_channels, widths[0])
        self.down = nn.ModuleList(
            nn.Sequential(
                conv(widths[level], widths[level + 1], 3, stride=2, padding=1),
                _ConvBlock(spatial_dim, widths[level + 1], widths[level + 1]),
            )
            for level in range(depth)
        )
        self.up = nn.ModuleList(
            _ConvBlock(spatial_dim, widths[level + 1] + widths[level], widths[level])
            for level in reversed(range(depth))
        )
        self.head = conv(width, out_channels, 1)
        self._record("unet", width=width, depth=depth, padding="zero")

    def forward(self, x: Tensor) -> Tensor:
        self._validate(x)
        x = self.stem(x)
        skips = [x]
        for down in self.down:
            x = down(x)
            skips.append(x)
        mode = "linear" if self.spatial_dim == 1 else "bilinear"
        for up, skip in zip(self.up, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[2:], mode=mode, align_corners=False)
            x = up(torch.cat((x, skip), dim=1))
        return self.head(x)


class SpectralConv(nn.Module):
    """Learnable low-mode Fourier convolution with complex-valued parameters.

    A 2D rFFT stores nonnegative last-axis frequencies, but both signs of the
    first axis. Two disjoint weight blocks cover those signs; mode counts are
    clipped to the actual grid without overlapping positive/negative blocks.
    """

    def __init__(
        self, spatial_dim: int, in_channels: int, out_channels: int,
        modes: int | Sequence[int] = 8,
    ) -> None:
        super().__init__()
        _conv(spatial_dim)
        if in_channels < 1 or out_channels < 1:
            raise ValueError("SpectralConv channel counts must be positive")
        self.spatial_dim = spatial_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = _spatial_tuple(modes, spatial_dim, "modes")
        scale = 1.0 / math.sqrt(in_channels * out_channels)
        shape = (in_channels, out_channels, *self.modes)
        self.weight_positive = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        if spatial_dim == 2:
            self.weight_negative = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        else:
            self.register_parameter("weight_negative", None)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != self.spatial_dim + 2 or x.shape[1] != self.in_channels:
            raise ValueError("SpectralConv received an incompatible input shape")
        spatial_shape = x.shape[2:]
        axes = tuple(range(2, x.ndim))
        spectrum = torch.fft.rfftn(x, dim=axes, norm="ortho")
        output = spectrum.new_zeros((x.shape[0], self.out_channels, *spectrum.shape[2:]))
        # Explicit complex casts also support a model whose real layers use float64.
        positive = self.weight_positive.to(dtype=spectrum.dtype)
        if self.spatial_dim == 1:
            count = min(self.modes[0], spectrum.shape[2])
            output[:, :, :count] = torch.einsum(
                "bim,iom->bom", spectrum[:, :, :count], positive[:, :, :count]
            )
        else:
            nx, ny = spatial_shape
            pos_count = min(self.modes[0], (nx + 1) // 2)
            neg_count = min(self.modes[0], nx // 2)
            last_count = min(self.modes[1], ny // 2 + 1)
            output[:, :, :pos_count, :last_count] = torch.einsum(
                "bixy,ioxy->boxy",
                spectrum[:, :, :pos_count, :last_count],
                positive[:, :, :pos_count, :last_count],
            )
            if neg_count:
                negative = self.weight_negative.to(dtype=spectrum.dtype)
                output[:, :, -neg_count:, :last_count] = torch.einsum(
                    "bixy,ioxy->boxy",
                    spectrum[:, :, -neg_count:, :last_count],
                    negative[:, :, :neg_count, :last_count],
                )
        return torch.fft.irfftn(output, s=spatial_shape, dim=axes, norm="ortho")


class FNO(_Predictor):
    """Fourier neural operator blocks, each with a learned local residual path."""

    def __init__(
        self, spatial_dim: int, in_channels: int, out_channels: int,
        width: int = 16, depth: int = 2, modes: int | Sequence[int] = 8,
    ) -> None:
        super().__init__(spatial_dim, in_channels, out_channels)
        if width < 1 or depth < 1:
            raise ValueError("FNO width and depth must be positive")
        conv = _conv(spatial_dim)
        self.lift = conv(in_channels, width, 1)
        self.spectral_layers = nn.ModuleList(
            SpectralConv(spatial_dim, width, width, modes) for _ in range(depth)
        )
        self.local_layers = nn.ModuleList(conv(width, width, 1) for _ in range(depth))
        self.head = nn.Sequential(conv(width, 2 * width, 1), nn.GELU(), conv(2 * width, out_channels, 1))
        self._record("fno", width=width, depth=depth, modes=self.spectral_layers[0].modes)

    def forward(self, x: Tensor) -> Tensor:
        self._validate(x)
        x = self.lift(x)
        for spectral, local in zip(self.spectral_layers, self.local_layers):
            x = F.gelu(spectral(x) + local(x))
        return self.head(x)


class ConvLSTMCell(nn.Module):
    """Convolutional input, forget, output and candidate gates."""

    def __init__(self, spatial_dim: int, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = _conv(spatial_dim)(in_channels + hidden_channels, 4 * hidden_channels, 3, padding=1)

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, Tensor]:
        if state is None:
            shape = (x.shape[0], self.hidden_channels, *x.shape[2:])
            hidden, cell = x.new_zeros(shape), x.new_zeros(shape)
        else:
            hidden, cell = state
        input_gate, forget_gate, output_gate, candidate = self.gates(torch.cat((x, hidden), dim=1)).chunk(4, dim=1)
        cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(candidate)
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        return hidden, cell


class ConvLSTM(_Predictor):
    """Stacked recurrent convolutional cells with an explicit state lifecycle.

    ``stateful=False`` makes each call a one-frame, zero-initialized sequence.
    Set ``stateful=True`` for temporal recurrence across calls, call
    :meth:`reset_state` before each independent batch/trajectory, and call
    :meth:`detach_state` at truncated-backpropagation boundaries. States are not
    included in a model checkpoint; saved weights start a new trajectory.
    """

    def __init__(
        self, spatial_dim: int, in_channels: int, out_channels: int,
        width: int = 16, depth: int = 2, stateful: bool = False,
    ) -> None:
        super().__init__(spatial_dim, in_channels, out_channels)
        if width < 1 or depth < 1:
            raise ValueError("ConvLSTM width and depth must be positive")
        self.cells = nn.ModuleList(
            ConvLSTMCell(spatial_dim, in_channels if index == 0 else width, width)
            for index in range(depth)
        )
        self.head = _conv(spatial_dim)(width, out_channels, 1)
        self.stateful = stateful
        self._state: list[tuple[Tensor, Tensor]] | None = None
        self._record("convlstm", width=width, depth=depth, stateful=stateful, padding="zero")

    def reset_state(self) -> None:
        self._state = None

    def detach_state(self) -> None:
        if self._state is not None:
            self._state = [(hidden.detach(), cell.detach()) for hidden, cell in self._state]

    def forward(self, x: Tensor) -> Tensor:
        self._validate(x)
        states = self._state if self.stateful else None
        if states is not None:
            previous = states[0][0]
            expected = (x.shape[0], self.cells[0].hidden_channels, *x.shape[2:])
            if previous.shape != expected or previous.device != x.device or previous.dtype != x.dtype:
                raise ValueError("ConvLSTM state shape/device/dtype changed; call reset_state() for a new sequence")
        new_states = []
        for index, cell in enumerate(self.cells):
            hidden, memory = cell(x, None if states is None else states[index])
            new_states.append((hidden, memory))
            x = hidden
        if self.stateful:
            self._state = new_states
        return self.head(x)


def _position_features(grid: tuple[int, ...], width: int, reference: Tensor) -> Tensor:
    """Resolution-independent, fixed Fourier position features on patch centers."""
    coordinates = [
        (torch.arange(size, device=reference.device, dtype=reference.dtype) + 0.5) / size
        for size in grid
    ]
    meshes = torch.meshgrid(*coordinates, indexing="ij")
    count = math.ceil(width / (2 * len(grid)))
    # Bounded log frequencies remain finite for wide embeddings, unlike 2**i.
    frequencies = torch.exp(
        torch.linspace(0.0, math.log(10000.0), count, device=reference.device, dtype=reference.dtype)
    )
    features = []
    for mesh in meshes:
        angle = 2 * math.pi * mesh.reshape(-1, 1) * frequencies
        features.extend((angle.sin(), angle.cos()))
    return torch.cat(features, dim=-1)[:, :width].unsqueeze(0)


class Transformer(_Predictor):
    """Patch-token spatial self-attention with exact patch unrolling/cropping.

    Large inputs must choose an appropriate patch size. Exceeding ``max_tokens``
    raises before attention allocation; all patch tokens are otherwise retained.
    """

    def __init__(
        self, spatial_dim: int, in_channels: int, out_channels: int,
        width: int = 16, depth: int = 2, patch_size: int | Sequence[int] = 4,
        num_heads: int = 4, max_tokens: int = 4096, dropout: float = 0.0,
    ) -> None:
        super().__init__(spatial_dim, in_channels, out_channels)
        if width < 1 or depth < 1 or num_heads < 1 or width % num_heads:
            raise ValueError("Transformer requires positive depth/width/num_heads and width divisible by num_heads")
        if max_tokens < 1 or not 0 <= dropout < 1:
            raise ValueError("max_tokens must be positive and dropout must be in [0, 1)")
        self.patch_size = _spatial_tuple(patch_size, spatial_dim, "patch_size")
        self.max_tokens = max_tokens
        self.patch_embedding = _conv(spatial_dim)(in_channels, width, self.patch_size, stride=self.patch_size)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=num_heads, dim_feedforward=4 * width, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.patch_head = nn.Linear(width, out_channels * math.prod(self.patch_size))
        self._record(
            "transformer", width=width, depth=depth, patch_size=self.patch_size,
            num_heads=num_heads, max_tokens=max_tokens, dropout=dropout,
            positions="fixed_patch_center_fourier_features", padding="zero_right_crop",
        )

    def forward(self, x: Tensor) -> Tensor:
        self._validate(x)
        original_shape = tuple(x.shape[2:])
        grid = tuple((size + patch - 1) // patch for size, patch in zip(original_shape, self.patch_size))
        token_count = math.prod(grid)
        if token_count > self.max_tokens:
            raise ValueError(
                f"Transformer needs {token_count} tokens, exceeding max_tokens={self.max_tokens}; "
                "increase patch_size or explicitly increase max_tokens after checking memory"
            )
        padding = []
        for size, patch in reversed(tuple(zip(original_shape, self.patch_size))):
            padding.extend((0, (-size) % patch))
        x = F.pad(x, padding)
        tokens = self.patch_embedding(x).flatten(2).transpose(1, 2)
        tokens = tokens + _position_features(grid, tokens.shape[-1], tokens)
        patches = self.patch_head(self.encoder(tokens))
        if self.spatial_dim == 1:
            n, = grid
            p, = self.patch_size
            output = patches.reshape(x.shape[0], n, self.out_channels, p).permute(0, 2, 1, 3)
            output = output.reshape(x.shape[0], self.out_channels, n * p)
        else:
            nx, ny = grid
            px, py = self.patch_size
            output = patches.reshape(x.shape[0], nx, ny, self.out_channels, px, py).permute(0, 3, 1, 4, 2, 5)
            output = output.reshape(x.shape[0], self.out_channels, nx * px, ny * py)
        crop = (slice(None), slice(None), *(slice(0, size) for size in original_shape))
        return output[crop]


def make_model(
    name: str,
    spatial_dim: int,
    in_channels: int,
    out_channels: int,
    width: int = 16,
    depth: int = 2,
    *,
    modes: int | Sequence[int] = 8,
    patch_size: int | Sequence[int] = 4,
    num_heads: int = 4,
    max_tokens: int = 4096,
    dropout: float = 0.0,
    stateful: bool = False,
) -> _Predictor:
    """Construct one real backbone; unused family-specific options are ignored.

    ``width`` and ``depth`` always apply. ``modes`` is FNO-specific;
    ``patch_size``, ``num_heads``, ``max_tokens`` and ``dropout`` are
    Transformer-specific; ``stateful`` is ConvLSTM-specific.
    """
    normalized = name.lower().replace("-", "").replace("_", "")
    common = dict(spatial_dim=spatial_dim, in_channels=in_channels, out_channels=out_channels, width=width, depth=depth)
    if normalized == "unet":
        return UNet(**common)
    if normalized == "fno":
        return FNO(**common, modes=modes)
    if normalized == "convlstm":
        return ConvLSTM(**common, stateful=stateful)
    if normalized == "transformer":
        return Transformer(**common, patch_size=patch_size, num_heads=num_heads, max_tokens=max_tokens, dropout=dropout)
    raise ValueError(f"Unknown backbone {name!r}; choose unet, fno, convlstm or transformer")


__all__ = ["UNet", "FNO", "SpectralConv", "ConvLSTM", "ConvLSTMCell", "Transformer", "make_model"]
