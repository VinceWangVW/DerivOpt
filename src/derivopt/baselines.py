"""Architecture-side multiscale baseline for 1D and 2D carried-state predictors."""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


class ArchMulti(nn.Module):
    """Two independent copies of the same backbone, with feature fusion.

    ``factory(in_channels, out_channels)`` must create a fresh backbone module.
    Both branches predict ``width`` features. The fine branch sees the supplied
    simulator-grid input; the coarse branch sees its stride-two average pooling.
    Coarse features are linearly/bilinearly interpolated back and concatenated
    with fine features before learned pointwise fusion. This module only changes
    the model: it neither implements nor bypasses an external carried-state codec.

    For ConvLSTM, only the fine branch retains temporal state. The coarse branch
    is a frame-local multiscale feature path. Thus the persistent hidden/cell
    tensors have the same shapes and memory cost as the single-scale backbone.
    """

    def __init__(self, factory: Callable[[int, int], nn.Module], spatial_dim: int,
                 in_channels: int, out_channels: int, width: int = 16):
        super().__init__()
        if spatial_dim not in (1, 2):
            raise ValueError("ArchMulti spatial_dim must be 1 or 2")
        if any(not isinstance(n, int) or n < 1 for n in (in_channels, out_channels, width)):
            raise ValueError("ArchMulti channel counts and width must be positive integers")
        self.spatial_dim, self.in_channels, self.out_channels = spatial_dim, in_channels, out_channels
        self.width = width
        self.fine_branch = factory(in_channels, width)
        self.coarse_branch = factory(in_channels, width)
        if not isinstance(self.fine_branch, nn.Module) or not isinstance(self.coarse_branch, nn.Module):
            raise TypeError("factory must return torch.nn.Module instances")
        if type(self.fine_branch) is not type(self.coarse_branch):
            raise ValueError("Both ArchMulti branches must use the same backbone class")
        fine_parameters = {id(parameter) for parameter in self.fine_branch.parameters()}
        coarse_parameters = {id(parameter) for parameter in self.coarse_branch.parameters()}
        if self.fine_branch is self.coarse_branch or fine_parameters & coarse_parameters:
            raise ValueError("factory must create independent branch parameters, not reuse a model")
        from .models import ConvLSTM
        if isinstance(self.coarse_branch, ConvLSTM):
            self.coarse_branch.stateful = False
            self.coarse_branch.architecture_metadata["stateful"] = False
        conv = nn.Conv1d if spatial_dim == 1 else nn.Conv2d
        self.fusion = nn.Sequential(conv(2 * width, width, 1), nn.GELU(), conv(width, out_channels, 1))
        self.architecture_metadata = {
            "name": "archmulti", "spatial_dim": spatial_dim, "in_channels": in_channels,
            "out_channels": out_channels, "feature_width": width,
            "downsample": "average_pool_kernel2_stride2_ceil_no_pad_count",
            "upsample": "linear" if spatial_dim == 1 else "bilinear", "align_corners": False,
            "fusion": "concat_pointwise_gelu_pointwise", "branch_parameters": "independent",
            "backbone": dict(getattr(self.fine_branch, "architecture_metadata", {})),
            "persistent_recurrence": "fine_branch_only",
        }

    def reset_state(self) -> None:
        """Start independent trajectories in every recurrent branch."""
        for branch in (self.fine_branch, self.coarse_branch):
            reset = getattr(branch, "reset_state", None)
            if callable(reset):
                reset()

    def detach_state(self) -> None:
        """Detach both recurrent histories at a truncated-backprop boundary."""
        for branch in (self.fine_branch, self.coarse_branch):
            detach = getattr(branch, "detach_state", None)
            if callable(detach):
                detach()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != self.spatial_dim + 2 or inputs.shape[1] != self.in_channels:
            raise ValueError(f"ArchMulti expects [B,{self.in_channels},*space] in {self.spatial_dim}D")
        if not inputs.is_floating_point() or any(n < 1 for n in inputs.shape):
            raise ValueError("ArchMulti expects nonempty real floating inputs")
        pool = F.avg_pool1d if self.spatial_dim == 1 else F.avg_pool2d
        coarse_inputs = pool(inputs, kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False)
        fine_features = self.fine_branch(inputs)
        coarse_features = self.coarse_branch(coarse_inputs)
        expected_fine = (inputs.shape[0], self.width, *inputs.shape[2:])
        expected_coarse = (inputs.shape[0], self.width, *coarse_inputs.shape[2:])
        if tuple(fine_features.shape) != expected_fine or tuple(coarse_features.shape) != expected_coarse:
            raise ValueError("Backbone branches must return width feature channels on their own input grids")
        mode = "linear" if self.spatial_dim == 1 else "bilinear"
        coarse_features = F.interpolate(coarse_features, size=inputs.shape[2:], mode=mode, align_corners=False)
        return self.fusion(torch.cat((fine_features, coarse_features), dim=1))


__all__ = ["ArchMulti"]
