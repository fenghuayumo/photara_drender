"""PyTorch autograd adapter for the portable Vulkan backend.

Tensor data is staged through host memory so the native extension does not link
against a specific PyTorch ABI. A future zero-copy backend can retain this API.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from ._asdiff_render import Rasterizer


class _RasterizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        positions: torch.Tensor,
        indices: torch.Tensor,
        resolution: Tuple[int, int],
        rasterizer: Rasterizer,
        cull_mode: str,
        output_barycentric_derivatives: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if positions.dtype != torch.float32 or positions.ndim != 2 or positions.shape[1] != 4:
            raise ValueError("positions must be a float32 tensor with shape [vertex_count, 4]")
        if indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError("indices must have shape [triangle_count, 3]")

        positions_numpy = positions.detach().cpu().contiguous().numpy()
        indices_numpy = indices.detach().cpu().contiguous().numpy().astype(np.uint32, copy=False)
        raster_numpy, derivative_numpy = rasterizer.forward(
            positions_numpy,
            indices_numpy,
            resolution,
            cull_mode,
            output_barycentric_derivatives,
        )
        raster = torch.from_numpy(raster_numpy).to(device=positions.device)
        derivatives = torch.from_numpy(derivative_numpy).to(device=positions.device)

        ctx.rasterizer = rasterizer
        ctx.save_for_backward(positions.detach(), indices.detach(), raster.detach())
        ctx.mark_non_differentiable(derivatives)
        return raster, derivatives

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_raster: Optional[torch.Tensor],
        _grad_derivatives: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], None, None, None, None, None]:
        positions, indices, raster = ctx.saved_tensors
        if grad_raster is None:
            return None, None, None, None, None, None

        positions_numpy = positions.cpu().contiguous().numpy()
        indices_numpy = indices.cpu().contiguous().numpy().astype(np.uint32, copy=False)
        raster_numpy = raster.cpu().contiguous().numpy()
        grad_raster_numpy = grad_raster.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        grad_positions_numpy = ctx.rasterizer.backward(
            positions_numpy,
            indices_numpy,
            raster_numpy,
            grad_raster_numpy,
        )
        grad_positions = torch.from_numpy(grad_positions_numpy).to(device=positions.device)
        return grad_positions, None, None, None, None, None


class _InterpolateFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        vertex_attributes: torch.Tensor,
        raster: torch.Tensor,
        indices: torch.Tensor,
        rasterizer: Rasterizer,
    ) -> torch.Tensor:
        if vertex_attributes.dtype != torch.float32 or vertex_attributes.ndim != 2:
            raise ValueError("vertex_attributes must be a float32 tensor with shape [vertex_count, attribute_count]")
        if raster.dtype != torch.float32 or raster.ndim != 3 or raster.shape[2] != 4:
            raise ValueError("raster must be a float32 tensor with shape [height, width, 4]")
        if indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError("indices must have shape [triangle_count, 3]")
        if vertex_attributes.device != raster.device:
            raise ValueError("vertex_attributes and raster must be on the same device")

        attributes_numpy = vertex_attributes.detach().cpu().contiguous().numpy()
        indices_numpy = indices.detach().cpu().contiguous().numpy().astype(np.uint32, copy=False)
        raster_numpy = raster.detach().cpu().contiguous().numpy()
        output_numpy = rasterizer.interpolate_forward(attributes_numpy, indices_numpy, raster_numpy)
        output = torch.from_numpy(output_numpy).to(device=vertex_attributes.device)
        ctx.rasterizer = rasterizer
        ctx.save_for_backward(vertex_attributes.detach(), raster.detach(), indices.detach())
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_interpolated: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        vertex_attributes, raster, indices = ctx.saved_tensors
        attributes_numpy = vertex_attributes.cpu().contiguous().numpy()
        indices_numpy = indices.cpu().contiguous().numpy().astype(np.uint32, copy=False)
        raster_numpy = raster.cpu().contiguous().numpy()
        grad_numpy = grad_interpolated.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        grad_attributes_numpy, grad_raster_numpy = ctx.rasterizer.interpolate_backward(
            attributes_numpy,
            indices_numpy,
            raster_numpy,
            grad_numpy,
        )
        grad_attributes = torch.from_numpy(grad_attributes_numpy).to(device=vertex_attributes.device)
        grad_raster = torch.from_numpy(grad_raster_numpy).to(device=raster.device)
        return grad_attributes, grad_raster, None, None


def rasterize(
    positions: torch.Tensor,
    indices: torch.Tensor,
    resolution: Sequence[int],
    *,
    rasterizer: Optional[Rasterizer] = None,
    device_index: int = 0,
    enable_validation: bool = False,
    cull_mode: str = "none",
    output_barycentric_derivatives: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rasterize clip-space vertices and connect the output to autograd.

    Resolution uses ``(height, width)`` order. Supplying a persistent rasterizer
    reuses its Vulkan device and compute pipelines.
    """

    if len(resolution) != 2 or int(resolution[0]) <= 0 or int(resolution[1]) <= 0:
        raise ValueError("resolution must be (height, width) with positive values")
    active_rasterizer = rasterizer or Rasterizer(device_index, enable_validation)
    normalized_resolution = (int(resolution[0]), int(resolution[1]))
    return _RasterizeFunction.apply(
        positions,
        indices,
        normalized_resolution,
        active_rasterizer,
        cull_mode,
        output_barycentric_derivatives,
    )


def interpolate(
    vertex_attributes: torch.Tensor,
    raster: torch.Tensor,
    indices: torch.Tensor,
    *,
    rasterizer: Rasterizer,
) -> torch.Tensor:
    """Interpolate vertex attributes and propagate gradients to attributes and raster."""

    return _InterpolateFunction.apply(vertex_attributes, raster, indices, rasterizer)
