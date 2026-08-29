"""PyTorch autograd adapter for the portable Vulkan backend.

Tensor data is staged through host memory so the native extension does not link
against a specific PyTorch ABI. A future zero-copy backend can retain this API.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from ._aether_drender import Rasterizer


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
        viewport: Optional[Tuple[float, float, float, float]],
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
            viewport,
        )
        raster = torch.from_numpy(raster_numpy).to(device=positions.device)
        derivatives = torch.from_numpy(derivative_numpy).to(device=positions.device)

        ctx.rasterizer = rasterizer
        ctx.viewport = viewport
        ctx.save_for_backward(positions.detach(), indices.detach(), raster.detach())
        ctx.mark_non_differentiable(derivatives)
        return raster, derivatives

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_raster: Optional[torch.Tensor],
        _grad_derivatives: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], None, None, None, None, None, None]:
        positions, indices, raster = ctx.saved_tensors
        if grad_raster is None:
            return None, None, None, None, None, None, None

        positions_numpy = positions.cpu().contiguous().numpy()
        indices_numpy = indices.cpu().contiguous().numpy().astype(np.uint32, copy=False)
        raster_numpy = raster.cpu().contiguous().numpy()
        grad_raster_numpy = grad_raster.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        grad_positions_numpy = ctx.rasterizer.backward(
            positions_numpy,
            indices_numpy,
            raster_numpy,
            grad_raster_numpy,
            ctx.viewport,
        )
        grad_positions = torch.from_numpy(grad_positions_numpy).to(device=positions.device)
        return grad_positions, None, None, None, None, None, None


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


class _TextureFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        texture_values: torch.Tensor,
        uv: torch.Tensor,
        raster: torch.Tensor,
        rasterizer: Rasterizer,
        address_mode: str,
    ) -> torch.Tensor:
        if texture_values.dtype != torch.float32 or texture_values.ndim != 3:
            raise ValueError("texture_values must be float32 with shape [texture_height, texture_width, channels]")
        if uv.dtype != torch.float32 or uv.ndim != 3 or uv.shape[2] != 2:
            raise ValueError("uv must be float32 with shape [image_height, image_width, 2]")
        if raster.dtype != torch.float32 or raster.ndim != 3 or raster.shape[2] != 4:
            raise ValueError("raster must be float32 with shape [image_height, image_width, 4]")
        if uv.shape[:2] != raster.shape[:2]:
            raise ValueError("uv and raster image dimensions must match")
        if texture_values.device != uv.device or uv.device != raster.device:
            raise ValueError("texture_values, uv, and raster must be on the same device")

        texture_numpy = texture_values.detach().cpu().contiguous().numpy()
        uv_numpy = uv.detach().cpu().contiguous().numpy()
        raster_numpy = raster.detach().cpu().contiguous().numpy()
        output_numpy = rasterizer.texture_forward(texture_numpy, uv_numpy, raster_numpy, address_mode)
        output = torch.from_numpy(output_numpy).to(device=texture_values.device)
        ctx.rasterizer = rasterizer
        ctx.address_mode = address_mode
        ctx.save_for_backward(texture_values.detach(), uv.detach(), raster.detach())
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_sampled: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, None, None, None]:
        texture_values, uv, raster = ctx.saved_tensors
        texture_numpy = texture_values.cpu().contiguous().numpy()
        uv_numpy = uv.cpu().contiguous().numpy()
        raster_numpy = raster.cpu().contiguous().numpy()
        grad_numpy = grad_sampled.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        compute_uv_gradient = ctx.needs_input_grad[1]
        grad_texture_numpy, grad_uv_numpy = ctx.rasterizer.texture_backward(
            texture_numpy,
            uv_numpy,
            raster_numpy,
            grad_numpy,
            ctx.address_mode,
            compute_uv_gradient,
        )
        grad_texture = torch.from_numpy(grad_texture_numpy).to(device=texture_values.device)
        grad_uv = None if grad_uv_numpy is None else torch.from_numpy(grad_uv_numpy).to(device=uv.device)
        return grad_texture, grad_uv, None, None, None


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
    viewport: Optional[Sequence[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rasterize clip-space vertices and connect the output to autograd.

    Resolution uses ``(height, width)`` order. Supplying a persistent rasterizer
    reuses its Vulkan device and compute pipelines.
    """

    if len(resolution) != 2 or int(resolution[0]) <= 0 or int(resolution[1]) <= 0:
        raise ValueError("resolution must be (height, width) with positive values")
    active_rasterizer = rasterizer or Rasterizer(device_index, enable_validation)
    normalized_resolution = (int(resolution[0]), int(resolution[1]))
    normalized_viewport = None
    if viewport is not None:
        if len(viewport) != 4:
            raise ValueError("viewport must be (x, y, width, height)")
        normalized_viewport = tuple(float(value) for value in viewport)
    return _RasterizeFunction.apply(
        positions,
        indices,
        normalized_resolution,
        active_rasterizer,
        cull_mode,
        output_barycentric_derivatives,
        normalized_viewport,
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


def texture(
    texture_values: torch.Tensor,
    uv: torch.Tensor,
    raster: torch.Tensor,
    *,
    rasterizer: Rasterizer,
    address_mode: str = "clamp",
) -> torch.Tensor:
    """Sample a texture bilinearly and propagate gradients to texels and UV coordinates."""

    return _TextureFunction.apply(texture_values, uv, raster, rasterizer, address_mode)
