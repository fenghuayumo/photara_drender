"""High-level differentiable texture-atlas rendering and optimization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Sequence, Tuple

import torch

from ._aether_drender import Rasterizer
from .torch import interpolate, rasterize, texture as sample_texture


class TexturedRender(NamedTuple):
    image: torch.Tensor
    valid_mask: torch.Tensor
    raster: torch.Tensor
    barycentric_derivatives: torch.Tensor
    uv: torch.Tensor


@dataclass(frozen=True)
class AtlasOptimizationOptions:
    steps: int = 150
    learning_rate: float = 1e-2
    photometric_epsilon: float = 1e-3
    total_variation_weight: float = 1e-4
    seam_weight: float = 1e-3
    initial_prior_weight: float = 0.0
    address_mode: str = "clamp"
    clamp_min: float = 0.0
    clamp_max: float = 1.0
    cache_view_rasters: bool = False


def render_textured_mesh(
    clip_positions: torch.Tensor,
    indices: torch.Tensor,
    vertex_uv: torch.Tensor,
    texture_values: torch.Tensor,
    resolution: Sequence[int],
    *,
    rasterizer: Rasterizer,
    viewport: Optional[Sequence[float]] = None,
    cull_mode: str = "none",
    address_mode: str = "clamp",
) -> TexturedRender:
    """Render a textured mesh and retain intermediate tensors needed by atlas losses."""

    raster, derivatives = rasterize(
        clip_positions,
        indices,
        resolution,
        rasterizer=rasterizer,
        viewport=viewport,
        cull_mode=cull_mode,
        output_barycentric_derivatives=True,
    )
    uv = interpolate(vertex_uv, raster, indices, rasterizer=rasterizer)
    image = sample_texture(texture_values, uv, raster, rasterizer=rasterizer, address_mode=address_mode)
    valid_mask = (raster[..., 3:4] > 0).to(dtype=image.dtype)
    return TexturedRender(image, valid_mask, raster, derivatives, uv)


def masked_charbonnier_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Robust photometric loss normalized by the number of valid projected pixels."""

    if rendered.shape != target.shape:
        raise ValueError("rendered and target must have the same shape")
    if valid_mask.shape != rendered.shape[:-1] + (1,):
        raise ValueError("valid_mask must have shape [height, width, 1]")
    per_channel = torch.sqrt((rendered - target).square() + epsilon * epsilon)
    weighted = per_channel * valid_mask
    denominator = valid_mask.sum().clamp_min(1.0) * rendered.shape[-1]
    return weighted.sum() / denominator


def atlas_total_variation(
    texture_values: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    chart_ids: Optional[torch.Tensor] = None,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Chart-aware robust total variation that does not blur across atlas seams."""

    if texture_values.ndim != 3:
        raise ValueError("texture_values must have shape [height, width, channels]")
    height, width = texture_values.shape[:2]
    if valid_mask is None:
        valid_mask = torch.ones((height, width), dtype=torch.bool, device=texture_values.device)
    else:
        valid_mask = valid_mask.to(device=texture_values.device, dtype=torch.bool)
    if valid_mask.shape != (height, width):
        raise ValueError("valid_mask must have shape [texture_height, texture_width]")
    if chart_ids is not None and chart_ids.shape != (height, width):
        raise ValueError("chart_ids must have shape [texture_height, texture_width]")

    horizontal_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
    vertical_mask = valid_mask[1:, :] & valid_mask[:-1, :]
    if chart_ids is not None:
        chart_ids = chart_ids.to(device=texture_values.device)
        horizontal_mask &= chart_ids[:, 1:] == chart_ids[:, :-1]
        vertical_mask &= chart_ids[1:, :] == chart_ids[:-1, :]
    horizontal = torch.sqrt((texture_values[:, 1:] - texture_values[:, :-1]).square() + epsilon * epsilon)
    vertical = torch.sqrt((texture_values[1:, :] - texture_values[:-1, :]).square() + epsilon * epsilon)
    numerator = (horizontal * horizontal_mask[..., None]).sum() + (vertical * vertical_mask[..., None]).sum()
    pair_count = horizontal_mask.sum() + vertical_mask.sum()
    return numerator / (pair_count.clamp_min(1) * texture_values.shape[-1])


def seam_consistency_loss(
    texture_values: torch.Tensor,
    seam_uv_pairs: torch.Tensor,
    *,
    rasterizer: Rasterizer,
    address_mode: str = "clamp",
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Penalize color disagreement between UV pairs that represent the same mesh seam."""

    if seam_uv_pairs.ndim != 3 or seam_uv_pairs.shape[1:] != (2, 2):
        raise ValueError("seam_uv_pairs must have shape [seam_count, 2, 2]")
    seam_count = seam_uv_pairs.shape[0]
    if seam_count == 0:
        return texture_values.sum() * 0.0
    uv = seam_uv_pairs.reshape(1, seam_count * 2, 2).contiguous()
    fake_raster = torch.zeros((1, seam_count * 2, 4), dtype=torch.float32, device=texture_values.device)
    fake_raster[..., 3] = 1.0
    colors = sample_texture(
        texture_values,
        uv,
        fake_raster,
        rasterizer=rasterizer,
        address_mode=address_mode,
    ).reshape(seam_count, 2, texture_values.shape[-1])
    difference = colors[:, 0] - colors[:, 1]
    return torch.sqrt(difference.square() + epsilon * epsilon).mean()


def optimize_texture_atlas(
    initial_texture: torch.Tensor,
    clip_positions_by_view: torch.Tensor,
    indices: torch.Tensor,
    vertex_uv: torch.Tensor,
    target_images: Sequence[torch.Tensor],
    *,
    rasterizer: Rasterizer,
    viewports: Optional[Sequence[Optional[Sequence[float]]]] = None,
    visibility_masks: Optional[Sequence[Optional[torch.Tensor]]] = None,
    atlas_valid_mask: Optional[torch.Tensor] = None,
    chart_ids: Optional[torch.Tensor] = None,
    seam_uv_pairs: Optional[torch.Tensor] = None,
    options: AtlasOptimizationOptions = AtlasOptimizationOptions(),
) -> Tuple[torch.Tensor, List[float]]:
    """Optimize atlas texels against multiple calibrated views.

    ``clip_positions_by_view`` has shape ``[view_count, vertex_count, 4]``.
    Each optimization step uses one view in deterministic round-robin order.
    """

    if clip_positions_by_view.ndim != 3 or clip_positions_by_view.shape[2] != 4:
        raise ValueError("clip_positions_by_view must have shape [view_count, vertex_count, 4]")
    view_count = clip_positions_by_view.shape[0]
    if len(target_images) != view_count or view_count == 0:
        raise ValueError("target_images must contain one image per view")
    if viewports is None:
        viewports = [None] * view_count
    if visibility_masks is None:
        visibility_masks = [None] * view_count
    if len(viewports) != view_count or len(visibility_masks) != view_count:
        raise ValueError("viewports and visibility_masks must match the view count")
    if options.steps <= 0 or options.learning_rate <= 0:
        raise ValueError("optimization steps and learning_rate must be positive")

    texture_parameter = torch.nn.Parameter(initial_texture.detach().clone())
    initial_reference = initial_texture.detach().clone()
    optimizer = torch.optim.Adam([texture_parameter], lr=options.learning_rate)
    history: List[float] = []
    view_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for step in range(options.steps):
        optimizer.zero_grad(set_to_none=True)
        view_index = step % view_count
        target = target_images[view_index]
        cached = view_cache.get(view_index)
        if cached is None:
            render = render_textured_mesh(
                clip_positions_by_view[view_index],
                indices,
                vertex_uv,
                texture_parameter,
                target.shape[:2],
                rasterizer=rasterizer,
                viewport=viewports[view_index],
                address_mode=options.address_mode,
            )
            image = render.image
            valid_mask = render.valid_mask
            if options.cache_view_rasters:
                view_cache[view_index] = (
                    render.raster.detach(),
                    render.uv.detach(),
                    render.valid_mask.detach(),
                )
        else:
            cached_raster, cached_uv, valid_mask = cached
            image = sample_texture(
                texture_parameter,
                cached_uv,
                cached_raster,
                rasterizer=rasterizer,
                address_mode=options.address_mode,
            )
        external_mask = visibility_masks[view_index]
        if external_mask is not None:
            if external_mask.ndim == 2:
                external_mask = external_mask[..., None]
            valid_mask = valid_mask * external_mask.to(device=valid_mask.device, dtype=valid_mask.dtype)
        loss = masked_charbonnier_loss(
            image,
            target,
            valid_mask,
            epsilon=options.photometric_epsilon,
        )
        if options.total_variation_weight != 0:
            loss = loss + options.total_variation_weight * atlas_total_variation(
                texture_parameter,
                valid_mask=atlas_valid_mask,
                chart_ids=chart_ids,
            )
        if options.seam_weight != 0 and seam_uv_pairs is not None:
            loss = loss + options.seam_weight * seam_consistency_loss(
                texture_parameter,
                seam_uv_pairs,
                rasterizer=rasterizer,
                address_mode=options.address_mode,
            )
        if options.initial_prior_weight != 0:
            prior_difference = (texture_parameter - initial_reference).square()
            if atlas_valid_mask is not None:
                prior_difference = prior_difference * atlas_valid_mask.to(
                    device=prior_difference.device,
                    dtype=prior_difference.dtype,
                )[..., None]
            loss = loss + options.initial_prior_weight * prior_difference.mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            texture_parameter.clamp_(options.clamp_min, options.clamp_max)
        history.append(float(loss.detach().cpu()))
    return texture_parameter.detach(), history
