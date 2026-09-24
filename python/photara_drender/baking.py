"""UV unwrapping and calibrated multi-view texture projection helpers."""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ._photara_drender import TextureBaker, pad_texture_atlas as _pad_texture_atlas, unwrap_uv


class UnwrappedMesh(NamedTuple):
    positions: np.ndarray
    normals: np.ndarray
    uv: np.ndarray
    indices: np.ndarray
    vertex_remap: np.ndarray
    face_chart_ids: np.ndarray
    chart_count: int
    max_stretch: float
    partition_count: int = 1


class BakedTexture(NamedTuple):
    color: np.ndarray
    confidence: np.ndarray
    source_view: np.ndarray
    valid_mask: np.ndarray
    coverage_mask: np.ndarray
    used_ray_query: bool


def pad_texture_atlas(
    color: np.ndarray,
    valid_mask: np.ndarray,
    padding: int,
    *,
    coverage_mask: Optional[np.ndarray] = None,
    fill_unobserved: bool = False,
    maximum_fill_distance: int = 0,
) -> np.ndarray:
    """Extend valid atlas texels into empty neighbors to prevent filtered seam cracks.

    The returned color is a copy. ``valid_mask`` keeps its projection-valid
    meaning and is therefore not modified by guard texels.

    ``padding`` is the guard width in texels. With ``fill_unobserved`` the colour
    of rasterized chart texels that never received a sample (masked out, rejected
    by the facing test, or occluded everywhere) is extrapolated from the nearest
    valid texel instead of staying black; ``coverage_mask`` marks those texels
    and can be omitted when only the guard band is wanted.
    """

    color = np.asarray(color, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if color.ndim != 3 or color.shape[2] not in (3, 4):
        raise ValueError("color must have shape [height, width, 3 or 4]")
    if valid.shape != color.shape[:2]:
        raise ValueError("valid_mask must match the color dimensions")
    if padding < 0:
        raise ValueError("padding must be non-negative")
    if padding == 0 and not fill_unobserved:
        return np.ascontiguousarray(color.copy())
    if not valid.any():
        return np.ascontiguousarray(color.copy())
    coverage = None
    if coverage_mask is not None:
        coverage = np.ascontiguousarray(np.asarray(coverage_mask, dtype=np.float32))
        if coverage.shape != valid.shape:
            raise ValueError("coverage_mask must match the color dimensions")
    if maximum_fill_distance < 0:
        raise ValueError("maximum_fill_distance must be non-negative")
    padded, _filled, _gutter, _unobserved, _remaining = _pad_texture_atlas(
        np.ascontiguousarray(color),
        valid.astype(np.float32),
        int(padding),
        coverage,
        bool(fill_unobserved),
        int(maximum_fill_distance),
    )
    return padded


def _compute_vertex_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(positions, dtype=np.float32)
    triangles = positions[indices]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    for corner in range(3):
        np.add.at(normals, indices[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, np.float32(1e-20))


def unwrap_mesh_uv(
    positions: np.ndarray,
    indices: np.ndarray,
    normals: Optional[np.ndarray] = None,
    *,
    resolution: Tuple[int, int] = (1024, 1024),
    gutter: float = 1.0,
    max_stretch: float = 1.0 / 6.0,
    max_chart_count: int = 0,
    quality: Optional[bool] = None,
    parallel_partitions: int = 1,
    worker_count: int = 0,
) -> UnwrappedMesh:
    """Unwrap a mesh; quality selects UVAtlas quality/fast mode, and None lets UVAtlas choose."""

    positions = np.ascontiguousarray(positions, dtype=np.float32)
    indices = np.ascontiguousarray(indices, dtype=np.uint32)
    if normals is None:
        normals = _compute_vertex_normals(positions, indices)
    else:
        normals = np.ascontiguousarray(normals, dtype=np.float32)
        if normals.shape != positions.shape:
            raise ValueError("normals must have the same shape as positions")
    result = unwrap_uv(
        positions,
        indices,
        resolution=resolution,
        gutter=gutter,
        max_stretch=max_stretch,
        max_chart_count=max_chart_count,
        quality=quality,
        parallel_partitions=parallel_partitions,
        worker_count=worker_count,
    )
    remap = result.vertex_remap
    return UnwrappedMesh(
        result.positions,
        np.ascontiguousarray(normals[remap]),
        result.uv,
        result.indices,
        remap,
        result.face_chart_ids,
        result.chart_count,
        result.max_stretch,
        result.partition_count,
    )


def project_texture_atlas(
    mesh: UnwrappedMesh,
    images: Sequence[np.ndarray],
    world_to_clip: np.ndarray,
    camera_positions: np.ndarray,
    *,
    baker: Optional[TextureBaker] = None,
    device_index: int = 0,
    visibility_masks: Optional[Sequence[Optional[np.ndarray]]] = None,
    viewports: Optional[Sequence[Optional[Sequence[float]]]] = None,
    resolution: Tuple[int, int] = (1024, 1024),
    blend_mode: str = "weighted_average",
    visibility_mode: str = "ray_query",
    pcf_radius: int = 1,
    allow_visibility_fallback: bool = True,
    softmax_scale: float = 0.0,
    mask_floor: float = 0.0,
    padding: int = 0,
    fill_unobserved: bool = False,
    maximum_fill_distance: int = 0,
) -> BakedTexture:
    """Project photographs into atlas space with authoritative ray-query visibility by default.

    ``blend_mode="softmax"`` weights every sample by its pixel footprint on the
    surface (``|n.v| / d^2``) and blends with a scene-relative exponential
    softmax, so the best-resolving view dominates instead of a linear average
    that mixes in grazing and distant samples. ``mask_floor`` keeps masked-out
    pixels as weak samples so geometry that the masks hide but the photographs
    show still receives texture. ``padding`` extends valid texels into the chart
    gutter by that many texels, and ``fill_unobserved`` gives rasterized texels
    that received no sample at all the colour of the nearest valid texel instead
    of leaving them black (flat per-chart fills; opt in deliberately).
    """

    if baker is None:
        baker = TextureBaker(device_index=device_index)
    float_images = [np.ascontiguousarray(image, dtype=np.float32) for image in images]
    float_masks = None
    if visibility_masks is not None:
        float_masks = [
            None if mask is None else np.ascontiguousarray(mask, dtype=np.float32)
            for mask in visibility_masks
        ]
    output = baker.bake(
        mesh.positions,
        mesh.normals,
        mesh.uv,
        mesh.indices,
        float_images,
        np.ascontiguousarray(world_to_clip, dtype=np.float32),
        np.ascontiguousarray(camera_positions, dtype=np.float32),
        visibility_masks=float_masks,
        viewports=viewports,
        resolution=resolution,
        blend_mode=blend_mode,
        visibility_mode=visibility_mode,
        pcf_radius=pcf_radius,
        allow_visibility_fallback=allow_visibility_fallback,
        softmax_scale=float(softmax_scale),
        mask_floor=float(mask_floor),
    )
    baked = BakedTexture(*output)
    if padding or fill_unobserved:
        baked = baked._replace(
            color=pad_texture_atlas(
                baked.color,
                baked.valid_mask,
                padding,
                coverage_mask=baked.coverage_mask,
                fill_unobserved=fill_unobserved,
                maximum_fill_distance=maximum_fill_distance,
            )
        )
    return baked
