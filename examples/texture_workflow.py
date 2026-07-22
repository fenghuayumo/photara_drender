"""Shared helpers for the COLMAP texture-baking examples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import asdiff_render


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    return np.where(image <= 0.0031308, image * 12.92, 1.055 * np.power(image, 1.0 / 2.4) - 0.055)


def load_mesh(path: Path) -> asdiff_render.UnwrappedMesh:
    archive = np.load(path)
    charts = archive.get("face_chart_ids", np.zeros(archive["indices"].shape[0], np.uint32))
    return asdiff_render.UnwrappedMesh(
        archive["positions"],
        archive["normals"],
        archive["uv"],
        archive["indices"],
        archive.get("vertex_remap", np.arange(archive["positions"].shape[0], dtype=np.uint32)),
        charts,
        int(archive["chart_count"]) if "chart_count" in archive else int(charts.max() + 1),
        float(archive["max_stretch"]) if "max_stretch" in archive else 0.0,
        int(archive["partition_count"]) if "partition_count" in archive else 1,
    )


def dense_seam_pairs(mesh: asdiff_render.UnwrappedMesh, samples_per_edge: int) -> np.ndarray:
    faces = mesh.indices
    remap = mesh.vertex_remap
    keys = []
    low_uv = []
    high_uv = []
    charts = []
    for first, second in ((0, 1), (1, 2), (2, 0)):
        first_vertex = faces[:, first]
        second_vertex = faces[:, second]
        first_original = remap[first_vertex]
        second_original = remap[second_vertex]
        swap = first_original > second_original
        low = np.minimum(first_original, second_original).astype(np.uint64)
        high = np.maximum(first_original, second_original).astype(np.uint64)
        keys.append((low << np.uint64(32)) | high)
        low_uv.append(np.where(swap[:, None], mesh.uv[second_vertex], mesh.uv[first_vertex]))
        high_uv.append(np.where(swap[:, None], mesh.uv[first_vertex], mesh.uv[second_vertex]))
        charts.append(mesh.face_chart_ids)
    key = np.concatenate(keys)
    low_uv_values = np.concatenate(low_uv)
    high_uv_values = np.concatenate(high_uv)
    chart_values = np.concatenate(charts)
    order = np.argsort(key)
    key = key[order]
    low_uv_values = low_uv_values[order]
    high_uv_values = high_uv_values[order]
    chart_values = chart_values[order]
    starts = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1]
    ends = np.r_[starts[1:], key.size]
    starts = starts[(ends - starts) == 2]
    starts = starts[chart_values[starts] != chart_values[starts + 1]]
    if starts.size == 0:
        return np.empty((0, 2, 2), dtype=np.float32)
    interpolation = ((np.arange(samples_per_edge, dtype=np.float32) + 0.5) / samples_per_edge)[None, :, None]
    first = low_uv_values[starts, None] * (1.0 - interpolation) + high_uv_values[starts, None] * interpolation
    second = low_uv_values[starts + 1, None] * (1.0 - interpolation) + high_uv_values[starts + 1, None] * interpolation
    return np.ascontiguousarray(np.stack((first, second), axis=2).reshape(-1, 2, 2), dtype=np.float32)


def erode_masks(masks: tuple[np.ndarray, ...], radius: int) -> list[np.ndarray]:
    result = []
    for mask in masks:
        if radius:
            image = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
            mask = np.asarray(image.filter(ImageFilter.MinFilter(radius * 2 + 1)), dtype=np.float32) / 255.0
        result.append(np.ascontiguousarray(mask, dtype=np.float32))
    return result


def export_model(mesh: asdiff_render.UnwrappedMesh, texture: Image.Image, path: Path) -> None:
    import trimesh

    export_uv = mesh.uv.copy()
    export_uv[:, 1] = 1.0 - export_uv[:, 1]
    visual = trimesh.visual.texture.TextureVisuals(uv=export_uv, image=texture)
    model = trimesh.Trimesh(
        vertices=mesh.positions,
        faces=mesh.indices,
        vertex_normals=mesh.normals,
        visual=visual,
        process=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".glb":
        model.export(path, file_type="glb")
    elif path.suffix.lower() == ".obj":
        model.export(path, file_type="obj", include_normals=True, include_texture=True)
    else:
        raise ValueError("output model must use .glb or .obj")


def masked_reprojection_diagnostics(
    mesh: asdiff_render.UnwrappedMesh,
    texture_linear: np.ndarray,
    projection: asdiff_render.ColmapProjection,
    masks: list[np.ndarray],
    output_dir: Path,
    view_count: int,
    device_index: int,
) -> None:
    """Save foreground-cropped target/render/error panels and masked metrics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    count = min(max(view_count, 1), len(projection.images))
    selected = np.unique(np.linspace(0, len(projection.images) - 1, count, dtype=np.int64))
    rasterizer = asdiff_render.Rasterizer(device_index=device_index)
    homogeneous = np.concatenate(
        (mesh.positions, np.ones((mesh.positions.shape[0], 1), dtype=np.float32)), axis=1
    )
    metrics = []
    for view_index in selected:
        target = projection.images[int(view_index)]
        height, width = target.shape[:2]
        clip = np.ascontiguousarray(
            homogeneous @ projection.world_to_clip[int(view_index)].T, dtype=np.float32
        )
        raster, _ = rasterizer.forward(
            clip, mesh.indices, (height, width), output_barycentric_derivatives=False
        )
        uv = rasterizer.interpolate_forward(mesh.uv, mesh.indices, raster)
        rendered = rasterizer.texture_forward(texture_linear, uv, raster)
        valid = (raster[..., 3] != 0.0) & (masks[int(view_index)] > 0.5)
        if not valid.any():
            continue
        difference = np.abs(rendered - target)
        values = difference[valid]
        mse = float(np.mean(np.square(rendered[valid] - target[valid])))
        ys, xs = np.nonzero(valid)
        margin = max(8, int(0.03 * max(height, width)))
        y0, y1 = max(0, int(ys.min()) - margin), min(height, int(ys.max()) + margin + 1)
        x0, x1 = max(0, int(xs.min()) - margin), min(width, int(xs.max()) + margin + 1)

        target_vis = np.clip(linear_to_srgb(np.clip(target, 0.0, 1.0)), 0.0, 1.0)
        render_vis = np.clip(linear_to_srgb(np.clip(rendered, 0.0, 1.0)), 0.0, 1.0)
        target_vis[~valid] = 0.0
        render_vis[~valid] = 0.0
        error = np.zeros_like(target_vis)
        error_level = np.clip(difference.mean(axis=2) * 5.0, 0.0, 1.0)
        error[..., 0] = error_level
        error[..., 1] = np.sqrt(error_level) * 0.8
        error[~valid] = 0.0
        panel = np.concatenate(
            (target_vis[y0:y1, x0:x1], render_vis[y0:y1, x0:x1], error[y0:y1, x0:x1]),
            axis=1,
        )
        image_name = Path(projection.image_names[int(view_index)]).stem
        Image.fromarray((panel * 255.0 + 0.5).astype(np.uint8), "RGB").save(
            output_dir / f"{int(view_index):04d}_{image_name}_masked_diff.png"
        )
        metrics.append(
            {
                "view_index": int(view_index),
                "image_name": projection.image_names[int(view_index)],
                "foreground_pixels": int(valid.sum()),
                "masked_mae_linear": float(values.mean()),
                "masked_rmse_linear": float(np.sqrt(mse)),
            }
        )
    with (output_dir / "masked_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump({"panels": "target | render | absolute error", "views": metrics}, stream, indent=2)
