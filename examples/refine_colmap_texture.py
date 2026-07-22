"""Refine a projected atlas against COLMAP photographs with seam-aware losses."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as torch_functional

import asdiff_render


def _linear_to_srgb(image: np.ndarray) -> np.ndarray:
    return np.where(image <= 0.0031308, image * 12.92, 1.055 * np.power(image, 1.0 / 2.4) - 0.055)


def _load_mesh(path: Path) -> asdiff_render.UnwrappedMesh:
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


def _dense_seam_pairs(mesh: asdiff_render.UnwrappedMesh, samples_per_edge: int) -> np.ndarray:
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


def _atlas_metadata(
    mesh: asdiff_render.UnwrappedMesh,
    resolution: tuple[int, int],
    rasterizer: asdiff_render.Rasterizer,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = resolution
    atlas_clip = np.empty((mesh.uv.shape[0], 4), dtype=np.float32)
    atlas_clip[:, :2] = mesh.uv * 2.0 - 1.0
    atlas_clip[:, 2] = 0.0
    atlas_clip[:, 3] = 1.0
    raster, _ = rasterizer.forward(atlas_clip, mesh.indices, resolution)
    triangle_id = np.rint(raster[..., 3]).astype(np.int64) - 1
    valid = triangle_id >= 0
    chart_ids = np.full((height, width), -1, dtype=np.int64)
    chart_ids[valid] = mesh.face_chart_ids[triangle_id[valid]]
    return valid, chart_ids


def _erode_masks(masks: tuple[np.ndarray, ...], radius: int) -> list[torch.Tensor]:
    result = []
    for mask in masks:
        value = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))
        if radius:
            inverted = 1.0 - value[None, None]
            value = 1.0 - torch_functional.max_pool2d(
                inverted, kernel_size=radius * 2 + 1, stride=1, padding=radius
            )[0, 0]
        result.append(value)
    return result


def _export_model(mesh: asdiff_render.UnwrappedMesh, texture: Image.Image, path: Path) -> None:
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


def _polish_seams(
    texture: torch.Tensor,
    seam_pairs: torch.Tensor,
    rasterizer: asdiff_render.Rasterizer,
    steps: int,
    learning_rate: float,
) -> tuple[torch.Tensor, list[float]]:
    if steps <= 0 or seam_pairs.shape[0] == 0:
        return texture, []
    parameter = torch.nn.Parameter(texture.detach().clone())
    optimizer = torch.optim.Adam([parameter], lr=learning_rate)
    history = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = asdiff_render.seam_consistency_loss(
            parameter, seam_pairs, rasterizer=rasterizer
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            parameter.clamp_(0.0, 1.0)
        history.append(float(loss.detach()))
    return parameter.detach(), history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_npz")
    parser.add_argument("baked_npz")
    parser.add_argument("sparse_path")
    parser.add_argument("images_path")
    parser.add_argument("output_png")
    parser.add_argument("--masks-path", required=True)
    parser.add_argument("--output-model")
    parser.add_argument("--steps", type=int, default=76)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--seam-weight", type=float, default=0.0)
    parser.add_argument("--tv-weight", type=float, default=0.0)
    parser.add_argument("--prior-weight", type=float, default=0.0)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--mask-erosion", type=int, default=2)
    parser.add_argument("--seam-samples", type=int, default=4)
    parser.add_argument("--seam-polish-steps", type=int, default=30)
    parser.add_argument("--seam-polish-learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--cache-view-rasters",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--image-stride", type=int, default=1)
    parser.add_argument("--device-index", type=int, default=0)
    arguments = parser.parse_args()

    mesh = _load_mesh(Path(arguments.mesh_npz))
    baked_archive = np.load(arguments.baked_npz)
    initial_color = asdiff_render.pad_texture_atlas(
        baked_archive["color"], baked_archive["valid_mask"], arguments.padding
    )[..., :3]
    height, width = initial_color.shape[:2]
    projection = asdiff_render.load_colmap_projection(
        arguments.sparse_path,
        arguments.images_path,
        mesh_positions=mesh.positions,
        image_stride=arguments.image_stride,
    )
    masks = asdiff_render.load_projection_masks(arguments.masks_path, projection.image_names)
    rasterizer = asdiff_render.Rasterizer(device_index=arguments.device_index)
    atlas_valid, chart_ids = _atlas_metadata(mesh, (height, width), rasterizer)
    seam_pairs = _dense_seam_pairs(mesh, arguments.seam_samples)

    homogeneous = np.concatenate(
        (mesh.positions, np.ones((mesh.positions.shape[0], 1), dtype=np.float32)), axis=1
    )
    clip_positions = np.stack(
        [homogeneous @ matrix.T for matrix in projection.world_to_clip]
    ).astype(np.float32)
    options = asdiff_render.AtlasOptimizationOptions(
        steps=arguments.steps,
        learning_rate=arguments.learning_rate,
        total_variation_weight=arguments.tv_weight,
        seam_weight=arguments.seam_weight,
        initial_prior_weight=arguments.prior_weight,
        cache_view_rasters=arguments.cache_view_rasters,
    )
    optimized, history = asdiff_render.optimize_texture_atlas(
        torch.from_numpy(initial_color),
        torch.from_numpy(clip_positions),
        torch.from_numpy(mesh.indices.astype(np.int64)),
        torch.from_numpy(mesh.uv),
        [torch.from_numpy(image) for image in projection.images],
        rasterizer=rasterizer,
        visibility_masks=_erode_masks(masks, arguments.mask_erosion),
        atlas_valid_mask=torch.from_numpy(atlas_valid),
        chart_ids=torch.from_numpy(chart_ids),
        seam_uv_pairs=torch.from_numpy(seam_pairs),
        options=options,
    )
    optimized_rgba = np.concatenate(
        (optimized.numpy(), atlas_valid[..., None].astype(np.float32)), axis=2
    )
    optimized_rgba = asdiff_render.pad_texture_atlas(optimized_rgba, atlas_valid, arguments.padding)
    optimized, seam_history = _polish_seams(
        torch.from_numpy(np.ascontiguousarray(optimized_rgba[..., :3])),
        torch.from_numpy(seam_pairs),
        rasterizer,
        arguments.seam_polish_steps,
        arguments.seam_polish_learning_rate,
    )
    optimized_rgba[..., :3] = optimized.numpy()
    output_path = Path(arguments.output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    srgb = np.clip(_linear_to_srgb(np.clip(optimized_rgba[..., :3], 0.0, 1.0)), 0.0, 1.0)
    texture = Image.fromarray((srgb * 255.0 + 0.5).astype(np.uint8), "RGB")
    texture.save(output_path)
    np.savez_compressed(
        output_path.with_suffix(".npz"),
        color=optimized_rgba,
        valid_mask=atlas_valid,
        history=np.asarray(history, dtype=np.float32),
        seam_history=np.asarray(seam_history, dtype=np.float32),
        seam_pairs=seam_pairs,
    )
    if arguments.output_model:
        _export_model(mesh, texture, Path(arguments.output_model))
    print(
        f"views={len(projection.images)} steps={len(history)} seams={len(seam_pairs)} "
        f"loss={history[0]:.6f}->{history[-1]:.6f} "
        + (f"seam={seam_history[0]:.6f}->{seam_history[-1]:.6f} " if seam_history else "")
        + f"output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
