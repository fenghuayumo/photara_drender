"""Refine a projected atlas against COLMAP photographs with seam-aware losses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageFilter

import photara_drender


def _linear_to_srgb(image: np.ndarray) -> np.ndarray:
    return np.where(image <= 0.0031308, image * 12.92, 1.055 * np.power(image, 1.0 / 2.4) - 0.055)


def _load_mesh(path: Path) -> photara_drender.UnwrappedMesh:
    archive = np.load(path)
    charts = archive.get("face_chart_ids", np.zeros(archive["indices"].shape[0], np.uint32))
    return photara_drender.UnwrappedMesh(
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


def _dense_seam_pairs(mesh: photara_drender.UnwrappedMesh, samples_per_edge: int) -> np.ndarray:
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


def _erode_masks(masks: tuple[np.ndarray, ...], radius: int) -> list[np.ndarray]:
    result = []
    for mask in masks:
        if radius:
            image = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
            mask = np.asarray(image.filter(ImageFilter.MinFilter(radius * 2 + 1)), dtype=np.float32) / 255.0
        result.append(np.ascontiguousarray(mask, dtype=np.float32))
    return result


def _export_model(mesh: photara_drender.UnwrappedMesh, texture: Image.Image, path: Path) -> None:
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


def _masked_reprojection_diagnostics(
    mesh: photara_drender.UnwrappedMesh,
    texture_linear: np.ndarray,
    projection: photara_drender.ColmapProjection,
    masks: list[np.ndarray],
    output_dir: Path,
    view_count: int,
    device_index: int,
) -> None:
    """Save foreground-cropped target/render/error panels and masked metrics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    count = min(max(view_count, 1), len(projection.images))
    selected = np.unique(np.linspace(0, len(projection.images) - 1, count, dtype=np.int64))
    rasterizer = photara_drender.Rasterizer(device_index=device_index)
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

        target_vis = np.clip(_linear_to_srgb(np.clip(target, 0.0, 1.0)), 0.0, 1.0)
        render_vis = np.clip(_linear_to_srgb(np.clip(rendered, 0.0, 1.0)), 0.0, 1.0)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_npz")
    parser.add_argument("baked_npz")
    parser.add_argument("sparse_path")
    parser.add_argument("images_path", help="original COLMAP RGB image directory")
    parser.add_argument("output_png")
    parser.add_argument(
        "--texture-source", choices=("delight", "rgb"), default="delight",
        help="must match the source used by the initial bake (default: delight)",
    )
    parser.add_argument(
        "--delighted-images-path",
        help="Intrinsic-delighted views; defaults to <images_path>_delighted",
    )
    parser.add_argument("--masks-path", required=True)
    parser.add_argument("--output-model")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--minimum-learning-rate", type=float, default=0.00025)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--mask-erosion", type=int, default=2)
    parser.add_argument("--seam-samples", type=int, default=4)
    parser.add_argument("--seam-polish-steps", type=int, default=30)
    parser.add_argument("--seam-polish-learning-rate", type=float, default=0.001)
    parser.add_argument("--image-stride", type=int, default=1)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--diagnostics-dir")
    parser.add_argument("--diagnostic-view-count", type=int, default=3)
    parser.add_argument("--no-diagnostics", action="store_true")
    arguments = parser.parse_args()

    mesh = _load_mesh(Path(arguments.mesh_npz))
    baked_archive = np.load(arguments.baked_npz)
    baked_source = str(baked_archive["texture_source"]) if "texture_source" in baked_archive else "rgb"
    if baked_source != arguments.texture_source:
        raise SystemExit(
            f"initial bake uses {baked_source!r}, but refinement requested "
            f"{arguments.texture_source!r}; projection and optimization must use the same views"
        )
    initial_color = photara_drender.pad_texture_atlas(
        baked_archive["color"], baked_archive["valid_mask"], arguments.padding
    )[..., :3]
    height, width = initial_color.shape[:2]
    try:
        source_images_path = photara_drender.resolve_texture_images_path(
            arguments.images_path,
            texture_source=arguments.texture_source,
            delighted_images_path=arguments.delighted_images_path,
        )
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    projection = photara_drender.load_colmap_projection(
        arguments.sparse_path,
        source_images_path,
        mesh_positions=mesh.positions,
        image_stride=arguments.image_stride,
    )
    masks = photara_drender.load_projection_masks(arguments.masks_path, projection.image_names)
    optimization_masks = _erode_masks(masks, arguments.mask_erosion)
    atlas_valid = np.ascontiguousarray(baked_archive["valid_mask"] > 0.5)
    seam_pairs = _dense_seam_pairs(mesh, arguments.seam_samples)
    baker = photara_drender.TextureBaker(device_index=arguments.device_index)
    start_time = time.perf_counter()
    optimized, history, seam_history, precompute_seconds, optimization_seconds = baker.refine_texture(
        np.ascontiguousarray(initial_color, dtype=np.float32),
        mesh.positions,
        mesh.uv,
        mesh.indices,
        list(projection.images),
        projection.world_to_clip,
        visibility_masks=optimization_masks,
        seam_uv_pairs=seam_pairs,
        steps=arguments.steps,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        minimum_learning_rate=arguments.minimum_learning_rate,
        seam_polish_steps=arguments.seam_polish_steps,
        seam_learning_rate=arguments.seam_polish_learning_rate,
    )
    elapsed_seconds = time.perf_counter() - start_time
    optimized_rgba = np.concatenate(
        (optimized, atlas_valid[..., None].astype(np.float32)), axis=2
    )
    # The input atlas was padded before refinement. Photometric steps leave
    # guard texels untouched and native seam polish intentionally adjusts them,
    # so padding again here would overwrite the seam solution.
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
        texture_source=np.asarray(arguments.texture_source),
        source_images_path=np.asarray(str(source_images_path.resolve())),
        mask_applied=np.asarray(True),
    )
    if arguments.output_model:
        _export_model(mesh, texture, Path(arguments.output_model))
    if not arguments.no_diagnostics:
        diagnostics_dir = (
            Path(arguments.diagnostics_dir)
            if arguments.diagnostics_dir
            else output_path.parent / f"{output_path.stem}_diagnostics"
        )
        _masked_reprojection_diagnostics(
            mesh,
            np.ascontiguousarray(optimized, dtype=np.float32),
            projection,
            optimization_masks,
            diagnostics_dir,
            arguments.diagnostic_view_count,
            arguments.device_index,
        )
    print(
        f"views={len(projection.images)} source={arguments.texture_source} "
        f"steps={len(history)} seams={len(seam_pairs)} "
        f"loss={history[0]:.6f}->{history[-1]:.6f} "
        + (f"seam={seam_history[0]:.6f}->{seam_history[-1]:.6f} " if len(seam_history) else "")
        + f"precompute={precompute_seconds:.3f}s optimize={optimization_seconds:.3f}s "
        + f"elapsed={elapsed_seconds:.3f}s output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
