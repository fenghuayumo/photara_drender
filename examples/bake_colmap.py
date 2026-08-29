"""Bake, refine in memory with native C++/Vulkan, then export once."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
from PIL import Image

import aether_drender
from texture_workflow import (
    dense_seam_pairs,
    erode_masks,
    export_model,
    linear_to_srgb,
    load_mesh,
    masked_reprojection_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_npz", help="NPZ containing positions, normals, uv, and indices")
    parser.add_argument("sparse_path", help="COLMAP text model directory")
    parser.add_argument("images_path", help="original COLMAP RGB image directory")
    parser.add_argument("output_png")
    parser.add_argument(
        "--texture-source", choices=("delight", "rgb"), default="delight",
        help="view set used by both projection and later refinement (default: delight)",
    )
    parser.add_argument(
        "--delighted-images-path",
        help="Intrinsic-delighted views; defaults to <images_path>_delighted",
    )
    parser.add_argument("--masks-path")
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--image-stride", type=int, default=1)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument(
        "--no-optimize", action="store_true",
        help="export the initial projection instead of running native C++ texture refinement",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--minimum-learning-rate", type=float, default=0.00025)
    parser.add_argument("--mask-erosion", type=int, default=2)
    parser.add_argument("--seam-samples", type=int, default=4)
    parser.add_argument("--seam-polish-steps", type=int, default=30)
    parser.add_argument("--seam-polish-learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--save-initial-bake",
        help="optional NPZ checkpoint; the initial projection is otherwise kept only in memory",
    )
    parser.add_argument("--diagnostics-dir")
    parser.add_argument("--diagnostic-view-count", type=int, default=3)
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument(
        "--output-model",
        help="optional textured .glb or .obj output",
    )
    arguments = parser.parse_args()

    if not arguments.no_optimize and not arguments.masks_path:
        raise SystemExit("native texture refinement requires --masks-path so background pixels are excluded")

    mesh = load_mesh(Path(arguments.mesh_npz))
    try:
        source_images_path = aether_drender.resolve_texture_images_path(
            arguments.images_path,
            texture_source=arguments.texture_source,
            delighted_images_path=arguments.delighted_images_path,
        )
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    projection = aether_drender.load_colmap_projection(
        arguments.sparse_path,
        source_images_path,
        mesh_positions=mesh.positions,
        image_stride=arguments.image_stride,
    )
    masks = None
    if arguments.masks_path:
        masks = aether_drender.load_projection_masks(arguments.masks_path, projection.image_names)
    baker = aether_drender.TextureBaker(device_index=arguments.device_index)
    start_time = time.perf_counter()
    baked = aether_drender.project_texture_atlas(
        mesh,
        projection.images,
        projection.world_to_clip,
        projection.camera_positions,
        visibility_masks=masks,
        resolution=(arguments.resolution, arguments.resolution),
        visibility_mode="ray_query",
        baker=baker,
        padding=arguments.padding,
    )
    if arguments.save_initial_bake:
        initial_path = Path(arguments.save_initial_bake)
        initial_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            initial_path,
            color=baked.color,
            confidence=baked.confidence,
            source_view=baked.source_view,
            valid_mask=baked.valid_mask,
            texture_source=np.asarray(arguments.texture_source),
            source_images_path=np.asarray(str(source_images_path.resolve())),
            mask_applied=np.asarray(masks is not None),
            visibility_mode=np.asarray("ray_query"),
            used_ray_query=np.asarray(baked.used_ray_query),
        )

    history = np.empty(0, dtype=np.float32)
    seam_history = np.empty(0, dtype=np.float32)
    seam_pairs = np.empty((0, 2, 2), dtype=np.float32)
    precompute_seconds = 0.0
    optimization_seconds = 0.0
    diagnostic_masks = list(masks) if masks is not None else None
    final_color = baked.color
    if not arguments.no_optimize:
        assert masks is not None
        diagnostic_masks = erode_masks(masks, arguments.mask_erosion)
        seam_pairs = dense_seam_pairs(mesh, arguments.seam_samples)
        optimized, history, seam_history, precompute_seconds, optimization_seconds = baker.refine_texture(
            np.ascontiguousarray(baked.color[..., :3], dtype=np.float32),
            mesh.positions,
            mesh.uv,
            mesh.indices,
            list(projection.images),
            projection.world_to_clip,
            visibility_masks=diagnostic_masks,
            seam_uv_pairs=seam_pairs,
            steps=arguments.steps,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            minimum_learning_rate=arguments.minimum_learning_rate,
            seam_polish_steps=arguments.seam_polish_steps,
            seam_learning_rate=arguments.seam_polish_learning_rate,
        )
        final_color = np.concatenate(
            (optimized, (baked.valid_mask > 0.5)[..., None].astype(np.float32)), axis=2
        )

    linear_rgb = np.clip(final_color[..., :3], 0.0, 1.0)
    srgb = np.clip(linear_to_srgb(linear_rgb), 0.0, 1.0)
    output_path = Path(arguments.output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    texture_image = Image.fromarray((srgb * 255.0 + 0.5).astype(np.uint8), "RGB")
    texture_image.save(output_path)
    np.savez_compressed(
        output_path.with_suffix(".npz"),
        color=final_color,
        confidence=baked.confidence,
        source_view=baked.source_view,
        valid_mask=baked.valid_mask,
        texture_source=np.asarray(arguments.texture_source),
        source_images_path=np.asarray(str(source_images_path.resolve())),
        mask_applied=np.asarray(masks is not None),
        visibility_mode=np.asarray("ray_query"),
        used_ray_query=np.asarray(baked.used_ray_query),
        optimized=np.asarray(not arguments.no_optimize),
        history=np.asarray(history, dtype=np.float32),
        seam_history=np.asarray(seam_history, dtype=np.float32),
        seam_pairs=seam_pairs,
    )
    if arguments.output_model:
        try:
            export_model(mesh, texture_image, Path(arguments.output_model))
        except ImportError as error:
            raise SystemExit("textured model export requires trimesh") from error
    if not arguments.no_diagnostics and diagnostic_masks is not None:
        diagnostics_dir = (
            Path(arguments.diagnostics_dir)
            if arguments.diagnostics_dir
            else output_path.parent / f"{output_path.stem}_diagnostics"
        )
        masked_reprojection_diagnostics(
            mesh,
            np.ascontiguousarray(final_color[..., :3], dtype=np.float32),
            projection,
            diagnostic_masks,
            diagnostics_dir,
            arguments.diagnostic_view_count,
            arguments.device_index,
        )
    elapsed_seconds = time.perf_counter() - start_time
    print(
        f"views={len(projection.images)} coverage={float(baked.valid_mask.mean()):.6f} "
        f"ray_query={baked.used_ray_query} source={arguments.texture_source} "
        f"optimized={not arguments.no_optimize}"
        + (
            f" loss={history[0]:.6f}->{history[-1]:.6f} "
            f"seam={seam_history[0]:.6f}->{seam_history[-1]:.6f} "
            f"precompute={precompute_seconds:.3f}s optimize={optimization_seconds:.3f}s"
            if len(history) and len(seam_history) else ""
        )
        + f" elapsed={elapsed_seconds:.3f}s output={output_path}"
        + (f" model={arguments.output_model}" if arguments.output_model else "")
    )


if __name__ == "__main__":
    main()
