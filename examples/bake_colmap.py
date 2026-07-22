"""Bake a prepared UV mesh from COLMAP photographs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import asdiff_render


def _linear_to_srgb(image: np.ndarray) -> np.ndarray:
    return np.where(image <= 0.0031308, image * 12.92, 1.055 * np.power(image, 1.0 / 2.4) - 0.055)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_npz", help="NPZ containing positions, normals, uv, and indices")
    parser.add_argument("sparse_path", help="COLMAP text model directory")
    parser.add_argument("images_path")
    parser.add_argument("output_png")
    parser.add_argument("--masks-path")
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--image-stride", type=int, default=1)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument(
        "--output-model",
        help="optional textured .glb or .obj output",
    )
    arguments = parser.parse_args()

    archive = np.load(arguments.mesh_npz)
    face_chart_ids = archive.get(
        "face_chart_ids", np.zeros(archive["indices"].shape[0], dtype=np.uint32)
    )
    chart_count = (
        int(archive["chart_count"])
        if "chart_count" in archive
        else int(face_chart_ids.max() + 1) if face_chart_ids.size else 0
    )
    mesh = asdiff_render.UnwrappedMesh(
        archive["positions"],
        archive["normals"],
        archive["uv"],
        archive["indices"],
        archive.get("vertex_remap", np.arange(archive["positions"].shape[0], dtype=np.uint32)),
        face_chart_ids,
        chart_count,
        float(archive["max_stretch"]) if "max_stretch" in archive else 0.0,
        int(archive["partition_count"]) if "partition_count" in archive else 1,
    )
    projection = asdiff_render.load_colmap_projection(
        arguments.sparse_path,
        arguments.images_path,
        mesh_positions=mesh.positions,
        image_stride=arguments.image_stride,
    )
    masks = None
    if arguments.masks_path:
        masks = asdiff_render.load_projection_masks(arguments.masks_path, projection.image_names)
    baked = asdiff_render.project_texture_atlas(
        mesh,
        projection.images,
        projection.world_to_clip,
        projection.camera_positions,
        visibility_masks=masks,
        resolution=(arguments.resolution, arguments.resolution),
        visibility_mode="ray_query",
        device_index=arguments.device_index,
        padding=arguments.padding,
    )

    linear_rgb = np.clip(baked.color[..., :3], 0.0, 1.0)
    srgb = np.clip(_linear_to_srgb(linear_rgb), 0.0, 1.0)
    output_path = Path(arguments.output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    texture_image = Image.fromarray((srgb * 255.0 + 0.5).astype(np.uint8), "RGB")
    texture_image.save(output_path)
    np.savez_compressed(
        output_path.with_suffix(".npz"),
        color=baked.color,
        confidence=baked.confidence,
        source_view=baked.source_view,
        valid_mask=baked.valid_mask,
    )
    if arguments.output_model:
        try:
            import trimesh
        except ImportError as error:
            raise SystemExit("textured model export requires trimesh") from error
        model_path = Path(arguments.output_model)
        model_format = model_path.suffix.lower()
        if model_format not in (".glb", ".obj"):
            raise SystemExit("--output-model supports the .glb and .obj formats")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # The Vulkan atlas uses array/image coordinates with V increasing from
        # top to bottom. Trimesh's in-memory UV convention is bottom-left (and
        # its glTF exporter performs its own conversion), so normalize the UVs
        # here for both OBJ and GLB exports.
        export_uv = mesh.uv.copy()
        export_uv[:, 1] = 1.0 - export_uv[:, 1]
        visual = trimesh.visual.texture.TextureVisuals(uv=export_uv, image=texture_image)
        textured_mesh = trimesh.Trimesh(
            vertices=mesh.positions,
            faces=mesh.indices,
            vertex_normals=mesh.normals,
            visual=visual,
            process=False,
        )
        if model_format == ".glb":
            textured_mesh.export(model_path, file_type="glb")
        else:
            textured_mesh.export(
                model_path,
                file_type="obj",
                include_normals=True,
                include_texture=True,
            )
    print(
        f"views={len(projection.images)} coverage={float(baked.valid_mask.mean()):.6f} "
        f"ray_query={baked.used_ray_query} output={output_path}"
        + (f" model={arguments.output_model}" if arguments.output_model else "")
    )


if __name__ == "__main__":
    main()
