"""Prepare a scanned mesh for texture baking and save it as NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import asdiff_render


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_mesh")
    parser.add_argument("output_npz")
    parser.add_argument("--quality", choices=("high", "medium", "low"), default="high")
    parser.add_argument("--target-triangle-count", type=int)
    parser.add_argument("--instant-remesh", action="store_true")
    parser.add_argument("--instant-vertex-count", type=int)
    parser.add_argument("--resolution", type=int, default=4096)
    parser.add_argument("--parallel-partitions", type=int, default=4)
    parser.add_argument("--worker-count", type=int, default=0)
    arguments = parser.parse_args()

    options = asdiff_render.MeshPreparationOptions(
        quality=arguments.quality,
        target_triangle_count=arguments.target_triangle_count,
        use_instant_remesh=arguments.instant_remesh,
        instant_vertex_count=arguments.instant_vertex_count,
        atlas_resolution=(arguments.resolution, arguments.resolution),
        atlas_parallel_partitions=arguments.parallel_partitions,
        atlas_worker_count=arguments.worker_count,
    )
    if not asdiff_render.has_mesh_ops_backend():
        raise SystemExit(
            "in-memory mesh ops unavailable; rebuild with ASDIFF_BUILD_MESH_TOOLS=ON and CGAL"
        )
    result = asdiff_render.prepare_mesh_for_baking(arguments.source_mesh, options=options)
    output_path = Path(arguments.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh = result.mesh
    np.savez_compressed(
        output_path,
        positions=mesh.positions,
        normals=mesh.normals,
        uv=mesh.uv,
        indices=mesh.indices,
        vertex_remap=mesh.vertex_remap,
        face_chart_ids=mesh.face_chart_ids,
        chart_count=np.uint32(mesh.chart_count),
        max_stretch=np.float32(mesh.max_stretch),
        partition_count=np.uint32(mesh.partition_count),
    )
    print(
        json.dumps(
            {
                "quality": arguments.quality,
                "source_faces": result.source_face_count,
                "remeshed_faces": result.remeshed_face_count,
                "manifold_faces": result.manifold_face_count,
                "charts": mesh.chart_count,
                "actual_partitions": mesh.partition_count,
                "max_stretch": mesh.max_stretch,
                "remesh_seconds": result.remesh_seconds,
                "cgal_seconds": result.cgal_seconds,
                "uv_atlas_seconds": result.uv_atlas_seconds,
                "output_npz": str(output_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
