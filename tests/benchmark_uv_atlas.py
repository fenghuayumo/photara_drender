"""Repeatable UVAtlas timing utility for a triangle mesh."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import trimesh

import aether_drender


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh")
    parser.add_argument("--resolution", type=int, default=4096)
    parser.add_argument("--gutter", type=float, default=1.0)
    parser.add_argument("--max-stretch", type=float, default=1.0 / 6.0)
    quality_group = parser.add_mutually_exclusive_group()
    quality_group.add_argument("--fast", action="store_true", help="Force fast geodesic partitioning")
    quality_group.add_argument("--quality", action="store_true", help="Force quality geodesic partitioning")
    parser.add_argument("--parallel-partitions", type=int, default=1)
    parser.add_argument("--worker-count", type=int, default=0)
    parser.add_argument("--output-npz", help="Optionally save the unwrapped mesh for texture baking")
    arguments = parser.parse_args()
    quality = True if arguments.quality else False if arguments.fast else None
    mesh = trimesh.load(arguments.mesh, process=False)
    positions = mesh.vertices.astype("float32")
    indices = mesh.faces.astype("uint32")
    normals = np.ascontiguousarray(mesh.vertex_normals, dtype=np.float32)
    begin = time.perf_counter()
    result = aether_drender.unwrap_uv(
        positions,
        indices,
        resolution=(arguments.resolution, arguments.resolution),
        gutter=arguments.gutter,
        max_stretch=arguments.max_stretch,
        quality=quality,
        parallel_partitions=arguments.parallel_partitions,
        worker_count=arguments.worker_count,
    )
    elapsed = time.perf_counter() - begin
    if arguments.output_npz:
        output_path = Path(arguments.output_npz)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            positions=result.positions,
            normals=normals[result.vertex_remap],
            uv=result.uv,
            indices=result.indices,
            vertex_remap=result.vertex_remap,
            face_chart_ids=result.face_chart_ids,
            chart_count=np.uint32(result.chart_count),
            max_stretch=np.float32(result.max_stretch),
            partition_count=np.uint32(result.partition_count),
        )
    print(
        json.dumps(
            {
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "runtime_default"),
                "seconds": elapsed,
                "input_vertices": int(positions.shape[0]),
                "input_faces": int(indices.shape[0]),
                "output_vertices": int(result.positions.shape[0]),
                "charts": int(result.chart_count),
                "max_stretch": float(result.max_stretch),
                "gutter": arguments.gutter,
                "quality": "default" if quality is None else "quality" if quality else "fast",
                "requested_partitions": arguments.parallel_partitions,
                "actual_partitions": int(result.partition_count),
                "worker_count": arguments.worker_count,
                "output_npz": arguments.output_npz,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
