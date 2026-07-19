"""Repeatable UVAtlas timing utility for a triangle mesh."""

from __future__ import annotations

import argparse
import json
import os
import time

import trimesh

import asdiff_render


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh")
    parser.add_argument("--resolution", type=int, default=4096)
    parser.add_argument("--max-stretch", type=float, default=0.3)
    parser.add_argument("--quality", action="store_true")
    arguments = parser.parse_args()
    mesh = trimesh.load(arguments.mesh, process=False)
    positions = mesh.vertices.astype("float32")
    indices = mesh.faces.astype("uint32")
    begin = time.perf_counter()
    result = asdiff_render.unwrap_uv(
        positions,
        indices,
        resolution=(arguments.resolution, arguments.resolution),
        gutter=4.0,
        max_stretch=arguments.max_stretch,
        quality=arguments.quality,
    )
    print(
        json.dumps(
            {
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "runtime_default"),
                "seconds": time.perf_counter() - begin,
                "input_vertices": int(positions.shape[0]),
                "input_faces": int(indices.shape[0]),
                "output_vertices": int(result.positions.shape[0]),
                "charts": int(result.chart_count),
                "max_stretch": float(result.max_stretch),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
