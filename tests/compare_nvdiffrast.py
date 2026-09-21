"""Optional numerical comparison against a local nvdiffrast checkout."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from photara_drender import Rasterizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvdiffrast-root", required=True)
    parser.add_argument("--check-backward", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, args.nvdiffrast_root)
    import nvdiffrast.torch as dr

    positions_numpy = np.array(
        [[-0.75, -0.75, 0.1, 1.0], [0.75, -0.75, 0.2, 1.0], [0.0, 0.75, 0.3, 1.0]],
        dtype=np.float32,
    )
    indices_numpy = np.array([[0, 1, 2]], dtype=np.uint32)
    raster, derivatives = Rasterizer().forward(positions_numpy, indices_numpy, (64, 64))

    cuda_context = dr.RasterizeCudaContext()
    positions = torch.from_numpy(positions_numpy).cuda()[None].requires_grad_(True)
    indices = torch.from_numpy(indices_numpy.astype(np.int32)).cuda()
    reference, reference_derivatives = dr.rasterize(
        cuda_context,
        positions,
        indices,
        (64, 64),
        grad_db=False,
    )
    reference_tensor = reference
    reference = reference[0].detach().cpu().numpy()
    reference_derivatives = reference_derivatives[0].detach().cpu().numpy()
    raster_mask = raster[..., 3] > 0
    reference_mask = reference[..., 3] > 0
    common_mask = raster_mask & reference_mask
    coverage_xor = np.logical_xor(raster_mask, reference_mask).sum()
    uv_error = np.abs(raster[common_mask, :2] - reference[common_mask, :2])
    derivative_error = np.abs(derivatives[common_mask] - reference_derivatives[common_mask])
    depth_error = np.abs(raster[common_mask, 2] - reference[common_mask, 2])
    random_generator = np.random.default_rng(7)
    output_gradient = random_generator.standard_normal(raster.shape, dtype=np.float32)
    position_gradient = Rasterizer().backward(
        positions_numpy,
        indices_numpy,
        raster,
        output_gradient,
    )
    reference_tensor.backward(torch.from_numpy(output_gradient).cuda()[None])
    reference_position_gradient = positions.grad[0].detach().cpu().numpy()
    position_gradient_error = np.abs(position_gradient - reference_position_gradient)
    print(f"coverage_xor={coverage_xor}")
    print(f"uv_max={uv_error.max():.8g} uv_mean={uv_error.mean():.8g}")
    print(f"derivative_max={derivative_error.max():.8g} derivative_mean={derivative_error.mean():.8g}")
    print(f"depth_max={depth_error.max():.8g}")
    print(
        f"position_gradient_max={position_gradient_error.max():.8g} "
        f"position_gradient_mean={position_gradient_error.mean():.8g}"
    )
    if (
        coverage_xor > 2
        or uv_error.max() > 2e-5
        or derivative_error.max() > 2e-5
        or depth_error.max() > 2e-5
        or (args.check_backward and position_gradient_error.max() > 2e-4)
    ):
        raise SystemExit("nvdiffrast comparison exceeded tolerance")


if __name__ == "__main__":
    main()
