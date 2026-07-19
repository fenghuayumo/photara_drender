import numpy as np

import asdiff_render


def main() -> None:
    rasterizer = asdiff_render.Rasterizer()
    positions = np.array(
        [[-0.75, -0.75, 0.0, 1.0], [0.75, -0.75, 0.0, 1.0], [0.0, 0.75, 0.0, 1.0]],
        dtype=np.float32,
    )
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    raster, derivatives = rasterizer.forward(positions, indices, (16, 20))
    assert raster.shape == (16, 20, 4)
    assert derivatives.shape == raster.shape
    assert np.count_nonzero(raster[..., 3]) > 0
    grad_raster = np.zeros_like(raster)
    grad_raster[..., 0] = 1.0
    grad_positions = rasterizer.backward(positions, indices, raster, grad_raster)
    assert grad_positions.shape == positions.shape
    assert np.isfinite(grad_positions).all()

    attributes = np.eye(3, dtype=np.float32)
    interpolated = rasterizer.interpolate_forward(attributes, indices, raster)
    assert interpolated.shape == (16, 20, 3)
    grad_attributes, grad_interpolation_raster = rasterizer.interpolate_backward(
        attributes,
        indices,
        raster,
        np.ones_like(interpolated),
    )
    assert grad_attributes.shape == attributes.shape
    assert grad_interpolation_raster.shape == raster.shape
    assert np.isfinite(grad_attributes).all()

    if asdiff_render.rasterize is not None:
        import torch

        torch_positions = torch.tensor(positions, requires_grad=True)
        torch_indices = torch.tensor(indices.astype(np.int64))
        torch_raster, _ = asdiff_render.rasterize(
            torch_positions,
            torch_indices,
            (16, 20),
            rasterizer=rasterizer,
        )
        torch_attributes = torch.eye(3, dtype=torch.float32, requires_grad=True)
        torch_image = asdiff_render.interpolate(
            torch_attributes,
            torch_raster,
            torch_indices,
            rasterizer=rasterizer,
        )
        torch_image[..., 0].sum().backward()
        assert torch_positions.grad is not None
        assert torch.isfinite(torch_positions.grad).all()
        assert torch_attributes.grad is not None
        assert torch.isfinite(torch_attributes.grad).all()


if __name__ == "__main__":
    main()
