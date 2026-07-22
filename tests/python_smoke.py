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

    world_positions = positions[:, :3].copy()
    unwrapped = asdiff_render.unwrap_mesh_uv(world_positions, indices, resolution=(64, 64))
    assert unwrapped.positions.shape[1] == 3 and unwrapped.uv.shape[1] == 2
    assert unwrapped.indices.shape == indices.shape
    assert unwrapped.face_chart_ids.shape == (indices.shape[0],)
    parallel_unwrapped = asdiff_render.unwrap_mesh_uv(
        world_positions, indices, resolution=(64, 64), parallel_partitions=2
    )
    assert parallel_unwrapped.indices.shape == indices.shape
    assert parallel_unwrapped.uv.shape[1] == 2
    assert parallel_unwrapped.face_chart_ids.shape == (indices.shape[0],)
    assert parallel_unwrapped.partition_count >= 1
    assert np.isfinite(parallel_unwrapped.uv).all()
    assert asdiff_render.MESH_QUALITY_TRIANGLE_COUNTS == {
        "high": 1_000_000,
        "medium": 500_000,
        "low": 100_000,
    }
    image = np.zeros((16, 16, 4), dtype=np.float32)
    image[..., 0] = 0.75
    image[..., 1] = 0.25
    image[..., 3] = 1.0
    matrix = np.eye(4, dtype=np.float32)[None]
    camera_positions = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)
    baked = asdiff_render.project_texture_atlas(
        unwrapped,
        [image],
        matrix,
        camera_positions,
        resolution=(16, 16),
        visibility_mode="hybrid_ray_query",
    )
    assert baked.color.shape == (16, 16, 4)
    assert baked.valid_mask.any()
    if rasterizer.device_info.supports_ray_query:
        assert baked.used_ray_query
    padded = asdiff_render.pad_texture_atlas(baked.color, baked.valid_mask, 2)
    assert padded.shape == baked.color.shape
    assert np.count_nonzero(padded[..., :3]) >= np.count_nonzero(baked.color[..., :3])

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

    uv_attributes = np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]], dtype=np.float32)
    uv = rasterizer.interpolate_forward(uv_attributes, indices, raster)
    texture_values = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3) / 48.0
    sampled = rasterizer.texture_forward(texture_values, uv, raster)
    assert sampled.shape == (16, 20, 3)
    grad_texture, grad_uv = rasterizer.texture_backward(
        texture_values,
        uv,
        raster,
        np.ones_like(sampled),
    )
    assert grad_texture.shape == texture_values.shape
    assert grad_uv.shape == uv.shape
    assert np.isfinite(grad_texture).all() and np.isfinite(grad_uv).all()
    for address_mode in ("wrap", "mirror"):
        addressed = rasterizer.texture_forward(texture_values, uv + 1.25, raster, address_mode)
        assert np.isfinite(addressed).all()

    viewport_raster, _ = rasterizer.forward(
        positions,
        indices,
        (20, 24),
        viewport=(4.0, 3.0, 16.0, 12.0),
    )
    assert np.count_nonzero(viewport_raster[:3, ..., 3]) == 0
    assert np.count_nonzero(viewport_raster[15:, ..., 3]) == 0

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
        torch_uv_attributes = torch.tensor(uv_attributes, requires_grad=True)
        torch_uv = asdiff_render.interpolate(
            torch_uv_attributes,
            torch_raster,
            torch_indices,
            rasterizer=rasterizer,
        )
        torch_texture = torch.tensor(texture_values, requires_grad=True)
        torch_sampled = asdiff_render.texture(
            torch_texture,
            torch_uv,
            torch_raster,
            rasterizer=rasterizer,
        )
        (torch_image[..., 0].sum() + torch_sampled.square().sum()).backward()
        assert torch_positions.grad is not None
        assert torch.isfinite(torch_positions.grad).all()
        assert torch_attributes.grad is not None
        assert torch.isfinite(torch_attributes.grad).all()
        assert torch_uv_attributes.grad is not None
        assert torch.isfinite(torch_uv_attributes.grad).all()
        assert torch_texture.grad is not None
        assert torch.isfinite(torch_texture.grad).all()

        high_level_render = asdiff_render.render_textured_mesh(
            torch_positions.detach(),
            torch_indices,
            torch_uv_attributes.detach(),
            torch_texture.detach(),
            (16, 20),
            rasterizer=rasterizer,
        )
        assert high_level_render.image.shape == (16, 20, 3)
        robust_loss = asdiff_render.masked_charbonnier_loss(
            high_level_render.image,
            torch_sampled.detach(),
            high_level_render.valid_mask,
        )
        assert torch.isfinite(robust_loss)
        assert torch.isfinite(asdiff_render.atlas_total_variation(torch_texture.detach()))

        options = asdiff_render.AtlasOptimizationOptions(
            steps=2,
            learning_rate=1e-2,
            total_variation_weight=0.0,
            seam_weight=0.0,
            cache_view_rasters=True,
        )
        optimized_texture, history = asdiff_render.optimize_texture_atlas(
            torch_texture.detach() * 0.9,
            torch_positions.detach()[None],
            torch_indices,
            torch_uv_attributes.detach(),
            [torch_sampled.detach()],
            rasterizer=rasterizer,
            options=options,
        )
        assert optimized_texture.shape == torch_texture.shape
        assert len(history) == 2 and np.isfinite(history).all()


if __name__ == "__main__":
    main()
