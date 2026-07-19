struct PushConstants
{
    uint vertex_count;
    uint triangle_count;
    uint width;
    uint height;
    uint cull_mode;
    uint output_derivatives;
    uint tile_count_x;
    uint tile_size;
};

[[vk::binding(0, 0)]] StructuredBuffer<float4> positions;
[[vk::binding(1, 0)]] StructuredBuffer<uint> indices;
[[vk::binding(2, 0)]] StructuredBuffer<uint> tile_offsets;
[[vk::binding(3, 0)]] StructuredBuffer<uint> tile_triangle_indices;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> raster;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> barycentric_derivatives;
[[vk::push_constant]] ConstantBuffer<PushConstants> push_constants;

float cross_2d(float2 a, float2 b)
{
    return a.x * b.y - a.y * b.x;
}

void store_raster(uint pixel_index, float4 value)
{
    const uint offset = pixel_index * 4;
    raster[offset + 0] = value.x;
    raster[offset + 1] = value.y;
    raster[offset + 2] = value.z;
    raster[offset + 3] = value.w;
}

void store_derivatives(uint pixel_index, float4 value)
{
    const uint offset = pixel_index * 4;
    barycentric_derivatives[offset + 0] = value.x;
    barycentric_derivatives[offset + 1] = value.y;
    barycentric_derivatives[offset + 2] = value.z;
    barycentric_derivatives[offset + 3] = value.w;
}

[numthreads(8, 8, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    const uint2 pixel = dispatch_thread_id.xy;
    if (pixel.x >= push_constants.width || pixel.y >= push_constants.height)
    {
        return;
    }

    const uint pixel_index = pixel.y * push_constants.width + pixel.x;
    const float2 sample_ndc = float2(
        (float(pixel.x) + 0.5f) * 2.0f / float(push_constants.width) - 1.0f,
        (float(pixel.y) + 0.5f) * 2.0f / float(push_constants.height) - 1.0f);
    float best_depth = 3.402823466e+38f;
    uint best_triangle = 0;
    float2 best_barycentric = 0.0f;
    float4 best_derivatives = 0.0f;

    const uint tile_index = (pixel.y / push_constants.tile_size) * push_constants.tile_count_x +
                            pixel.x / push_constants.tile_size;
    const uint tile_begin = tile_offsets[tile_index];
    const uint tile_end = tile_offsets[tile_index + 1];
    for (uint tile_entry = tile_begin; tile_entry < tile_end; ++tile_entry)
    {
        const uint triangle_index = tile_triangle_indices[tile_entry];
        const uint index_offset = triangle_index * 3;
        const uint3 vertex_indices = uint3(
            indices[index_offset + 0],
            indices[index_offset + 1],
            indices[index_offset + 2]);
        if (vertex_indices.x >= push_constants.vertex_count ||
            vertex_indices.y >= push_constants.vertex_count ||
            vertex_indices.z >= push_constants.vertex_count)
        {
            continue;
        }
        const float4 p0 = positions[vertex_indices.x];
        const float4 p1 = positions[vertex_indices.y];
        const float4 p2 = positions[vertex_indices.z];
        if (p0.w <= 1e-8f || p1.w <= 1e-8f || p2.w <= 1e-8f)
        {
            continue;
        }

        const float2 s0 = p0.xy / p0.w;
        const float2 s1 = p1.xy / p1.w;
        const float2 s2 = p2.xy / p2.w;
        const float screen_area = cross_2d(s1 - s0, s2 - s0);
        if (abs(screen_area) <= 1e-12f)
        {
            continue;
        }
        if ((push_constants.cull_mode == 1 && screen_area <= 0.0f) ||
            (push_constants.cull_mode == 2 && screen_area >= 0.0f))
        {
            continue;
        }

        const float edge_0 = cross_2d(s1 - sample_ndc, s2 - sample_ndc);
        const float edge_1 = cross_2d(s2 - sample_ndc, s0 - sample_ndc);
        const float edge_2 = cross_2d(s0 - sample_ndc, s1 - sample_ndc);
        const float coverage_sign = screen_area > 0.0f ? 1.0f : -1.0f;
        const float coverage_epsilon = -1e-7f;
        if (edge_0 * coverage_sign < coverage_epsilon ||
            edge_1 * coverage_sign < coverage_epsilon ||
            edge_2 * coverage_sign < coverage_epsilon)
        {
            continue;
        }

        const float2 q0 = p0.xy - sample_ndc * p0.w;
        const float2 q1 = p1.xy - sample_ndc * p1.w;
        const float2 q2 = p2.xy - sample_ndc * p2.w;
        const float a0 = cross_2d(q1, q2);
        const float a1 = cross_2d(q2, q0);
        const float a2 = cross_2d(q0, q1);
        const float homogeneous_area = a0 + a1 + a2;
        if (abs(homogeneous_area) <= 1e-12f)
        {
            continue;
        }
        const float inverse_area = 1.0f / homogeneous_area;
        const float2 barycentric = clamp(float2(a0, a1) * inverse_area, 0.0f, 1.0f);

        const float interpolated_z = p0.z * a0 + p1.z * a1 + p2.z * a2;
        const float interpolated_w = p0.w * a0 + p1.w * a1 + p2.w * a2;
        if (abs(interpolated_w) <= 1e-12f)
        {
            continue;
        }
        const float depth = clamp(interpolated_z / interpolated_w, -1.0f, 1.0f);
        if (depth > best_depth ||
            (depth == best_depth && best_triangle != 0 && triangle_index + 1 >= best_triangle))
        {
            continue;
        }

        best_depth = depth;
        best_triangle = triangle_index + 1;
        best_barycentric = barycentric;
        if (push_constants.output_derivatives != 0)
        {
            const float da0_dx = -p1.w * p2.y + p2.w * p1.y;
            const float da0_dy = -p1.x * p2.w + p1.w * p2.x;
            const float da1_dx = -p2.w * p0.y + p2.y * p0.w;
            const float da1_dy = -p2.x * p0.w + p2.w * p0.x;
            const float da2_dx = -p0.w * p1.y + p0.y * p1.w;
            const float da2_dy = -p0.x * p1.w + p0.w * p1.x;
            const float d_area_dx = da0_dx + da1_dx + da2_dx;
            const float d_area_dy = da0_dy + da1_dy + da2_dy;
            const float2 d_bc_dx = float2(
                da0_dx - barycentric.x * d_area_dx,
                da1_dx - barycentric.y * d_area_dx) * inverse_area * 2.0f / float(push_constants.width);
            const float2 d_bc_dy = float2(
                da0_dy - barycentric.x * d_area_dy,
                da1_dy - barycentric.y * d_area_dy) * inverse_area * 2.0f / float(push_constants.height);
            best_derivatives = float4(d_bc_dx.x, d_bc_dy.x, d_bc_dx.y, d_bc_dy.y);
        }
    }

    if (best_triangle == 0)
    {
        store_raster(pixel_index, 0.0f);
        store_derivatives(pixel_index, 0.0f);
    }
    else
    {
        store_raster(pixel_index, float4(best_barycentric, best_depth, float(best_triangle)));
        store_derivatives(pixel_index, best_derivatives);
    }
}
