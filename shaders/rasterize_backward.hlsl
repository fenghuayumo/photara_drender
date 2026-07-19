struct PushConstants
{
    uint vertex_count;
    uint triangle_count;
    uint width;
    uint height;
    float4 viewport;
};

[[vk::binding(0, 0)]] StructuredBuffer<float4> positions;
[[vk::binding(1, 0)]] StructuredBuffer<uint> indices;
[[vk::binding(2, 0)]] StructuredBuffer<float> raster;
[[vk::binding(3, 0)]] StructuredBuffer<float> grad_raster;
[[vk::binding(4, 0)]] StructuredBuffer<float> grad_barycentric_derivatives;
[[vk::binding(5, 0)]] RWStructuredBuffer<float4> corner_gradients;
[[vk::push_constant]] ConstantBuffer<PushConstants> push_constants;

float cross_2d(float2 a, float2 b)
{
    return a.x * b.y - a.y * b.x;
}

float2 barycentric_derivative(
    float2 barycentric,
    float da0,
    float da1,
    float da2,
    float inverse_area)
{
    const float d_area = da0 + da1 + da2;
    return float2(
        da0 - barycentric.x * d_area,
        da1 - barycentric.y * d_area) * inverse_area;
}

[numthreads(64, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    const uint triangle_index = dispatch_thread_id.x;
    if (triangle_index >= push_constants.triangle_count)
    {
        return;
    }

    const uint index_offset = triangle_index * 3;
    const uint3 vertex_indices = uint3(
        indices[index_offset + 0],
        indices[index_offset + 1],
        indices[index_offset + 2]);
    float4 triangle_gradients[3] = {float4(0.0f, 0.0f, 0.0f, 0.0f),
                                    float4(0.0f, 0.0f, 0.0f, 0.0f),
                                    float4(0.0f, 0.0f, 0.0f, 0.0f)};
    if (vertex_indices.x >= push_constants.vertex_count ||
        vertex_indices.y >= push_constants.vertex_count ||
        vertex_indices.z >= push_constants.vertex_count)
    {
        corner_gradients[index_offset + 0] = triangle_gradients[0];
        corner_gradients[index_offset + 1] = triangle_gradients[1];
        corner_gradients[index_offset + 2] = triangle_gradients[2];
        return;
    }

    const float4 p[3] = {
        positions[vertex_indices.x],
        positions[vertex_indices.y],
        positions[vertex_indices.z]
    };
    if (p[0].w <= 1e-8f || p[1].w <= 1e-8f || p[2].w <= 1e-8f)
    {
        corner_gradients[index_offset + 0] = triangle_gradients[0];
        corner_gradients[index_offset + 1] = triangle_gradients[1];
        corner_gradients[index_offset + 2] = triangle_gradients[2];
        return;
    }
    const float2 s0 = p[0].xy / p[0].w;
    const float2 s1 = p[1].xy / p[1].w;
    const float2 s2 = p[2].xy / p[2].w;
    const float2 min_ndc = min(s0, min(s1, s2));
    const float2 max_ndc = max(s0, max(s1, s2));
    const int min_x = clamp((int)floor((min_ndc.x * 0.5f + 0.5f) * push_constants.viewport.z + push_constants.viewport.x),
                            0, (int)push_constants.width);
    const int max_x = clamp((int)ceil((max_ndc.x * 0.5f + 0.5f) * push_constants.viewport.z + push_constants.viewport.x),
                            0, (int)push_constants.width);
    const int min_y = clamp((int)floor((min_ndc.y * 0.5f + 0.5f) * push_constants.viewport.w + push_constants.viewport.y),
                            0, (int)push_constants.height);
    const int max_y = clamp((int)ceil((max_ndc.y * 0.5f + 0.5f) * push_constants.viewport.w + push_constants.viewport.y),
                            0, (int)push_constants.height);
    for (int y = min_y; y < max_y; ++y)
    {
        for (int x = min_x; x < max_x; ++x)
        {
            const uint pixel_index = (uint)y * push_constants.width + (uint)x;
            const uint raster_offset = pixel_index * 4;
            if ((uint)round(raster[raster_offset + 3]) != triangle_index + 1)
            {
                continue;
            }
            const float2 loss_gradient = float2(
                grad_raster[raster_offset + 0],
                grad_raster[raster_offset + 1]);
            if (all(loss_gradient == 0.0f))
            {
                continue;
            }

            const float2 sample_ndc = float2(
                (float(x) - push_constants.viewport.x + 0.5f) * 2.0f / push_constants.viewport.z - 1.0f,
                (float(y) - push_constants.viewport.y + 0.5f) * 2.0f / push_constants.viewport.w - 1.0f);
            const float2 q[3] = {
                p[0].xy - sample_ndc * p[0].w,
                p[1].xy - sample_ndc * p[1].w,
                p[2].xy - sample_ndc * p[2].w
            };
            const float area = cross_2d(q[1], q[2]) + cross_2d(q[2], q[0]) + cross_2d(q[0], q[1]);
            if (abs(area) <= 1e-12f)
            {
                continue;
            }
            const float epsilon = area >= 0.0f ? 1e-6f : -1e-6f;
            const float inverse_area = 1.0f / (area + epsilon);
            const float a0 = cross_2d(q[1], q[2]);
            const float a1 = cross_2d(q[2], q[0]);
            const float b0 = a0 * inverse_area;
            const float b1 = a1 * inverse_area;
            const float grad_b0 = loss_gradient.x * inverse_area;
            const float grad_b1 = loss_gradient.y * inverse_area;
            const float grad_common = grad_b0 * b0 + grad_b1 * b1;

            const float grad_p0_x = grad_common * (q[2].y - q[1].y) - grad_b1 * q[2].y;
            const float grad_p1_x = grad_common * (q[0].y - q[2].y) + grad_b0 * q[2].y;
            const float grad_p2_x = grad_common * (q[1].y - q[0].y) -
                                    grad_b0 * q[1].y + grad_b1 * q[0].y;
            const float grad_p0_y = grad_common * (q[1].x - q[2].x) + grad_b1 * q[2].x;
            const float grad_p1_y = grad_common * (q[2].x - q[0].x) - grad_b0 * q[2].x;
            const float grad_p2_y = grad_common * (q[0].x - q[1].x) +
                                    grad_b0 * q[1].x - grad_b1 * q[0].x;
            triangle_gradients[0] += float4(
                grad_p0_x,
                grad_p0_y,
                0.0f,
                -sample_ndc.x * grad_p0_x - sample_ndc.y * grad_p0_y);
            triangle_gradients[1] += float4(
                grad_p1_x,
                grad_p1_y,
                0.0f,
                -sample_ndc.x * grad_p1_x - sample_ndc.y * grad_p1_y);
            triangle_gradients[2] += float4(
                grad_p2_x,
                grad_p2_y,
                0.0f,
                -sample_ndc.x * grad_p2_x - sample_ndc.y * grad_p2_y);
        }
    }

    corner_gradients[index_offset + 0] = triangle_gradients[0];
    corner_gradients[index_offset + 1] = triangle_gradients[1];
    corner_gradients[index_offset + 2] = triangle_gradients[2];
}
