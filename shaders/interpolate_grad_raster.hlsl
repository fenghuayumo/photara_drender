struct PushConstants
{
    uint vertex_count;
    uint triangle_count;
    uint pixel_count;
    uint attribute_count;
};

[[vk::binding(0, 0)]] StructuredBuffer<float> vertex_attributes;
[[vk::binding(1, 0)]] StructuredBuffer<uint> indices;
[[vk::binding(2, 0)]] StructuredBuffer<float> raster;
[[vk::binding(3, 0)]] StructuredBuffer<float> grad_interpolated;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> grad_raster;
[[vk::push_constant]] ConstantBuffer<PushConstants> push_constants;

[numthreads(64, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    const uint pixel_index = dispatch_thread_id.x;
    if (pixel_index >= push_constants.pixel_count)
    {
        return;
    }

    const uint raster_offset = pixel_index * 4;
    const uint triangle_id = (uint)round(raster[raster_offset + 3]);
    float2 gradient = 0.0f;
    if (triangle_id != 0 && triangle_id <= push_constants.triangle_count)
    {
        const uint index_offset = (triangle_id - 1) * 3;
        const uint3 vertex_indices = uint3(
            indices[index_offset + 0],
            indices[index_offset + 1],
            indices[index_offset + 2]);
        if (vertex_indices.x < push_constants.vertex_count &&
            vertex_indices.y < push_constants.vertex_count &&
            vertex_indices.z < push_constants.vertex_count)
        {
            for (uint attribute_index = 0; attribute_index < push_constants.attribute_count; ++attribute_index)
            {
                const float a0 = vertex_attributes[vertex_indices.x * push_constants.attribute_count + attribute_index];
                const float a1 = vertex_attributes[vertex_indices.y * push_constants.attribute_count + attribute_index];
                const float a2 = vertex_attributes[vertex_indices.z * push_constants.attribute_count + attribute_index];
                const float output_gradient =
                    grad_interpolated[pixel_index * push_constants.attribute_count + attribute_index];
                gradient += output_gradient * float2(a0 - a2, a1 - a2);
            }
        }
    }
    grad_raster[raster_offset + 0] = gradient.x;
    grad_raster[raster_offset + 1] = gradient.y;
    grad_raster[raster_offset + 2] = 0.0f;
    grad_raster[raster_offset + 3] = 0.0f;
}

