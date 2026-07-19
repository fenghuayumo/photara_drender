struct PushConstants
{
    uint vertex_count;
    uint triangle_count;
    uint pixel_count;
    uint attribute_count;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint> indices;
[[vk::binding(1, 0)]] StructuredBuffer<float> raster;
[[vk::binding(2, 0)]] StructuredBuffer<float> grad_interpolated;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> grad_attributes;
[[vk::push_constant]] ConstantBuffer<PushConstants> push_constants;

[numthreads(64, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    const uint output_index = dispatch_thread_id.x;
    const uint output_count = push_constants.vertex_count * push_constants.attribute_count;
    if (output_index >= output_count)
    {
        return;
    }

    const uint vertex_index = output_index / push_constants.attribute_count;
    const uint attribute_index = output_index % push_constants.attribute_count;
    float gradient = 0.0f;
    for (uint pixel_index = 0; pixel_index < push_constants.pixel_count; ++pixel_index)
    {
        const uint raster_offset = pixel_index * 4;
        const uint triangle_id = (uint)round(raster[raster_offset + 3]);
        if (triangle_id == 0 || triangle_id > push_constants.triangle_count)
        {
            continue;
        }
        const uint index_offset = (triangle_id - 1) * 3;
        const uint3 vertex_indices = uint3(
            indices[index_offset + 0],
            indices[index_offset + 1],
            indices[index_offset + 2]);
        const float2 uv = float2(raster[raster_offset + 0], raster[raster_offset + 1]);
        float weight = 0.0f;
        if (vertex_indices.x == vertex_index)
        {
            weight += uv.x;
        }
        if (vertex_indices.y == vertex_index)
        {
            weight += uv.y;
        }
        if (vertex_indices.z == vertex_index)
        {
            weight += 1.0f - uv.x - uv.y;
        }
        gradient += weight * grad_interpolated[pixel_index * push_constants.attribute_count + attribute_index];
    }
    grad_attributes[output_index] = gradient;
}

