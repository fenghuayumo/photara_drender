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
[[vk::binding(3, 0)]] RWStructuredBuffer<float> interpolated;
[[vk::push_constant]] ConstantBuffer<PushConstants> push_constants;

[numthreads(64, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    const uint output_index = dispatch_thread_id.x;
    const uint output_count = push_constants.pixel_count * push_constants.attribute_count;
    if (output_index >= output_count)
    {
        return;
    }

    const uint pixel_index = output_index / push_constants.attribute_count;
    const uint attribute_index = output_index % push_constants.attribute_count;
    const uint raster_offset = pixel_index * 4;
    const uint triangle_id = (uint)round(raster[raster_offset + 3]);
    if (triangle_id == 0 || triangle_id > push_constants.triangle_count)
    {
        interpolated[output_index] = 0.0f;
        return;
    }

    const uint index_offset = (triangle_id - 1) * 3;
    const uint3 vertex_indices = uint3(
        indices[index_offset + 0],
        indices[index_offset + 1],
        indices[index_offset + 2]);
    if (vertex_indices.x >= push_constants.vertex_count ||
        vertex_indices.y >= push_constants.vertex_count ||
        vertex_indices.z >= push_constants.vertex_count)
    {
        interpolated[output_index] = 0.0f;
        return;
    }

    const float2 uv = float2(raster[raster_offset + 0], raster[raster_offset + 1]);
    const float3 barycentric = float3(uv.x, uv.y, 1.0f - uv.x - uv.y);
    const float a0 = vertex_attributes[vertex_indices.x * push_constants.attribute_count + attribute_index];
    const float a1 = vertex_attributes[vertex_indices.y * push_constants.attribute_count + attribute_index];
    const float a2 = vertex_attributes[vertex_indices.z * push_constants.attribute_count + attribute_index];
    interpolated[output_index] = dot(barycentric, float3(a0, a1, a2));
}

