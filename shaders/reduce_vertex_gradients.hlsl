struct PushConstants
{
    uint vertex_count;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint> vertex_offsets;
[[vk::binding(1, 0)]] StructuredBuffer<uint> vertex_corners;
[[vk::binding(2, 0)]] StructuredBuffer<float4> corner_gradients;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> vertex_gradients;
[[vk::push_constant]] ConstantBuffer<PushConstants> push_constants;

[numthreads(64, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    const uint vertex = dispatch_thread_id.x;
    if (vertex >= push_constants.vertex_count)
    {
        return;
    }

    float4 gradient = 0.0f;
    const uint corner_begin = vertex_offsets[vertex];
    const uint corner_end = vertex_offsets[vertex + 1];
    for (uint adjacency_index = corner_begin; adjacency_index < corner_end; ++adjacency_index)
    {
        gradient += corner_gradients[vertex_corners[adjacency_index]];
    }
    vertex_gradients[vertex] = gradient;
}
