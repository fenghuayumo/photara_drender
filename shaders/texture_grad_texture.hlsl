struct PushConstants
{
    uint texel_count;
    uint pixel_count;
    uint channel_count;
};

[[vk::binding(0, 0)]] StructuredBuffer<float> grad_sampled;
[[vk::binding(1, 0)]] StructuredBuffer<uint> texel_offsets;
[[vk::binding(2, 0)]] StructuredBuffer<uint> sample_entries;
[[vk::binding(3, 0)]] StructuredBuffer<float> sample_weights;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> grad_texture;
[[vk::push_constant]] ConstantBuffer<PushConstants> push_constants;

[numthreads(64, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    const uint output_index = dispatch_thread_id.x;
    const uint output_count = push_constants.texel_count * push_constants.channel_count;
    if (output_index >= output_count)
    {
        return;
    }

    const uint texel_index = output_index / push_constants.channel_count;
    const uint channel_index = output_index % push_constants.channel_count;
    const uint entry_begin = texel_offsets[texel_index];
    const uint entry_end = texel_offsets[texel_index + 1];
    float gradient = 0.0f;
    for (uint adjacency_index = entry_begin; adjacency_index < entry_end; ++adjacency_index)
    {
        const uint sample_entry = sample_entries[adjacency_index];
        const uint pixel_index = sample_entry / 4;
        gradient += sample_weights[sample_entry] *
                    grad_sampled[pixel_index * push_constants.channel_count + channel_index];
    }
    grad_texture[output_index] = gradient;
}

