struct PushConstants
{
    uint pair_count;
    uint texture_width;
    uint texture_height;
    uint loss_index;
    float seam_epsilon;
    float normalization;
};

[[vk::binding(0, 0)]] StructuredBuffer<float> texture_values;
[[vk::binding(1, 0)]] StructuredBuffer<float> seam_uv_pairs;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> texture_gradient;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> loss_history;
[[vk::push_constant]] ConstantBuffer<PushConstants> push_constants;

groupshared float group_loss[64];

void atomic_add(RWStructuredBuffer<uint> buffer, uint index, float value)
{
    uint expected = buffer[index];
    uint original;
    do
    {
        original = expected;
        InterlockedCompareExchange(buffer[index], original, asuint(asfloat(original) + value), expected);
    }
    while (expected != original);
}

void sample_layout(float2 uv, out uint texels[4], out float weights[4])
{
    const float2 texel_position = uv * float2(push_constants.texture_width, push_constants.texture_height) - 0.5f;
    const int2 base = (int2)floor(texel_position);
    const float2 fraction = texel_position - floor(texel_position);
    const uint x0 = (uint)clamp(base.x, 0, (int)push_constants.texture_width - 1);
    const uint x1 = (uint)clamp(base.x + 1, 0, (int)push_constants.texture_width - 1);
    const uint y0 = (uint)clamp(base.y, 0, (int)push_constants.texture_height - 1);
    const uint y1 = (uint)clamp(base.y + 1, 0, (int)push_constants.texture_height - 1);
    texels[0] = y0 * push_constants.texture_width + x0;
    texels[1] = y0 * push_constants.texture_width + x1;
    texels[2] = y1 * push_constants.texture_width + x0;
    texels[3] = y1 * push_constants.texture_width + x1;
    weights[0] = (1.0f - fraction.x) * (1.0f - fraction.y);
    weights[1] = fraction.x * (1.0f - fraction.y);
    weights[2] = (1.0f - fraction.x) * fraction.y;
    weights[3] = fraction.x * fraction.y;
}

[numthreads(64, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID, uint group_index : SV_GroupIndex)
{
    const uint pair_index = dispatch_thread_id.x;
    float local_loss = 0.0f;
    if (pair_index < push_constants.pair_count)
    {
        const uint pair_offset = pair_index * 4;
        const float2 uv0 = float2(seam_uv_pairs[pair_offset], seam_uv_pairs[pair_offset + 1]);
        const float2 uv1 = float2(seam_uv_pairs[pair_offset + 2], seam_uv_pairs[pair_offset + 3]);
        uint texels0[4];
        uint texels1[4];
        float weights0[4];
        float weights1[4];
        sample_layout(uv0, texels0, weights0);
        sample_layout(uv1, texels1, weights1);
        for (uint channel = 0; channel < 3; ++channel)
        {
            float color0 = 0.0f;
            float color1 = 0.0f;
            for (uint corner = 0; corner < 4; ++corner)
            {
                color0 += texture_values[texels0[corner] * 3 + channel] * weights0[corner];
                color1 += texture_values[texels1[corner] * 3 + channel] * weights1[corner];
            }
            const float difference = color0 - color1;
            const float robust = sqrt(difference * difference + push_constants.seam_epsilon * push_constants.seam_epsilon);
            const float gradient = difference / robust * push_constants.normalization;
            local_loss += robust * push_constants.normalization;
            for (uint corner = 0; corner < 4; ++corner)
            {
                atomic_add(texture_gradient, texels0[corner] * 3 + channel, gradient * weights0[corner]);
                atomic_add(texture_gradient, texels1[corner] * 3 + channel, -gradient * weights1[corner]);
            }
        }
    }
    group_loss[group_index] = local_loss;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 32; stride > 0; stride >>= 1)
    {
        if (group_index < stride)
        {
            group_loss[group_index] += group_loss[group_index + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (group_index == 0 && group_loss[0] != 0.0f)
    {
        atomic_add(loss_history, push_constants.loss_index, group_loss[0]);
    }
}
