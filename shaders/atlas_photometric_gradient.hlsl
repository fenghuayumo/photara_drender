struct PushConstants
{
    uint pixel_count;
    uint texture_width;
    uint texture_height;
    uint loss_index;
    float photometric_epsilon;
    float normalization;
};

[[vk::binding(0, 0)]] StructuredBuffer<float> texture_values;
[[vk::binding(1, 0)]] StructuredBuffer<float> uv_values;
[[vk::binding(2, 0)]] StructuredBuffer<float> target_rgb;
[[vk::binding(3, 0)]] StructuredBuffer<float> valid_mask;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> texture_gradient;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> loss_history;
[[vk::push_constant]] ConstantBuffer<PushConstants> push_constants;

groupshared float group_loss[64];

void atomic_add_gradient(uint index, float value)
{
    uint expected = texture_gradient[index];
    uint original;
    do
    {
        original = expected;
        InterlockedCompareExchange(texture_gradient[index], original, asuint(asfloat(original) + value), expected);
    }
    while (expected != original);
}

void atomic_add_loss(uint index, float value)
{
    uint expected = loss_history[index];
    uint original;
    do
    {
        original = expected;
        InterlockedCompareExchange(loss_history[index], original, asuint(asfloat(original) + value), expected);
    }
    while (expected != original);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID, uint group_index : SV_GroupIndex)
{
    const uint pixel_index = dispatch_thread_id.x;
    float local_loss = 0.0f;
    if (pixel_index < push_constants.pixel_count)
    {
        const float mask = valid_mask[pixel_index];
        if (mask > 0.0f)
        {
            const float2 uv = float2(uv_values[pixel_index * 2], uv_values[pixel_index * 2 + 1]);
            const float2 texel_position = uv * float2(push_constants.texture_width, push_constants.texture_height) - 0.5f;
            const int2 base = (int2)floor(texel_position);
            const float2 fraction = texel_position - floor(texel_position);
            const uint x0 = (uint)clamp(base.x, 0, (int)push_constants.texture_width - 1);
            const uint x1 = (uint)clamp(base.x + 1, 0, (int)push_constants.texture_width - 1);
            const uint y0 = (uint)clamp(base.y, 0, (int)push_constants.texture_height - 1);
            const uint y1 = (uint)clamp(base.y + 1, 0, (int)push_constants.texture_height - 1);
            const uint texels[4] = {
                y0 * push_constants.texture_width + x0,
                y0 * push_constants.texture_width + x1,
                y1 * push_constants.texture_width + x0,
                y1 * push_constants.texture_width + x1
            };
            const float weights[4] = {
                (1.0f - fraction.x) * (1.0f - fraction.y),
                fraction.x * (1.0f - fraction.y),
                (1.0f - fraction.x) * fraction.y,
                fraction.x * fraction.y
            };
            for (uint channel = 0; channel < 3; ++channel)
            {
                float rendered = 0.0f;
                for (uint corner = 0; corner < 4; ++corner)
                {
                    rendered += texture_values[texels[corner] * 3 + channel] * weights[corner];
                }
                const float difference = rendered - target_rgb[pixel_index * 3 + channel];
                const float robust = sqrt(difference * difference +
                                          push_constants.photometric_epsilon * push_constants.photometric_epsilon);
                const float output_gradient = difference / robust * mask * push_constants.normalization;
                local_loss += robust * mask * push_constants.normalization;
                for (uint corner = 0; corner < 4; ++corner)
                {
                    atomic_add_gradient(texels[corner] * 3 + channel, output_gradient * weights[corner]);
                }
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
        atomic_add_loss(push_constants.loss_index, group_loss[0]);
    }
}
