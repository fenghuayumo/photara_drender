struct PushConstants
{
    uint value_count;
    float learning_rate;
    float beta1;
    float beta2;
    float adam_epsilon;
    float bias_correction1;
    float bias_correction2;
    float clamp_min;
    float clamp_max;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float> texture_values;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> texture_gradient;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> first_moment;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> second_moment;
[[vk::push_constant]] ConstantBuffer<PushConstants> push_constants;

[numthreads(64, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    const uint index = dispatch_thread_id.x;
    if (index >= push_constants.value_count)
    {
        return;
    }
    const float gradient = texture_gradient[index];
    const float moment1 = push_constants.beta1 * first_moment[index] +
                          (1.0f - push_constants.beta1) * gradient;
    const float moment2 = push_constants.beta2 * second_moment[index] +
                          (1.0f - push_constants.beta2) * gradient * gradient;
    first_moment[index] = moment1;
    second_moment[index] = moment2;
    const float corrected1 = moment1 / push_constants.bias_correction1;
    const float corrected2 = moment2 / push_constants.bias_correction2;
    texture_values[index] = clamp(
        texture_values[index] - push_constants.learning_rate * corrected1 /
            (sqrt(corrected2) + push_constants.adam_epsilon),
        push_constants.clamp_min,
        push_constants.clamp_max);
    texture_gradient[index] = 0.0f;
}
