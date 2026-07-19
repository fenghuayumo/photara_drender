struct PushConstants
{
    uint pixel_count;
    uint texture_width;
    uint texture_height;
    uint channel_count;
    uint address_mode;
};

[[vk::binding(0, 0)]] StructuredBuffer<float> texture_values;
[[vk::binding(1, 0)]] StructuredBuffer<float> uv_values;
[[vk::binding(2, 0)]] StructuredBuffer<float> raster;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> sampled_values;
[[vk::push_constant]] ConstantBuffer<PushConstants> push_constants;

int positive_modulo(int value, int divisor)
{
    const int remainder = value % divisor;
    return remainder < 0 ? remainder + divisor : remainder;
}

int address_coordinate(int coordinate, uint size)
{
    if (push_constants.address_mode == 0)
    {
        return clamp(coordinate, 0, (int)size - 1);
    }
    if (push_constants.address_mode == 1)
    {
        return positive_modulo(coordinate, (int)size);
    }
    const int period = (int)size * 2;
    const int mirrored = positive_modulo(coordinate, period);
    return mirrored < (int)size ? mirrored : period - 1 - mirrored;
}

[numthreads(64, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    const uint output_index = dispatch_thread_id.x;
    const uint output_count = push_constants.pixel_count * push_constants.channel_count;
    if (output_index >= output_count)
    {
        return;
    }

    const uint pixel_index = output_index / push_constants.channel_count;
    const uint channel_index = output_index % push_constants.channel_count;
    if ((uint)round(raster[pixel_index * 4 + 3]) == 0)
    {
        sampled_values[output_index] = 0.0f;
        return;
    }

    const float2 uv = float2(uv_values[pixel_index * 2 + 0], uv_values[pixel_index * 2 + 1]);
    const float2 texel_position = uv * float2(push_constants.texture_width, push_constants.texture_height) - 0.5f;
    const int2 base_coordinate = (int2)floor(texel_position);
    const float2 fraction = texel_position - floor(texel_position);
    const int x0 = address_coordinate(base_coordinate.x, push_constants.texture_width);
    const int x1 = address_coordinate(base_coordinate.x + 1, push_constants.texture_width);
    const int y0 = address_coordinate(base_coordinate.y, push_constants.texture_height);
    const int y1 = address_coordinate(base_coordinate.y + 1, push_constants.texture_height);
    const uint offsets[4] = {
        ((uint)y0 * push_constants.texture_width + (uint)x0) * push_constants.channel_count + channel_index,
        ((uint)y0 * push_constants.texture_width + (uint)x1) * push_constants.channel_count + channel_index,
        ((uint)y1 * push_constants.texture_width + (uint)x0) * push_constants.channel_count + channel_index,
        ((uint)y1 * push_constants.texture_width + (uint)x1) * push_constants.channel_count + channel_index
    };
    const float4 weights = float4(
        (1.0f - fraction.x) * (1.0f - fraction.y),
        fraction.x * (1.0f - fraction.y),
        (1.0f - fraction.x) * fraction.y,
        fraction.x * fraction.y);
    sampled_values[output_index] =
        texture_values[offsets[0]] * weights.x +
        texture_values[offsets[1]] * weights.y +
        texture_values[offsets[2]] * weights.z +
        texture_values[offsets[3]] * weights.w;
}

