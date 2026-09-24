struct PushConstants
{
    float4 matrix_row_0;
    float4 matrix_row_1;
    float4 matrix_row_2;
    float4 matrix_row_3;
    float4 camera_position_weight;
    float4 viewport;
    uint4 dimensions;
    float4 settings;
};

[[vk::binding(0, 0)]] StructuredBuffer<float> atlas_positions;
[[vk::binding(1, 0)]] StructuredBuffer<float> atlas_normals;
[[vk::binding(2, 0)]] StructuredBuffer<float> atlas_raster;
[[vk::binding(3, 0)]] StructuredBuffer<float> shadow_raster;
[[vk::binding(4, 0)]] StructuredBuffer<float> photo;
[[vk::binding(5, 0)]] StructuredBuffer<float> photo_mask;
[[vk::binding(6, 0)]] StructuredBuffer<float> ray_visibility;
[[vk::binding(7, 0)]] RWStructuredBuffer<float> accumulated_color;
[[vk::binding(8, 0)]] RWStructuredBuffer<float> accumulated_weight;
[[vk::binding(9, 0)]] RWStructuredBuffer<float> best_confidence;
[[vk::binding(10, 0)]] RWStructuredBuffer<uint> source_view;
[[vk::push_constant]] ConstantBuffer<PushConstants> push_constants;

float4 transform_position(float3 position)
{
    const float4 value = float4(position, 1.0f);
    return float4(
        dot(push_constants.matrix_row_0, value),
        dot(push_constants.matrix_row_1, value),
        dot(push_constants.matrix_row_2, value),
        dot(push_constants.matrix_row_3, value));
}

float load_channel(int2 coordinate, uint channel)
{
    const uint width = push_constants.dimensions.z;
    const uint height = push_constants.dimensions.w;
    const uint packed = asuint(push_constants.settings.z);
    const uint channel_count = packed & 0xffu;
    coordinate = clamp(coordinate, int2(0, 0), int2((int)width - 1, (int)height - 1));
    return photo[((uint)coordinate.y * width + (uint)coordinate.x) * channel_count + channel];
}

float sample_photo(float2 pixel_position, uint channel)
{
    const float2 texel_position = pixel_position - 0.5f;
    const int2 base = (int2)floor(texel_position);
    const float2 fraction = texel_position - floor(texel_position);
    const float v00 = load_channel(base, channel);
    const float v10 = load_channel(base + int2(1, 0), channel);
    const float v01 = load_channel(base + int2(0, 1), channel);
    const float v11 = load_channel(base + int2(1, 1), channel);
    return lerp(lerp(v00, v10, fraction.x), lerp(v01, v11, fraction.x), fraction.y);
}

float sample_mask(float2 pixel_position)
{
    const uint width = push_constants.dimensions.z;
    const uint height = push_constants.dimensions.w;
    const float2 texel_position = pixel_position - 0.5f;
    const int2 base = (int2)floor(texel_position);
    const float2 fraction = texel_position - floor(texel_position);
    const int2 p00 = clamp(base, int2(0, 0), int2((int)width - 1, (int)height - 1));
    const int2 p10 = clamp(base + int2(1, 0), int2(0, 0), int2((int)width - 1, (int)height - 1));
    const int2 p01 = clamp(base + int2(0, 1), int2(0, 0), int2((int)width - 1, (int)height - 1));
    const int2 p11 = clamp(base + int2(1, 1), int2(0, 0), int2((int)width - 1, (int)height - 1));
    const float v00 = photo_mask[(uint)p00.y * width + (uint)p00.x];
    const float v10 = photo_mask[(uint)p10.y * width + (uint)p10.x];
    const float v01 = photo_mask[(uint)p01.y * width + (uint)p01.x];
    const float v11 = photo_mask[(uint)p11.y * width + (uint)p11.x];
    return lerp(lerp(v00, v10, fraction.x), lerp(v01, v11, fraction.x), fraction.y);
}

float shadow_visibility(float2 pixel_position, float projected_depth, uint pcf_radius)
{
    const uint image_width = push_constants.dimensions.z;
    const uint image_height = push_constants.dimensions.w;
    const int2 center = (int2)floor(pixel_position);
    float visible = 0.0f;
    float sample_count = 0.0f;
    for (int y = -(int)pcf_radius; y <= (int)pcf_radius; ++y)
    {
        for (int x = -(int)pcf_radius; x <= (int)pcf_radius; ++x)
        {
            const int2 coordinate = center + int2(x, y);
            if (coordinate.x < 0 || coordinate.y < 0 || coordinate.x >= (int)image_width || coordinate.y >= (int)image_height)
            {
                continue;
            }
            const uint offset = ((uint)coordinate.y * image_width + (uint)coordinate.x) * 4;
            if ((uint)round(shadow_raster[offset + 3]) != 0)
            {
                visible += projected_depth <= shadow_raster[offset + 2] + push_constants.settings.x ? 1.0f : 0.0f;
                sample_count += 1.0f;
            }
        }
    }
    return sample_count > 0.0f ? visible / sample_count : 0.0f;
}

[numthreads(64, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    const uint atlas_width = push_constants.dimensions.x;
    const uint atlas_height = push_constants.dimensions.y;
    const uint pixel_index = dispatch_thread_id.x;
    if (pixel_index >= atlas_width * atlas_height || (uint)round(atlas_raster[pixel_index * 4 + 3]) == 0)
    {
        return;
    }

    const float3 position = float3(
        atlas_positions[pixel_index * 3 + 0],
        atlas_positions[pixel_index * 3 + 1],
        atlas_positions[pixel_index * 3 + 2]);
    const float3 normal = normalize(float3(
        atlas_normals[pixel_index * 3 + 0],
        atlas_normals[pixel_index * 3 + 1],
        atlas_normals[pixel_index * 3 + 2]));
    const float4 clip = transform_position(position);
    if (clip.w <= 1e-8f)
    {
        return;
    }
    const float3 ndc = clip.xyz / clip.w;
    if (any(ndc.xy < -1.0f) || any(ndc.xy > 1.0f) || ndc.z < -1.0f || ndc.z > 1.0f)
    {
        return;
    }
    const float2 pixel_position = push_constants.viewport.xy + (ndc.xy * 0.5f + 0.5f) * push_constants.viewport.zw;
    if (any(pixel_position < push_constants.viewport.xy) ||
        any(pixel_position >= push_constants.viewport.xy + push_constants.viewport.zw))
    {
        return;
    }

    const uint packed = asuint(push_constants.settings.z);
    const uint channel_count = packed & 0xffu;
    const uint blend_mode = (packed >> 8u) & 0xffu;
    const uint pcf_radius = (packed >> 16u) & 0xffu;
    const bool has_photo_mask = ((packed >> 24u) & 1u) != 0;
    const uint visibility_mode = (packed >> 25u) & 0x3u;
    float visibility = ray_visibility[pixel_index];
    if (visibility_mode == 0u)
    {
        visibility = shadow_visibility(pixel_position, ndc.z, pcf_radius);
    }
    else if (visibility_mode == 1u)
    {
        visibility *= shadow_visibility(pixel_position, ndc.z, pcf_radius);
    }
    if (has_photo_mask)
    {
        visibility *= saturate(sample_mask(pixel_position));
    }
    const float3 view_direction = normalize(push_constants.camera_position_weight.xyz - position);
    const float view_cosine = dot(normal, view_direction);
    if (view_cosine < push_constants.settings.y || visibility <= 0.0f)
    {
        return;
    }

    // Photographs are already in the caller's blend space (sRGB or linear).
    const float3 color = float3(sample_photo(pixel_position, 0), sample_photo(pixel_position, 1), sample_photo(pixel_position, 2));
    const float alpha = channel_count == 4 ? sample_photo(pixel_position, 3) : 1.0f;
    const float confidence = visibility * alpha * view_cosine * push_constants.camera_position_weight.w;
    if (confidence <= 0.0f)
    {
        return;
    }
    const uint view_index = asuint(push_constants.settings.w);
    if (confidence > best_confidence[pixel_index])
    {
        best_confidence[pixel_index] = confidence;
        source_view[pixel_index] = view_index;
    }
    if (blend_mode == 0)
    {
        if (confidence >= accumulated_weight[pixel_index])
        {
            accumulated_color[pixel_index * 3 + 0] = color.x;
            accumulated_color[pixel_index * 3 + 1] = color.y;
            accumulated_color[pixel_index * 3 + 2] = color.z;
            accumulated_weight[pixel_index] = confidence;
        }
    }
    else
    {
        accumulated_color[pixel_index * 3 + 0] += color.x * confidence;
        accumulated_color[pixel_index * 3 + 1] += color.y * confidence;
        accumulated_color[pixel_index * 3 + 2] += color.z * confidence;
        accumulated_weight[pixel_index] += confidence;
    }
}
