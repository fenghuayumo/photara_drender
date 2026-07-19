struct PushConstants
{
    float4 camera_position_bias;
    uint4 dimensions;
};

[[vk::binding(0, 0)]] StructuredBuffer<float> atlas_positions;
[[vk::binding(1, 0)]] StructuredBuffer<float> atlas_normals;
[[vk::binding(2, 0)]] StructuredBuffer<float> atlas_raster;
[[vk::binding(3, 0)]] RaytracingAccelerationStructure scene;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> visibility;
[[vk::push_constant]] ConstantBuffer<PushConstants> push_constants;

[numthreads(64, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    const uint pixel_index = dispatch_thread_id.x;
    if (pixel_index >= push_constants.dimensions.x * push_constants.dimensions.y)
    {
        return;
    }
    if ((uint)round(atlas_raster[pixel_index * 4 + 3]) == 0)
    {
        visibility[pixel_index] = 0.0f;
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
    const float3 camera_vector = push_constants.camera_position_bias.xyz - position;
    const float distance_to_camera = length(camera_vector);
    if (distance_to_camera <= push_constants.camera_position_bias.w * 2.0f)
    {
        visibility[pixel_index] = 1.0f;
        return;
    }
    const float3 direction = camera_vector / distance_to_camera;
    const float normal_sign = dot(normal, direction) >= 0.0f ? 1.0f : -1.0f;
    RayDesc ray;
    ray.Origin = position + normal * (push_constants.camera_position_bias.w * normal_sign);
    ray.Direction = direction;
    ray.TMin = push_constants.camera_position_bias.w;
    ray.TMax = max(ray.TMin, distance_to_camera - push_constants.camera_position_bias.w * 2.0f);
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xff, ray);
    while (query.Proceed())
    {
    }
    visibility[pixel_index] = query.CommittedStatus() == COMMITTED_NOTHING ? 1.0f : 0.0f;
}
