// Frustum Culling Compute Shader
// Culls voxel instances that are outside the camera's view frustum

struct CameraMatrix {
    view_proj: mat4x4<f32>,
}

struct FrustumPlanes {
    // 6 frustum planes: left, right, bottom, top, near, far
    // Each plane stored as vec4(a, b, c, d) where ax + by + cz + d = 0
    planes: array<vec4<f32>, 6>,
}

struct CellInstance {
    position: vec3<i32>,  // xyz = coord (matches Vector3<i32>)
    value: u32,           // u32 value
}

struct CullParams {
    instance_count: u32,
    visible_count: atomic<u32>,
    _padding: array<u32, 2>,
}

@group(0) @binding(0) var<storage, read> input_instances: array<CellInstance>;
@group(0) @binding(1) var<storage, read_write> output_instances: array<CellInstance>;
@group(0) @binding(2) var<storage, read_write> cull_params: CullParams;
@group(0) @binding(3) var<uniform> frustum_planes: FrustumPlanes;

// Test if a point is inside a frustum plane
fn point_in_plane(point: vec3<f32>, plane: vec4<f32>) -> bool {
    return dot(point, plane.xyz) + plane.w >= 0.0;
}

// DEBUG: Temporary test to isolate the issue
fn cube_in_frustum(center: vec3<f32>) -> bool {
    // DEBUG: Only keep voxels near origin to test if algorithm works
    let distance_from_origin = length(center);
    return distance_from_origin < 10.0;
    
    // Original frustum test (temporarily disabled)
    /*
    let half_size = 0.5; // Half-size of unit cube
    
    // Test against each frustum plane
    for (var i = 0u; i < 6u; i++) {
        let plane = frustum_planes.planes[i];
        let normal = plane.xyz;
        let distance = plane.w;
        
        // Calculate the "positive vertex" (farthest point in plane normal direction)
        let positive_vertex = center + vec3<f32>(
            select(-half_size, half_size, normal.x >= 0.0),
            select(-half_size, half_size, normal.y >= 0.0),
            select(-half_size, half_size, normal.z >= 0.0)
        );
        
        // If the positive vertex is behind the plane, the whole cube is outside
        if dot(positive_vertex, normal) + distance < 0.0 {
            return false;
        }
    }
    
    return true;
    */
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    // Bounds check
    if index >= cull_params.instance_count {
        return;
    }
    
    let instance = input_instances[index];
    let world_pos = vec3<f32>(
        f32(instance.position.x),
        f32(instance.position.y), 
        f32(instance.position.z)
    );
    
    // Test if the cube is in the frustum
    if cube_in_frustum(world_pos) {
        // Atomically get the next output slot
        let output_index = atomicAdd(&cull_params.visible_count, 1u);
        
        // CRITICAL: Check bounds to prevent buffer overflow
        let max_output = arrayLength(&output_instances);
        if output_index < max_output {
            output_instances[output_index] = instance;
        }
        // If we overflow, we've lost instances - this could cause holes
    }
}