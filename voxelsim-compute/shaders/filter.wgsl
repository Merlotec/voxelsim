// The camera data structure, matching the Rust struct.
struct CameraUniform {
    view_proj: mat4x4<f32>,
}

// Use a push constant for the camera data. This removes the need for a bind group.
var<push_constant> camera: CameraUniform;

struct OutputData {
    coord: vec3<i32>,
    _p0: i32,
};

// The output buffer is a struct containing an atomic counter and the data array
struct OutputBuffer {
    count: atomic<u32>,
    // Use a runtime-sized array for the data
    data: array<OutputData>,
};

// A separate buffer to hold one atomic flag per instance
struct AtomicFlags {
    flags: array<atomic<u32>>,
};

// BINDINGS
@group(0) @binding(0) var external_depth_texture: texture_depth_2d;
@group(0) @binding(1) var external_depth_sampler: sampler;
@group(0) @binding(2) var<storage, read_write> output_buffer: OutputBuffer;
@group(0) @binding(3) var<storage, read_write> instance_flags: AtomicFlags;

const EPSILON: f32 = 0.00001;

struct InstanceInput {
    @builtin(instance_index) index: u32,
    @location(1) coord: vec3<i32>,
    @location(2) value: u32,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) coord: vec3<i32>,
    @location(1) value: u32,
    @location(2) instance_index: u32,

}

// VERTEX SHADER
// Pass through vertex data and instance index
@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput
) -> VertexOutput {
    var out: VertexOutput;
    
    let instance_pos = vec3<f32>(instance.coord);
    let world_position = vec4<f32>(model.position + instance_pos, 1.0);
    out.clip_position = camera.view_proj * world_position;
    out.coord = instance.coord;
    out.value = instance.value;
    out.instance_index = instance.index;
    return out;
}


// FRAGMENT SHADER
@fragment
fn fs_main(in: VertexOutput) -> @location(0) u32 {
    // 1. Convert clip space coordinates to texture coordinates
    // clip_position is in screen space (pixels), we need normalized [0,1] coordinates
    let screen_size = vec2<f32>(textureDimensions(external_depth_texture, 0));
    let tex_coords = in.clip_position.xy / screen_size;
    
    // 2. Sample the external depth texture using the sampler (for interpolation)
    let sampled_depth = textureSample(external_depth_texture, external_depth_sampler, tex_coords);
    
    // 3. Perform the depth check - if this fragment is BEHIND the already rendered geometry
    // (larger depth value), then it should be filtered out (removed from the scene)
    if (in.clip_position.z + EPSILON < sampled_depth) {
        // 3. Try to claim the write for this instance.
        // atomicExchange sets the flag to 1 and returns the *old* value.
        let previous_flag = atomicExchange(&instance_flags.flags[in.instance_index], 1u);

        // 4. If the old value was 0, we are the first! Write to the output buffer.
        if (previous_flag == 0u) {
            // Atomically increment the output counter and get the index to write to
            let output_index = atomicAdd(&output_buffer.count, 1u);
            
            // Write our instance data to the list
            output_buffer.data[output_index].coord = in.coord;
            
        }
    }

    // 5. Discard the fragment. We don't want to write to any color/depth attachment.
    return 0;
}
