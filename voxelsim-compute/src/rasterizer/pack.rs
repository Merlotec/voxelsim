use wgpu::{ComputePipeline, BindGroupLayout, BindGroup};

use crate::rasterizer::camera::CameraMatrix;

pub struct PackCompute {
    pub pipeline: ComputePipeline,
    pub layout: BindGroupLayout,
    pub group: BindGroup,
    pub output_buffer: wgpu::Buffer,
    pub output_buffer_size: wgpu::BufferAddress,
    pub dispatch_indirect: wgpu::Buffer,
}

impl PackCompute {
    pub fn create(device: &wgpu::Device, culled_max: usize) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("../../shaders/pack_instances.wgsl"));

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PackCompute BindGroup Layout"),
            entries: &[
                // Culled instances buffer (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Cull params buffer (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Noise params uniform (group matches rasterizer noise layout)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output buffer (write-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PackCompute Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<CameraMatrix>() as u32,
            }],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PackInstances Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Allocate output buffer: 4 bytes for count (align to 16) + N * 16 bytes per item
        let out_item_size = 16u64;
        let output_buffer_size = 16u64 + out_item_size * culled_max as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PackInstances Output Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Temporary group; actual buffers are set by RasterizerState at runtime
        let placeholder = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pack placeholder"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PackCompute BindGroup"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: placeholder.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: placeholder.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: placeholder.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: output_buffer.as_entire_binding() },
            ],
        });

        // Indirect dispatch args buffer (x,y,z). Initialize to (0,1,1).
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct DispatchArgs { x: u32, y: u32, z: u32 }
        let dispatch_indirect = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Pack Dispatch Indirect"),
            contents: bytemuck::bytes_of(&DispatchArgs { x: 0, y: 1, z: 1 }),
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
        });

        Self { pipeline, layout, group, output_buffer, output_buffer_size, dispatch_indirect }
    }
}
