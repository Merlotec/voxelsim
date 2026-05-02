use crate::rasterizer::CellInstance;
use crate::rasterizer::camera::CameraMatrix;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CullParams {
    pub instance_count: u32,
    pub visible_count: u32, // Note: This will be atomic in the shader, but stored as u32 on CPU
    pub _padding: [u32; 2],
}


pub struct FrustumCulling {
    pub compute_pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub culled_instance_buffer: wgpu::Buffer,
    pub cull_params_buffer: wgpu::Buffer,
    pub frustum_planes_buffer: wgpu::Buffer,
    pub bind_group: Option<wgpu::BindGroup>,
    pub max_instances: u32,
    pub readback_staging_buffer: wgpu::Buffer, // Cached staging buffer for readbacks
}

impl FrustumCulling {
    pub fn new(device: &wgpu::Device, max_instances: u32) -> Self {
        // Create compute shader
        let shader = device
            .create_shader_module(wgpu::include_wgsl!("../../shaders/frustum_cull_clean.wgsl"));

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Frustum Culling Bind Group Layout"),
            entries: &[
                // Input instances (read-only storage buffer)
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
                // Output instances (read-write storage buffer)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Cull parameters (read-write storage buffer)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Camera uniform buffer (matches voxelcoord.wgsl)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create compute pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Frustum Culling Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Frustum Culling Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create buffers
        let culled_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Culled Instance Buffer"),
            size: (std::mem::size_of::<CellInstance>() * max_instances as usize) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cull_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cull Parameters Buffer"),
            size: std::mem::size_of::<CullParams>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let frustum_planes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniform Buffer"),
            size: std::mem::size_of::<CameraMatrix>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create cached staging buffer for readbacks
        let readback_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Culling Readback Staging Buffer"),
            size: std::mem::size_of::<CullParams>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            compute_pipeline,
            bind_group_layout,
            culled_instance_buffer,
            cull_params_buffer,
            frustum_planes_buffer,
            bind_group: None,
            max_instances,
            readback_staging_buffer,
        }
    }

    pub fn create_bind_group(
        &mut self,
        device: &wgpu::Device,
        input_instance_buffer: &wgpu::Buffer,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Frustum Culling Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.culled_instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.cull_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.frustum_planes_buffer.as_entire_binding(),
                },
            ],
        });
        self.bind_group = Some(bind_group);
    }


    pub fn dispatch_culling(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        camera_matrix: &CameraMatrix,
        instance_count: u32,
    ) {
        // Reset visible count to 0
        let cull_params = CullParams {
            instance_count,
            visible_count: 0,
            _padding: [0, 0],
        };

        // Update uniform buffers
        queue.write_buffer(
            &self.cull_params_buffer,
            0,
            bytemuck::bytes_of(&cull_params),
        );
        queue.write_buffer(
            &self.frustum_planes_buffer,
            0,
            bytemuck::bytes_of(camera_matrix),
        );

        // Dispatch compute shader
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Frustum Culling Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.compute_pipeline);
        if let Some(bind_group) = &self.bind_group {
            compute_pass.set_bind_group(0, bind_group, &[]);
            
            // Dispatch with workgroups of 64 threads each
            let workgroup_count = (instance_count + 63) / 64; // Round up division
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
    }

    /// Read back the number of visible instances after culling (synchronous version)
    pub fn get_visible_count_sync(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> u32 {
        // Use cached staging buffer instead of creating new one each frame
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Read Visible Count Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &self.cull_params_buffer,
            0,
            &self.readback_staging_buffer,
            0,
            std::mem::size_of::<CullParams>() as u64,
        );

        queue.submit(Some(encoder.finish()));

        // Map and read the cached buffer synchronously
        let buffer_slice = self.readback_staging_buffer.slice(..);

        // Use a simple boolean flag instead of futures
        use std::sync::{Arc, Mutex};
        let result_flag = Arc::new(Mutex::new(None));
        let flag_clone = result_flag.clone();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            *flag_clone.lock().unwrap() = Some(result);
        });

        // Poll until the mapping is complete
        loop {
            device.poll(wgpu::wgt::PollType::Poll).unwrap();
            if let Some(result) = result_flag.lock().unwrap().as_ref() {
                if result.is_ok() {
                    break;
                } else {
                    return 0;
                }
            }
            std::thread::sleep(std::time::Duration::from_micros(10));
        }

        // Read the data
        let data = buffer_slice.get_mapped_range();
        let cull_params: &CullParams =
            bytemuck::from_bytes(&data[..std::mem::size_of::<CullParams>()]);
        let visible_count = cull_params.visible_count;
        drop(data);
        self.readback_staging_buffer.unmap();
        visible_count
    }

    /// Debug: Read back a few culled instances to verify they're valid
    pub fn debug_read_culled_instances(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        count: u32,
    ) {
        if count == 0 {
            return;
        }

        let instances_to_read = count.min(5); // Read up to 5 instances for debugging
        let size_per_instance = std::mem::size_of::<CellInstance>() as u64;
        let total_size = size_per_instance * instances_to_read as u64;

        // Create staging buffer
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Culled Instances Staging"),
            size: total_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Copy some culled instances
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Debug Read Culled Instances"),
        });

        encoder.copy_buffer_to_buffer(
            &self.culled_instance_buffer,
            0,
            &staging_buffer,
            0,
            total_size,
        );

        queue.submit(Some(encoder.finish()));

        // Read back synchronously
        let buffer_slice = staging_buffer.slice(..);
        let result_flag = std::sync::Arc::new(std::sync::Mutex::new(None));
        let flag_clone = result_flag.clone();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            *flag_clone.lock().unwrap() = Some(result);
        });

        // Poll until complete
        loop {
            device.poll(wgpu::wgt::PollType::Poll).unwrap();
            if let Some(result) = result_flag.lock().unwrap().as_ref() {
                if result.is_ok() {
                    break;
                } else {
                    return;
                }
            }
            std::thread::sleep(std::time::Duration::from_micros(10));
        }

        // Print the data
        let data = buffer_slice.get_mapped_range();
        println!("    🔍 First {} culled instances:", instances_to_read);

        let mut min_pos = [i32::MAX, i32::MAX, i32::MAX];
        let mut max_pos = [i32::MIN, i32::MIN, i32::MIN];

        for i in 0..instances_to_read {
            let offset = i as usize * std::mem::size_of::<CellInstance>();
            if offset + std::mem::size_of::<CellInstance>() <= data.len() {
                let instance_bytes = &data[offset..offset + std::mem::size_of::<CellInstance>()];
                let instance: &CellInstance = bytemuck::from_bytes(instance_bytes);

                // Track bounds
                min_pos[0] = min_pos[0].min(instance.position.x);
                min_pos[1] = min_pos[1].min(instance.position.y);
                min_pos[2] = min_pos[2].min(instance.position.z);
                max_pos[0] = max_pos[0].max(instance.position.x);
                max_pos[1] = max_pos[1].max(instance.position.y);
                max_pos[2] = max_pos[2].max(instance.position.z);

                println!(
                    "      [{}]: pos=({}, {}, {}), value={}, distance={:.1}",
                    i,
                    instance.position.x,
                    instance.position.y,
                    instance.position.z,
                    instance.value,
                    ((instance.position.x * instance.position.x
                        + instance.position.y * instance.position.y
                        + instance.position.z * instance.position.z) as f32)
                        .sqrt()
                );
            }
        }

        if instances_to_read > 0 {
            println!(
                "    📐 Culled range: X[{}..{}], Y[{}..{}], Z[{}..{}]",
                min_pos[0], max_pos[0], min_pos[1], max_pos[1], min_pos[2], max_pos[2]
            );
        }

        drop(data);
        staging_buffer.unmap();
    }

    /// Read back the number of visible instances after culling (async version)
    pub async fn get_visible_count(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> u32 {
        // Create a staging buffer to read back the visible count
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visible Count Staging Buffer"),
            size: std::mem::size_of::<CullParams>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Copy the cull params to staging buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Read Visible Count Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &self.cull_params_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of::<CullParams>() as u64,
        );

        queue.submit(Some(encoder.finish()));

        // Map and read the buffer
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        device.poll(wgpu::wgt::PollType::Wait).unwrap();

        if let Some(Ok(())) = receiver.receive().await {
            let data = buffer_slice.get_mapped_range();
            let cull_params: &CullParams =
                bytemuck::from_bytes(&data[..std::mem::size_of::<CullParams>()]);
            let visible_count = cull_params.visible_count;
            drop(data);
            staging_buffer.unmap();
            visible_count
        } else {
            0
        }
    }
}
