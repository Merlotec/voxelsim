use voxelsim::Coord;
use crate::buf::StagingBufferPool;

pub struct FilterBindings {
    pub layout: wgpu::BindGroupLayout,
    pub group: wgpu::BindGroup,
    pub output_buffer: wgpu::Buffer,
    pub output_buffer_size: wgpu::BufferAddress,
    pub flags_buffer: wgpu::Buffer,
    pub flags_buffer_size: wgpu::BufferAddress,
}

impl FilterBindings {
    /// Ensure the output and flags buffers can handle `instance_count` items.
    /// Recreates buffers and the bind group if growth is needed.
    pub fn ensure_capacity(
        &mut self,
        device: &wgpu::Device,
        depth_texture_view: &wgpu::TextureView,
        depth_texture_sampler: &wgpu::Sampler,
        instance_count: usize,
    ) {
        // Current capacities in number of items
        let coord_cap = if self.output_buffer_size >= 4 {
            ((self.output_buffer_size - 4) as usize) / std::mem::size_of::<Coord>()
        } else {
            0
        };
        let flag_cap = (self.flags_buffer_size as usize) / std::mem::size_of::<u32>();

        if instance_count <= coord_cap && instance_count <= flag_cap {
            return; // Already sufficient
        }

        // Grow to next power of two to reduce reallocations
        let mut new_count = instance_count.max(1);
        new_count = new_count.next_power_of_two();

        // Recreate buffers with new sizes
        let new_output_size =
            (4 + std::mem::size_of::<Coord>() * new_count) as wgpu::BufferAddress;
        let new_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: new_output_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let new_flags_size = (std::mem::size_of::<u32>() * new_count) as wgpu::BufferAddress;
        let new_flags = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Atomic Flags Buffer"),
            size: new_flags_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Recreate bind group using the existing layout
        let new_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(depth_texture_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: new_output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: new_flags.as_entire_binding(),
                },
            ],
        });

        // Swap in new resources
        self.output_buffer = new_output;
        self.output_buffer_size = new_output_size;
        self.flags_buffer = new_flags;
        self.flags_buffer_size = new_flags_size;
        self.group = new_group;
    }
    pub async fn get_filter_list(&self, staging_pool: &mut StagingBufferPool, queue: &wgpu::Queue) -> Vec<Coord> {
        let data = staging_pool.read_gpu_buffer(queue, &self.output_buffer, true).await;
        let count = u32::from_le_bytes(data[0..4].try_into().unwrap());

        let output_data_slice: &[Coord] = bytemuck::cast_slice(&data[4..]);
        let valid = &output_data_slice[..count as usize];

        valid.to_vec()
    }

    pub async fn get_filter_list_legacy(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<Coord> {
        let data = crate::buf::read_gpu_buffer(device, queue, &self.output_buffer, true).await;
        let count = u32::from_le_bytes(data[0..4].try_into().unwrap());

        let output_data_slice: &[Coord] = bytemuck::cast_slice(&data[4..]);
        let valid = &output_data_slice[..count as usize];

        valid.to_vec()
    }

    pub fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Render Bind Group Layout"),
            entries: &[
                // External Depth Texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler for the depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), // Or NonFiltering
                    count: None,
                },
                // Output Buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Atomic Flags Buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    pub fn create(
        device: &wgpu::Device,
        depth_texture_view: &wgpu::TextureView,
        depth_texture_sampler: &wgpu::Sampler,
        instance_count: usize,
    ) -> Self {
        let output_buffer_size =
            (4 + std::mem::size_of::<Coord>() * instance_count) as wgpu::BufferAddress;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_buffer_size,
            // Needs STORAGE for shader writes, and COPY_SRC to read it back on the CPU.
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // The atomic flag buffer, one u32 per instance.
        let flags_buffer_size =
            (std::mem::size_of::<u32>() * instance_count) as wgpu::BufferAddress;
        let flags_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Atomic Flags Buffer"),
            size: flags_buffer_size,
            // Needs STORAGE for shader writes, and COPY_DST to clear it each frame.
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = Self::layout(device);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(depth_texture_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: flags_buffer.as_entire_binding(),
                },
            ],
        });
        Self {
            layout,
            group,
            output_buffer,
            output_buffer_size,
            flags_buffer,
            flags_buffer_size,
        }
    }

    pub fn clear_buffers(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Batch clears into a single submission, avoid per-clear polling
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FilterBuffersClearEncoder"),
        });

        encoder.clear_buffer(&self.flags_buffer, 0, None);
        encoder.clear_buffer(&self.output_buffer, 0, None);

        queue.submit(Some(encoder.finish()));
    }
}
