pub struct TextureSet {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl TextureSet {
    pub fn create_render_target_texture(
        device: &wgpu::Device,
        size: [u32; 2],
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        label: &str,
    ) -> Self {
        let size3d = wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some(label),
            size: size3d,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            // Usage needs to include RENDER_ATTACHMENT to be a render target
            // and TEXTURE_BINDING to be readable by a shader.
            usage,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,

            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
        }
    }

    pub fn create_depth_texture(device: &wgpu::Device, size: [u32; 2], label: &str) -> Self {
        let size = wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CellTexel {
    coord: Coord,
    value: u32,
}

use voxelsim::Coord;
use crate::buf::StagingBufferPool;
use futures_intrusive::channel::shared::oneshot_channel;

use crate::buf::BatchedReadOperation;

impl StagingBufferPool {
    /// Prepare a texture extraction operation without polling
    pub fn prepare_texture_extraction(
        &mut self,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> BatchedReadOperation {
        let texture_format = texture.format();
        let (block_width, block_height) = texture_format.block_dimensions();
        let bytes_per_block = texture_format.block_copy_size(None).unwrap();

        let buffer_size = wgpu::util::align_to(
            texture.width() / block_width * bytes_per_block,
            wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
        ) * (texture.height() / block_height);

        let usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;

        // Get or create a buffer from the pool
        let output_buffer = self.get_or_create_buffer(buffer_size as u64, usage);

        // Issue the copy command
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Batched Texture-to-Buffer Encoder"),
        });

        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(wgpu::util::align_to(
                        texture.width() / block_width * bytes_per_block,
                        wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
                    )),
                    rows_per_image: None,
                },
            },
            texture.size(),
        );

        // Submit and capture submission index
        let submission_index = queue.submit(Some(encoder.finish()));
        
        let buffer_slice = output_buffer.slice(..);
        let (sender, receiver) = oneshot_channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        BatchedReadOperation {
            staging_buffer: output_buffer,
            size: buffer_size as u64,
            usage,
            receiver,
            submission_index,
        }
    }

    pub async fn extract_texture_data<T>(
        &mut self,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Result<Vec<T>, wgpu::Error>
    where
        T: Copy,
    {
        let texture_format = texture.format();
        let (block_width, block_height) = texture_format.block_dimensions();
        let bytes_per_block = texture_format.block_copy_size(None).unwrap();

        let buffer_size = wgpu::util::align_to(
            texture.width() / block_width * bytes_per_block,
            wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
        ) * (texture.height() / block_height);

        let usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;

        // Get or create a buffer from the pool
        let output_buffer = self.get_or_create_buffer(buffer_size as u64, usage);

        // Issue the copy command
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Texture-to-Buffer Encoder"),
        });

        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(wgpu::util::align_to(
                        texture.width() / block_width * bytes_per_block,
                        wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
                    )),
                    rows_per_image: None,
                },
            },
            texture.size(),
        );

        // Submit and map the buffer
        queue.submit(Some(encoder.finish()));
        
        let buffer_slice = output_buffer.slice(..);
        let (sender, receiver) = oneshot_channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        // Wait for operations to complete
        self.device.poll(wgpu::wgt::PollType::Wait).unwrap();

        // Wait for mapping completion
        if let Some(Ok(())) = receiver.receive().await {
            let data = buffer_slice.get_mapped_range();

            let data_slice = unsafe {
                std::slice::from_raw_parts::<T>(
                    data.as_ptr() as *const T,
                    data.len() / std::mem::size_of::<T>(),
                )
            };

            let result = data_slice.to_vec();
            
            drop(data);
            output_buffer.unmap();

            // Return buffer to pool for reuse
            self.return_buffer(output_buffer, buffer_size as u64, usage);

            Ok(result)
        } else {
            // Even on failure, return the buffer to pool
            self.return_buffer(output_buffer, buffer_size as u64, usage);
            Err(wgpu::Error::Validation {
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Failed to map texture readback buffer",
                )),
                description: "Buffer mapping failed".to_string(),
            })
        }
    }
}

pub async fn extract_texture_data<T>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) -> Result<Vec<T>, wgpu::Error>
where
    T: Copy,
{
    let texture_format = texture.format();
    let (block_width, block_height) = texture_format.block_dimensions();
    let bytes_per_block = texture_format.block_copy_size(None).unwrap();

    let buffer_size = wgpu::util::align_to(
        texture.width() / block_width * bytes_per_block,
        wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
    ) * (texture.height() / block_height);

    // 1. Create the destination buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Texture Readback Buffer"),
        size: buffer_size as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // 2. Issue the copy command
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Texture-to-Buffer Encoder"),
    });

    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(wgpu::util::align_to(
                    texture.width() / block_width * bytes_per_block,
                    wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
                )),
                rows_per_image: None,
            },
        },
        texture.size(),
    );

    // 3. Submit and map the buffer
    queue.submit(Some(encoder.finish()));
    
    let buffer_slice = output_buffer.slice(..);
    let (sender, receiver) = oneshot_channel();
    
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    // Wait for operations to complete
    device.poll(wgpu::wgt::PollType::Wait).unwrap();

    // Wait for mapping completion
    if let Some(Ok(())) = receiver.receive().await {
        let data = buffer_slice.get_mapped_range();

        let data_slice = unsafe {
            std::slice::from_raw_parts::<T>(
                data.as_ptr() as *const T,
                data.len() / std::mem::size_of::<T>(),
            )
        };

        let result = data_slice.to_vec();
        drop(data);
        output_buffer.unmap();

        Ok(result)
    } else {
        Err(wgpu::Error::Validation {
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to map texture readback buffer",
            )),
            description: "Buffer mapping failed".to_string(),
        })
    }
}

pub fn create_depth_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Depth Comparison Sampler"),

        // How to handle coordinates outside the 0.0-1.0 range.
        // ClampToEdge is standard for screen-space textures.
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,

        // Filtering for magnification and minification.
        // Linear provides smooth sampling between texels.
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,

        // Mipmap filtering. Nearest is fine if you're not using mipmaps.
        mipmap_filter: wgpu::FilterMode::Nearest,

        // Level-of-detail clamps (can usually be left as default).
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,

        ..Default::default()
    })
}
