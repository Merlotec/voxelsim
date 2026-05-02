use futures_intrusive::channel::shared::oneshot_channel;
use std::collections::HashMap;

pub struct BatchedReadOperation {
    pub staging_buffer: wgpu::Buffer,
    pub size: u64,
    pub usage: wgpu::BufferUsages,
    pub receiver:
        futures_intrusive::channel::shared::OneshotReceiver<Result<(), wgpu::BufferAsyncError>>,
    pub submission_index: wgpu::SubmissionIndex,
}

pub struct StagingBufferPool {
    buffer_pool: HashMap<u64, Vec<wgpu::Buffer>>,
    texture_pool: HashMap<u64, Vec<wgpu::Buffer>>,
    pub device: wgpu::Device,
}

impl StagingBufferPool {
    pub fn new(device: wgpu::Device) -> Self {
        Self {
            buffer_pool: HashMap::new(),
            texture_pool: HashMap::new(),
            device,
        }
    }

    pub fn get_or_create_buffer(&mut self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let pool = if usage.contains(wgpu::BufferUsages::MAP_READ) {
            &mut self.buffer_pool
        } else {
            &mut self.texture_pool
        };

        if let Some(buffers) = pool.get_mut(&size) {
            if let Some(buffer) = buffers.pop() {
                return buffer;
            }
        }

        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pooled Staging Buffer"),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    pub fn return_buffer(&mut self, buffer: wgpu::Buffer, size: u64, usage: wgpu::BufferUsages) {
        let pool = if usage.contains(wgpu::BufferUsages::MAP_READ) {
            &mut self.buffer_pool
        } else {
            &mut self.texture_pool
        };

        pool.entry(size).or_insert_with(Vec::new).push(buffer);
    }

    /// Prepare a buffer read operation without polling
    pub fn prepare_buffer_read(
        &mut self,
        queue: &wgpu::Queue,
        source_buffer: &wgpu::Buffer,
        clear: bool,
    ) -> BatchedReadOperation {
        let buffer_size = source_buffer.size();
        let usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;

        // Get or create a staging buffer from the pool
        let staging_buffer = self.get_or_create_buffer(buffer_size, usage);

        // Create a command encoder to copy the data
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Batched Read Buffer Encoder"),
            });

        encoder.copy_buffer_to_buffer(
            source_buffer,
            0, // source offset
            &staging_buffer,
            0, // destination offset
            buffer_size,
        );

        // Clear the buffer if required.
        if clear {
            encoder.clear_buffer(source_buffer, 0, None);
        }

        // Submit the command to the GPU and capture submission index
        let submission_index = queue.submit(Some(encoder.finish()));

        // Set up async mapping but don't poll yet
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = oneshot_channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        BatchedReadOperation {
            staging_buffer,
            size: buffer_size,
            usage,
            receiver,
            submission_index,
        }
    }

    /// Wait for specific submissions to complete with individual timing
    pub fn wait_for_submissions(&self, submission_indices: &[wgpu::SubmissionIndex]) {
        if submission_indices.is_empty() {
            return;
        }

        // Use efficient blocking wait - let GPU driver handle scheduling
        self.device.poll(wgpu::wgt::PollType::Wait).unwrap();
    }

    /// Execute multiple batched read operations in parallel
    pub async fn execute_batched_reads_parallel(
        &mut self,
        operations: Vec<BatchedReadOperation>,
    ) -> Vec<Result<Vec<u8>, String>> {
        // Create futures for all operations
        let mut futures = Vec::new();

        for operation in operations {
            // Create a future that doesn't hold any references to self
            let future = async move {
                // Wait for the mapping to complete
                if let Some(Ok(())) = operation.receiver.receive().await {
                    let buffer_slice = operation.staging_buffer.slice(..);
                    let data = buffer_slice.get_mapped_range();
                    let result = data.to_vec();

                    drop(data);
                    operation.staging_buffer.unmap();

                    // Return the buffer and metadata for cleanup
                    Ok((
                        result,
                        operation.staging_buffer,
                        operation.size,
                        operation.usage,
                    ))
                } else {
                    // Return the buffer and metadata for cleanup
                    Err((
                        "Failed to read buffer from GPU".to_string(),
                        operation.staging_buffer,
                        operation.size,
                        operation.usage,
                    ))
                }
            };
            futures.push(future);
        }

        // Await all futures in parallel using futures crate
        let results = futures::future::join_all(futures).await;

        // Process results and return buffers to pool
        let mut final_results = Vec::new();
        for result in results {
            match result {
                Ok((data, buffer, size, usage)) => {
                    self.return_buffer(buffer, size, usage);
                    final_results.push(Ok(data));
                }
                Err((error, buffer, size, usage)) => {
                    self.return_buffer(buffer, size, usage);
                    final_results.push(Err(error));
                }
            }
        }

        final_results
    }

    /// Execute a batched read operation after polling for its specific submission
    pub async fn execute_batched_read(
        &mut self,
        operation: BatchedReadOperation,
    ) -> Result<Vec<u8>, String> {
        // Wait for the mapping to complete
        if let Some(Ok(())) = operation.receiver.receive().await {
            let buffer_slice = operation.staging_buffer.slice(..);
            let data = buffer_slice.get_mapped_range();
            let result = data.to_vec();

            drop(data);
            operation.staging_buffer.unmap();

            // Return buffer to pool for reuse
            self.return_buffer(operation.staging_buffer, operation.size, operation.usage);

            Ok(result)
        } else {
            // Even on failure, return the buffer to pool
            self.return_buffer(operation.staging_buffer, operation.size, operation.usage);
            Err("Failed to read buffer from GPU".to_string())
        }
    }
}

pub async fn clear_gpu_buffer(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Clear Buffer Encoder"),
    });

    encoder.clear_buffer(buffer, 0, None);

    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::wgt::PollType::Poll).unwrap();
}

impl StagingBufferPool {
    pub async fn read_gpu_buffer(
        &mut self,
        queue: &wgpu::Queue,
        source_buffer: &wgpu::Buffer,
        clear: bool,
    ) -> Vec<u8> {
        let buffer_size = source_buffer.size();
        let usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;

        // Get or create a staging buffer from the pool
        let staging_buffer = self.get_or_create_buffer(buffer_size, usage);

        // Create a command encoder to copy the data
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Read Buffer Encoder"),
            });

        encoder.copy_buffer_to_buffer(
            source_buffer,
            0, // source offset
            &staging_buffer,
            0, // destination offset
            buffer_size,
        );

        // Clear the buffer if required.
        if clear {
            encoder.clear_buffer(source_buffer, 0, None);
        }

        // Submit the command to the GPU
        queue.submit(Some(encoder.finish()));

        // Map the staging buffer to read its contents
        let buffer_slice = staging_buffer.slice(..);

        // Asynchronously request to map the buffer.
        let (sender, receiver) = oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        // Wait for operations to complete
        self.device.poll(wgpu::wgt::PollType::Wait).unwrap();

        // Wait for the mapping to complete and check for errors.
        if let Some(Ok(())) = receiver.receive().await {
            // Get the mapped buffer view
            let data = buffer_slice.get_mapped_range();
            let result = data.to_vec();

            // Unmap the buffer
            drop(data);
            staging_buffer.unmap();

            // Return buffer to pool for reuse
            self.return_buffer(staging_buffer, buffer_size, usage);

            result
        } else {
            // Even on failure, return the buffer to pool
            self.return_buffer(staging_buffer, buffer_size, usage);
            panic!("Failed to read buffer from GPU!");
        }
    }
}

pub async fn read_gpu_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source_buffer: &wgpu::Buffer,
    clear: bool,
) -> Vec<u8> {
    let buffer_size = source_buffer.size();

    // 1. Create a staging buffer to copy the data into
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: buffer_size,
        // This buffer is for copying data from the GPU and mapping it for reading on the CPU
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 2. Create a command encoder to copy the data
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Read Buffer Encoder"),
    });

    encoder.copy_buffer_to_buffer(
        source_buffer,
        0, // source offset
        &staging_buffer,
        0, // destination offset
        buffer_size,
    );

    // Clear the buffer if required.
    if clear {
        encoder.clear_buffer(source_buffer, 0, None);
    }

    // Submit the command to the GPU
    queue.submit(Some(encoder.finish()));

    // 3. Map the staging buffer to read its contents
    let buffer_slice = staging_buffer.slice(..);

    // Asynchronously request to map the buffer.
    // The `*_async` methods are non-blocking.
    let (sender, receiver) = oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    // Wait for operations to complete
    device.poll(wgpu::wgt::PollType::Wait).unwrap();

    // Wait for the mapping to complete and check for errors.
    if let Some(Ok(())) = receiver.receive().await {
        // Get the mapped buffer view
        let data = buffer_slice.get_mapped_range();
        let result = data.to_vec();

        // Unmap the buffer after you're done with it
        drop(data);
        staging_buffer.unmap();

        result
    } else {
        panic!("Failed to read buffer from GPU!");
    }
}
