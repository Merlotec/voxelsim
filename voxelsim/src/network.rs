use crate::phase::PhaseGrid;
use crate::viewport::{CameraOrientation, CameraProjection};
use crate::{Agent, VoxelGrid};
use serde::{Deserialize, Serialize};

use std::collections::HashMap;
use std::io::Write;
use std::net::TcpStream;
use std::sync::Arc;

pub struct PovStream {
    virtual_world_stream: TcpStream,
    agent_stream: TcpStream,
}

#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct AsyncRendererClient {
    inner: Arc<RendererClient>,
}

impl AsyncRendererClient {
    pub fn new(client: RendererClient) -> Self {
        Self {
            inner: Arc::new(client),
        }
    }

    pub fn send_world(&self, data: VoxelGrid) {
        let inner = self.inner.clone();
        std::thread::spawn(move || {
            let _ = inner.send_world(&data);
        });
    }

    pub fn send_agents(&self, data: HashMap<usize, Agent>) {
        let inner = self.inner.clone();
        std::thread::spawn(move || {
            let _ = inner.send_agents(&data);
        });
    }

    pub fn send_pov(&self, stream_idx: usize, data: PovData) {
        let inner = self.inner.clone();
        std::thread::spawn(move || {
            let _ = inner.send_pov(stream_idx, &data);
        });
    }
}
/// Network client for sending data to the renderer
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct RendererClient {
    // First stream is the virtual world data, second is tth
    pov_streams: Vec<PovStream>,
    world_stream: Option<TcpStream>,
    agent_stream: Option<TcpStream>,
    host: String,
    world_port: u16,
    agent_port: u16,
    pov_start_port: u16,
    pov_agent_start_port: u16,
}

impl RendererClient {
    pub fn new(
        host: &str,
        world_port: u16,
        agent_port: u16,
        pov_start_port: u16,
        pov_agent_start_port: u16,
        pov_count: u16,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut client = Self {
            pov_streams: Vec::new(),
            world_stream: None,
            agent_stream: None,
            host: host.to_string(),
            world_port,
            agent_port,
            pov_start_port,
            pov_agent_start_port,
        };

        client.connect(pov_count)?;

        Ok(client)
    }

    fn connect(&mut self, pov_count: u16) -> Result<(), Box<dyn std::error::Error>> {
        // Connect to world port
        self.world_stream = Some(TcpStream::connect(format!(
            "{}:{}",
            self.host, self.world_port
        ))?);

        // Connect to agent port
        self.agent_stream = Some(TcpStream::connect(format!(
            "{}:{}",
            self.host, self.agent_port
        ))?);

        self.pov_streams.clear();
        for i in 0..pov_count {
            let port = self.pov_start_port + i;
            let agent_port = self.pov_agent_start_port + i;
            if let (Ok(virtual_world_stream), Ok(agent_stream)) = (
                TcpStream::connect(format!("{}:{}", self.host, port)),
                TcpStream::connect(format!("{}:{}", self.host, agent_port)),
            ) {
                self.pov_streams.push(PovStream {
                    virtual_world_stream,
                    agent_stream,
                });
            } else {
                return Err(format!("Could not connect pov/agent stream for agent {}", i).into());
            }
        }

        Ok(())
    }

    pub fn send_world(&self, data: &VoxelGrid) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref stream) = self.world_stream {
            Self::send_data(data, stream)
        } else {
            Err("Not connected to world port".into())
        }
    }

    pub fn send_agents(
        &self,
        data: &HashMap<usize, Agent>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref stream) = self.agent_stream {
            Self::send_data(data, stream)?;
        } else {
            return Err("Not connected to agent port".into());
        }
        for stream in self.pov_streams.iter() {
            Self::send_data(data, &stream.agent_stream)?;
        }
        Ok(())
    }

    pub fn send_pov(
        &self,
        stream_idx: usize,
        data: &PovData,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref mut stream) = self.pov_streams.get(stream_idx) {
            Self::send_data(data, &stream.virtual_world_stream)
        } else {
            Err("Not connected to pov port".into())
        }
    }

    pub fn send_pov_ref(
        &self,
        stream_idx: usize,
        data: &PovDataRef,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref mut stream) = self.pov_streams.get(stream_idx) {
            Self::send_data(data, &stream.virtual_world_stream)
        } else {
            Err("Not connected to pov port".into())
        }
    }
    pub fn send_data<D>(data: D, mut stream: &TcpStream) -> Result<(), Box<dyn std::error::Error>>
    where
        D: serde::Serialize,
    {
        let bin = bincode::serde::encode_to_vec(data, bincode::config::standard())?;

        // Send length prefix (4 bytes, little-endian)
        let len = (bin.len() as u32).to_le_bytes();
        stream.write_all(&len)?;

        // Send the actual data
        stream.write_all(&bin)?;
        stream.flush()?;

        Ok(())
    }

    pub fn disconnect(&mut self) {
        self.world_stream = None;
        self.agent_stream = None;
    }
}

impl Drop for RendererClient {
    fn drop(&mut self) {
        self.disconnect();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PovData {
    pub virtual_world: VoxelGrid,
    pub dynamic_world: PhaseGrid,
    pub agent_id: usize,
    pub proj: CameraProjection,
    pub orientation: CameraOrientation,
}

#[derive(Debug, Clone, Serialize)]
pub struct PovDataRef<'a, 'b> {
    pub virtual_world: &'a VoxelGrid,
    pub dynamic_world: &'b PhaseGrid,
    pub agent_id: usize,
    pub proj: CameraProjection,
    pub orientation: CameraOrientation,
}
