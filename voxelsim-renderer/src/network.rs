use crossbeam_channel::{Receiver, Sender};
use serde::de::DeserializeOwned;
use std::marker::PhantomData;
use std::sync::Arc;
use tokio::io::AsyncReadExt;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use voxelsim::{Agent, VoxelGrid};

pub struct NetworkSubscriber<T> {
    sender: Sender<T>,
    port: u16,
    addr: String,
}

impl<T> NetworkSubscriber<T> {
    pub fn new(addr: String, port: u16) -> (Self, Receiver<T>) {
        let (sender, receiver) = crossbeam_channel::unbounded::<T>();

        (Self { sender, port, addr }, receiver)
    }
}

impl<T: 'static + DeserializeOwned + Send + Sync> NetworkSubscriber<T> {
    pub async fn start(&mut self) {
        println!("Starting network listener on port {}...", self.port);

        // Start world data listener
        let port = self.port;
        let sender = self.sender.clone();
        tokio::spawn(async move {
            if let Err(e) = Self::listen_data(port, sender).await {
                eprintln!("World data listener error: {}", e);
            }
        });
    }

    /// Listen for agent data on the specified port
    async fn listen_data(port: u16, sender: Sender<T>) -> tokio::io::Result<()> {
        let listener = TcpListener::bind(format!("127.0.0.1:{}", port)).await?;

        loop {
            let (socket, addr) = listener.accept().await?;
            let sender = sender.clone();
            tokio::spawn(async move {
                if let Err(e) = Self::handle_connection(socket, sender).await {
                    eprintln!("Agent connection error: {}", e);
                }
            });
        }
    }
    /// Handle a world data connection
    async fn handle_connection(mut socket: TcpStream, sender: Sender<T>) -> tokio::io::Result<()> {
        loop {
            // 1) Read the 4-byte length prefix
            let mut len_buf = [0u8; 4];
            if let Err(e) = socket.read_exact(&mut len_buf).await {
                if e.kind() == tokio::io::ErrorKind::UnexpectedEof {
                    break; // connection closed cleanly
                } else {
                    return Err(e);
                }
            }
            let msg_len = u32::from_le_bytes(len_buf) as usize;

            // 2) Read the frame payload
            let mut buf = vec![0u8; msg_len];
            socket.read_exact(&mut buf).await?;

            // 3) Deserialize with bincode
            match bincode::serde::decode_from_slice::<T, _>(&buf, bincode::config::standard()) {
                Ok((data, _)) => {
                    sender.send(data);
                }
                Err(e) => {
                    eprintln!("Failed to deserialize VoxelGrid: {}", e);
                }
            }
        }

        Ok(())
    }
}
