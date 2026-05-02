// Only import Tokio stuff for native builds
#[cfg(not(target_arch = "wasm32"))]
use {
    crossbeam_channel::{Receiver, Sender},
    serde::de::DeserializeOwned,
    std::marker::PhantomData,
    std::sync::Arc,
    tokio::io::AsyncReadExt,
    tokio::io::{AsyncBufReadExt, BufReader},
    tokio::net::{TcpListener, TcpStream},
    voxelsim::{Agent, VoxelGrid},
    std::env,
};

pub mod convert;
pub mod pov;
pub mod render;
pub mod world;

// Network module only for native builds (uses Tokio)
#[cfg(not(target_arch = "wasm32"))]
pub mod network;

// Network module for WASM builds (uses WebSocket)
#[cfg(target_arch = "wasm32")]
pub mod network_wasm;

fn main() {
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        println!("Starting VoxelSim Renderer (WASM Demo Mode)...");
        world::run_world_demo();
        return;
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        println!("Starting VoxelSim Renderer...");

        // Collect all the command‐line arguments into a Vec<String>
        let args: Vec<String> = std::env::args().collect();

        // Look for "--mode" in the args
        if let Some(pos) = args.iter().position(|arg| arg == "--virtual") {
            // Make sure there's something after "--mode"
            if let Some(val) = args.get(pos + 1) {
                if let Ok(num) = val.parse::<u16>() {
                    // <-- YOUR special‐mode branch
                    println!("Running virtual viewport with offset {}", num);
                    pov::run_pov_server(num); // ... do your special work here ...
                } else {
                    println!("Please specify a virtual viewport offset.");
                }
            } else {
                eprintln!("Error: '--virtual' was provided but no port offset followed it.");
            }
        } else {
            println!("Running in world mode.");
            world::run_world_server();
        }
    }
}
