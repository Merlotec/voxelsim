pub mod backend;
pub mod controller;
pub mod server;
pub mod slam;
pub mod state;

use std::net::TcpListener;
use std::sync::{Arc, Mutex};

fn main() -> std::io::Result<()> {
    let addr =
        std::env::var("VOXELSIM_DAEMON_ADDR").unwrap_or_else(|_| "127.0.0.1:7000".to_string());
    let mav_addr =
        std::env::var("VOXELSIM_MAV_ADDR").unwrap_or_else(|_| "udpin:0.0.0.0:14540".to_string());

    // Bind TCP listener once and keep accepting connections forever
    let listener = TcpListener::bind(&addr)?;
    println!("Daemon listening on {}", addr);

    // Shared flight state persists across client connections
    let conn = Arc::new(controller::ConnectionInterface::connect(&mav_addr)?);
    let agent = Arc::new(Mutex::new(voxelsim::Agent::new(0)));
    let world = Arc::new(Mutex::new(voxelsim::VoxelGrid::new()));

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                // Handle connection synchronously; on disconnect, continue accepting
                if let Err(e) = server::handle_client(s, conn.clone(), agent.clone(), world.clone())
                {
                    eprintln!("Client error: {}", e);
                }
            }
            Err(e) => eprintln!("Accept error: {}", e),
        }
    }

    Ok(())
}
