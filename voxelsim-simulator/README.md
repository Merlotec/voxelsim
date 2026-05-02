# VoxelSim Simulator

Core simulation engine for VoxelSim with optional Python bindings. This project contains the terrain generation, agent simulation, and physics logic. It provides both a Rust library and optional Python bindings via PyO3.

## Features

- **Voxel-based 3D world simulation** with procedural terrain generation
- **Agent physics simulation** with thrust-based movement and collision detection  
- **Network client** for sending data to the renderer
- **Optional Python bindings** for easy scripting and integration
- **Serialization support** with binary (bincode) protocol 

## Building

### Rust Library Only
```bash
cargo build --release
```

### With Python Bindings
```bash
# Build with Python support
cargo build --release --features python

# For Python development (requires maturin)
pip install maturin
maturin develop --features python
```

## Rust Usage

```rust
use voxelsim::*;

// Create terrain configuration
let config = TerrainConfig {
    size: [200, 64, 200],
    height_scale: 80.0,
    seed: 42,
    ..Default::default()
};

// Create world and generate terrain
let mut world = VoxelGrid::new();
world.generate_terrain(&config);

// Create agents
let mut agent = Agent::new(0);
agent.set_position(50.0, 30.0, 50.0);
agent.add_move_command("forward", 1.0);

// Create simulation environment
let mut env = GlobalEnv {
    world,
    agents: vec![agent],
};

// Network client for sending to renderer
let mut client = RendererClient::new("127.0.0.1", 8080, 8081);
client.connect()?;
client.send_world(&env.world)?;
client.send_agents(&env.agents)?;
```

## Python Usage (requires `python` feature)

After building with `maturin develop --features python`:

```python
import voxelsim

# Create world with terrain
world = voxelsim.VoxelGrid()
world.generate_default_terrain(42)  # seed

# Create agent with default drone dynamics
dynamics = voxelsim.AgentDynamics.default_drone()
agent = voxelsim.Agent(0)  # id
env = voxelsim.GlobalEnv(world, [agent])

# Connect to renderer
client = voxelsim.RendererClient("127.0.0.1", 8080, 8081)
client.connect_py()

# Send data to renderer
env.send_world(client)
env.send_agents(client)

# Run simulation step
env.update_with_callback(dynamics, 0.1, step_callback, collision_callback)
```

## Network Protocol

The simulator sends data to the renderer using binary (bincode):

### Binary Protocol 
- **Format**: 4-byte length prefix + bincode-serialized data
- **Performance**: High (recommended for real-time simulation)
- **Methods**: `send_world()`, `send_agents()`

### Network Ports
- **Port 8080**: World data (VoxelGrid)
- **Port 8081**: Agent data (Vec<Agent>)

## Core API Reference

### TerrainConfig
Configuration for procedural terrain generation:
```rust
TerrainConfig {
    size: [i32; 3],           // World dimensions [x, y, z]
    height_scale: f64,        // Terrain height variation
    flat_band: f64,           // Flatness factor (0.0-1.0)
    max_terrain_height: i32,  // Maximum terrain height
    forest_scale: f64,        // Tree density scale
    max_forest_height: i32,   // Maximum tree height  
    seed: u32,                // Random seed
    base_thickness: i32,      // Ground layer thickness
}
```

### VoxelGrid
The 3D voxel world:
- `VoxelGrid::new()` - Create empty world
- `generate_terrain(config)` - Generate terrain from config
- `set(coord, cell)` - Set voxel at coordinate
- `cells()` - Get all voxels as HashMap

### Agent
Individual simulation agents:
- `Agent::new(id)` - Create agent with ID
- `set_position(x, y, z)` - Set agent position
- `perform_sequence(commands)` - Add movement commands
- `step(dynamics, delta)` - Update physics simulation

### GlobalEnv  
Main simulation environment:
- `GlobalEnv { world, agents }` - Contains world and all agents
- `step(dynamics, delta)` - Step all agents
- `update(dynamics, delta, step_fn, collision_fn)` - Step with callbacks

### RendererClient
Network client for renderer communication:
- `RendererClient::new(host, world_port, agent_port)` - Create client
- `connect()` - Connect to renderer
- `send_world(grid)` - Send world data (binary)
- `send_agents(agents)` - Send agent data (binary)


## Architecture

```
User Code (Rust/Python)
        ↓
VoxelSim Library
        ↓ (Binary over TCP)
VoxelSim Renderer  
        ↓ (3D Graphics)
Screen/Window
```

## Dependencies

- **nalgebra**: Linear algebra and 3D math
- **noise**: Procedural terrain generation
- **serde**: Serialization framework
- **bincode**: Binary serialization (fast)
- **pyo3**: Python bindings (optional, `python` feature)
- **bitflags**: Voxel cell flags
- **tinyvec**: Small vector optimization
