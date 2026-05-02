# VoxelSim Renderer

3D renderer for VoxelSim using the Bevy engine. This project receives world and agent data over network sockets and renders the simulation in real-time with interactive camera controls.

## Features

- **Real-time 3D voxel rendering** with optimized mesh generation
- **Procedural terrain visualization** including trees and natural features
- **Agent rendering** with position tracking and movement visualization
- **Interactive camera controls** for scene navigation
- **Automatic world updates** from simulation data
- **Performance monitoring** and frame rate optimization

## Dependencies

- **voxelsim**: Core simulation logic and data structures
- **Bevy**: Modern 3D rendering engine and ECS framework
- **tokio**: Async networking for real-time data streaming
- **bincode**: Binary deserialization (high performance)

## Building

```bash
cargo build --release
```

## Running

```bash
cargo run --release
```

The renderer will start and listen on:
- **Port 8080**: World data (terrain and voxel grids)
- **Port 8081**: Agent data (positions, velocities, commands)


### Binary Protocol
High-performance protocol for real-time simulation:
- **Format**: 4-byte little-endian length prefix + bincode data
- **World Data**: Serialized `VoxelGrid` struct
- **Agent Data**: Serialized `Vec<Agent>` array

**Reading binary data (Rust example):**
```rust
// Read 4-byte length prefix
let mut len_buf = [0u8; 4];
stream.read_exact(&mut len_buf)?;
let msg_len = u32::from_le_bytes(len_buf) as usize;

// Read message data
let mut msg_buf = vec![0u8; msg_len];
stream.read_exact(&mut msg_buf)?;

// Deserialize
let world: VoxelGrid = bincode::serde::decode_from_slice(
    &msg_buf, 
    bincode::config::standard()
)?.0;
```

## Usage with VoxelSim

### From Rust
```rust
use voxelsim::*;

// Create simulation
let mut world = VoxelGrid::new();
world.generate_terrain(&TerrainConfig::default());
let agents = vec![Agent::new(0)];

// Connect to renderer
let mut client = RendererClient::new("127.0.0.1", 8080, 8081);
client.connect()?;

// Send data (binary - fast)
client.send_world(&world)?;
client.send_agents(&agents)?;
```

### From Python
```python
import voxelsim

# Create simulation
world = voxelsim.VoxelGrid()
world.generate_default_terrain(42)
agent = voxelsim.Agent(0)
env = voxelsim.GlobalEnv(world, [agent])

# Connect to renderer  
client = voxelsim.RendererClient("127.0.0.1", 8080, 8081)
client.connect_py()

# Send data to renderer
env.send_world(client)
env.send_agents(client)
```

## Controls

### Camera Controls
- **Mouse Movement**: Orbit camera around the scene center
- **Mouse Scroll**: Zoom in and out
- **Middle Mouse**: Pan camera position
- **Right Click + Drag**: Free-look camera rotation

### Agent Controls
- **Z**: Cycle camera focus between agents
- **A/B/C/D**: Send movement commands to focused agent
- **Space**: Toggle agent trail visualization
- **Tab**: Toggle UI overlay with performance metrics

### World Controls  
- **R**: Request world regeneration (if simulator supports it)
- **G**: Toggle voxel grid wireframe overlay
- **L**: Toggle lighting effects
- **F**: Toggle fullscreen mode
- **Escape**: Exit application

## Rendering Features

### Voxel Rendering
- **Mesh optimization**: Automatic face culling for hidden voxels
- **Texture mapping**: Different materials for terrain types
- **Level-of-detail**: Distance-based mesh simplification
- **Instanced rendering**: Efficient grass and small object rendering

### Terrain Visualization
- **Procedural trees**: Multiple tree types (Oak, Pine, Birch, Willow)
- **Ground textures**: Grass, dirt, stone, and sand materials
- **Natural features**: Caves, cliffs, and water bodies
- **Environmental effects**: Fog, shadows, and ambient lighting

### Agent Visualization
- **3D models**: Distinct visual representation for each agent
- **Movement trails**: Configurable path history visualization
- **Status indicators**: Health, energy, and command queues
- **Physics debugging**: Velocity vectors and collision boundaries

## Performance

The renderer is optimized for real-time simulation with:
- **60+ FPS**: Smooth rendering even with large worlds
- **Efficient networking**: Minimal latency data processing
- **Memory management**: Automatic cleanup of old data
- **Scalable rendering**: Handles 1000+ agents and 100k+ voxels

### Performance Monitoring
Press `Tab` to toggle the performance overlay showing:
- Frame rate (FPS)
- Network message rates
- Memory usage
- Active agent count
- Rendered voxel count

## Architecture

```
VoxelSim Simulator
        ↓ (TCP Socket: Binary)
Network Receiver (Tokio)
        ↓ (Event System)
Bevy ECS World Update
        ↓ (Component Systems)
3D Rendering Pipeline
        ↓ (GPU Shaders)
Screen Output
```

### Internal Components
- **NetworkPlugin**: Handles TCP connections and data parsing
- **WorldPlugin**: Manages voxel grid updates and mesh generation
- **AgentPlugin**: Tracks agent positions and rendering
- **CameraPlugin**: Implements interactive camera controls
- **UIPlugin**: Provides performance monitoring and debug info

## Troubleshooting

### Connection Issues
- **Port conflicts**: Ensure ports 8080/8081 are available
- **Firewall**: Check that local connections are allowed
- **Binding order**: Start renderer before connecting simulator

### Performance Issues
- **Large worlds**: Use level-of-detail settings for 500k+ voxels
- **Many agents**: Consider agent culling beyond 1000 units
- **Network lag**: Switch to binary protocol for better performance

### Rendering Issues  
- **Missing voxels**: Check that VoxelGrid serialization is complete
- **Agent positions**: Verify agent coordinate system matches world
- **Texture problems**: Ensure all required assets are bundled

## Development

### Adding New Features
1. Implement new Bevy systems in `src/` modules
2. Register systems with appropriate plugins
3. Add UI controls in `UIPlugin` if needed
4. Update network protocol if new data types required

### Debug Mode
Run with debug logging:
```bash
RUST_LOG=debug cargo run
```

### Asset Pipeline
Custom materials and models can be added to the `assets/` directory and loaded through Bevy's asset system.
