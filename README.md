# VoxelSim (vxs)

VoxelSim is a modular voxel-based simulation and rendering stack. The workspace contains Rust crates for world representation, planning, rendering, and dynamics, plus Python bindings (via PyO3) that expose a unified `voxelsim` module for scripting and experiments.

## Repository Layout

- `voxelsim/`: Core library — voxel world (`VoxelGrid`), agents, planning, networking, and Python bindings behind `feature = "python"`.
- `voxelsim-compute/`: Vision/compute pipeline used to synthesize agent POVs; includes `AgentVisionRenderer`, `FilterWorld`, and noise configuration. Python bindings behind `feature = "python"`.
- `voxelsim-renderer/`: Bevy-based renderer that listens on TCP ports and visualizes worlds, agents, and POV feeds.
- `voxelsim-simulator/`: Dynamics and terrain generation (quad dynamics, terrain generator). Optional `px4` submodule when built with `feature = "px4"`. Python bindings behind `feature = "python"`.
- `voxelsim-daemon/`, `voxelsim-agent/`: Supporting services (not required for basic Python workflows).
- `voxelsim-py/`: PyO3 shim crate that composes `voxelsim`, `voxelsim-compute`, and `voxelsim-simulator` into a single Python extension module named `voxelsim`.
- `python/`: Example and experiment scripts. Note: some are written against older bindings; see “Outdated Scripts” below.
- `px4-mc/`: PX4-related integration assets (only needed if using the PX4 dynamics feature).

## Quick Start

1) Build the renderer and start it:
- `cargo run -p voxelsim-renderer --release`
- Listens on:
  - World: `8080`
  - Agents: `8081`
  - POV streams (virtual world): from `8090`
  - POV streams (agents-of-POV): from `9090`

2) Install the Python bindings (`voxelsim`):
- Recommended (maturin):
  - `pip install maturin`
  - `maturin develop -m voxelsim-py/Cargo.toml`
- Alternative (no install):
  - `cargo build -p voxelsim-py --release`
  - Add the built extension dir to `PYTHONPATH`, e.g. `export PYTHONPATH=$PWD/voxelsim-py/target/release:$PYTHONPATH`

3) Run an example:
- `python python/povtest.py`

## Python Bindings (`voxelsim`)

The `voxelsim` module merges submodules from three Rust crates:
- Core (`voxelsim`): world, agents, planning, networking
- Compute (`voxelsim-compute`): POV rendering pipeline and filter world
- Simulator (`voxelsim-simulator`): terrain and dynamics (+ optional `px4` module)

Below is the current API surface most scripts should use. Methods with `_py` suffix are the Python-exposed wrappers.

### World and Cells
- `vxs.VoxelGrid.from_dict_py({(x,y,z): vxs.Cell.filled(), ...}) -> VoxelGrid`: Construct from a Python dict.
- `vxs.VoxelGrid.to_dict_py() -> Dict[Tuple[int,int,int], vxs.Cell]`: Export sparse cells.
- `vxs.VoxelGrid.as_numpy() -> (coords: np.ndarray[n,3], values: np.ndarray[n])`: Extract as arrays (1.0 filled, 0.5 sparse, 0.0 empty).
- `vxs.VoxelGrid.collisions_py(pos: [f64;3], dims: [f64;3]) -> List[((i32,i32,i32), vxs.Cell)]`: AABB vs voxel intersections near `pos`.
- `vxs.VoxelGrid.dense_snapshot_py(centre: [i32;3], half_dims: [i32;3]) -> DenseSnapshot`: Dense cuboid sample around a centre.
- `vxs.Cell.filled()`, `vxs.Cell.sparse()`: Constructors.
- `vxs.Cell.is_filled_py()`, `vxs.Cell.is_sparse_py()`, `vxs.Cell.bits_py()`.

### Agents, Actions, Planning, and View
- `vxs.Agent(id: int)`: Create an agent.
- `agent.set_hold_py(coord: [i32;3], yaw: float)`: Place and hold at grid coord with yaw.
- `agent.get_pos() -> [f64;3]`, `agent.get_coord_py() -> [i32;3]`.
- `agent.camera_view_py(orientation: vxs.CameraOrientation) -> vxs.CameraView`.
- `agent.perform_oneshot_py(intent: vxs.ActionIntent)`: Start a new action (overwrites any current one).
- `agent.push_back_intent_py(intent: vxs.ActionIntent)`: Queue an intent behind the current action.
- `agent.get_action_py() -> Optional[vxs.Action]`.

- `vxs.ActionIntent(urgency: float, yaw: float, move_sequence: List[vxs.MoveDir], next: Optional[vxs.ActionIntent])`.
- `intent.get_move_sequence() -> List[vxs.MoveDir]`, `intent.len() -> int`.
- `vxs.MoveDir`: directions enum
  - Named members: `vxs.MoveDir.Forward`, `Back`, `Left`, `Right`, `Up`, `Down`, `None`, `Undecided`
  - Helpers: `vxs.MoveDir.from_code_py(code: int)`, and classmethods like `up()`, `down()`, etc.

- Planning:
  - `vxs.AStarActionPlanner(padding: int)`: Initialize with obstacle padding radius.
- `planner.plan_action_py(world: VoxelGrid, origin: [i32;3], dst: [i32;3], urgency: float, yaw: float) -> vxs.ActionIntent`.

- View and camera:
  - `vxs.CameraProjection.new(aspect, fov_vertical, max_distance, near_distance)` or `vxs.CameraProjection.default_py()`.
  - `vxs.CameraOrientation.vertical_tilt_py(angle_radians)` (and related constructors).

### Compute Pipeline (POV Generation)
- `vxs.FilterWorld()`: Shared, thread-safe virtual world for POV visualization and dense sampling.
  - `fw.send_pov_py(client, stream_idx, agent_id, proj, orientation)`: Send current POV to the renderer.
  - `fw.send_pov_async_py(async_client, ...)`: Async variant.
  - `fw.is_updating_py(timestamp: float) -> bool`: True if an update for `timestamp` is in-progress.
  - `fw.dense_snapshot_py(centre, half_dims) -> DenseSnapshot`, `fw.as_numpy() -> (coords, values)`.
  - `fw.timestamp_py() -> Optional[float]`.
- `vxs.NoiseParams.default_with_seed_py([x,y,z])` or `vxs.NoiseParams.none_py()`.
- `vxs.AgentVisionRenderer(world: VoxelGrid, view_size: [u32;2], noise: NoiseParams)`.
  - `renderer.update_filter_world_py(camera_view, proj, fw, timestamp, callback)`
  - `renderer.update_filter_world_with_uncertainty_py(camera_view, proj, fw, unc_world, timestamp, callback)`
  - `renderer.render_changeset_py(camera_view, proj, fw, timestamp, callback)`
- `vxs.UncertaintyWorld.new_py(origin: [f64;3], node_size: f64)`, `vxs.UncertaintyWorld.default_py()`.

### Networking (Renderer Client)
- `vxs.RendererClient.default_localhost_py(pov_count: int) -> RendererClient`: Connects to world/agent sockets plus `pov_count` POV streams starting at default ports.
- `client.send_world_py(world)`, `client.send_agents_py({id: agent, ...})`.
- Async variant: `vxs.AsyncRendererClient.default_localhost_py(pov_count)`; call `send_world_py`, `send_agents_py` on it.

### Simulator (Terrain & Dynamics)
- Terrain:
  - `vxs.TerrainGenerator()`, `vxs.TerrainConfig.default_py()`.
  - `gen.generate_terrain_py(cfg)`, `gen.generate_world_py() -> VoxelGrid`.
- Quad dynamics:
  - `vxs.QuadParams.default_py()`
  - `vxs.QuadDynamics(params)`
  - `dyn.update_agent_dynamics_py(agent, env, chase_target, delta)`
- Environment state:
  - `vxs.EnvState.default_py()`
- Chasing:
  - `vxs.FixedLookaheadChaser.default_py()`
  - `chaser.step_chase_py(agent, dt) -> vxs.ChaseTarget`

Optional PX4 module (feature-gated):
- Build with `maturin develop -m voxelsim-py/Cargo.toml --features px4` (or `cargo build -p voxelsim-py --release --features px4`).
- Access as `vxs.px4.Px4Dynamics.default_py()` and `vxs.px4.Px4SettingsPy`.

## Example (Minimal Loop)

```python
import voxelsim as vxs, time

# World
gen = vxs.TerrainGenerator()
gen.generate_terrain_py(vxs.TerrainConfig.default_py())
world = gen.generate_world_py()

# Agent
agent = vxs.Agent(0)
agent.set_hold_py([50, 50, -20], 0.0)

# POV setup
fw = vxs.FilterWorld()
proj = vxs.CameraProjection.default_py()
cam = vxs.CameraOrientation.vertical_tilt_py(-0.5)
noise = vxs.NoiseParams.default_with_seed_py([0.0, 0.0, 0.0])
renderer = vxs.AgentVisionRenderer(world, [200, 150], noise)

# Network client (1 POV stream)
client = vxs.RendererClient.default_localhost_py(1)
client.send_world_py(world)
client.send_agents_py({0: agent})

last_ts = time.time()
while True:
    now = time.time()
    if not fw.is_updating_py(last_ts) and (now - last_ts) > 0.1:
        fw.send_pov_py(client, 0, 0, proj, cam)
        renderer.update_filter_world_py(agent.camera_view_py(cam), proj, fw, now, lambda *_: None)
        last_ts = now
    time.sleep(0.01)
```

## Outdated Scripts (and how to update)

Some scripts in `python/` target older versions of the bindings. The current API is reflected in `python/povtest.py`. Known outdated patterns:

- Renderer client connection:
  - Old: `client = vxs.RendererClient.default_localhost_py()` + `client.connect_py(1)`
  - New: `client = vxs.RendererClient.default_localhost_py(pov_count)` (no separate `connect_py`)

- POV renderer construction:
  - Old: `renderer = vxs.AgentVisionRenderer(world, [w, h])`
  - New: `renderer = vxs.AgentVisionRenderer(world, [w, h], vxs.NoiseParams...)`

- Agent action APIs:
  - Old: `agent.perform_py(intent)`; `action.get_intent()`
  - New: `agent.perform_oneshot_py(intent)`; `action.get_intent_queue()` (list)

- Dynamics types:
  - Old: `voxelsim.PengQuadDynamics` (nonexistent)
  - New: `vxs.QuadDynamics(vxs.QuadParams.default_py())` or `vxs.px4.Px4Dynamics.default_py()` when built with `--features px4`.

- Client POV streaming:
  - Use `FilterWorld.send_pov_py(client, stream_idx, agent_id, proj, orientation)` (not methods directly on the client).

Files with legacy usage to update or use as reference only:
- `python/povtest_legacy.py`
- `python/sim_from_world.py`
- `python/test.py`

For up-to-date references, prefer:
- `python/povtest.py`

## Building the Workspace

- Release build of all crates: `cargo build --release`
- Individual crates: `cargo build -p voxelsim[...specific...] --release`
- Python extension only: see “Install the Python bindings” above.

## Ports and Protocol

- All network messages are framed as: 4-byte little-endian length prefix + bincode payload.
- Default ports:
  - World: `8080`
  - Agents: `8081`
  - POV virtual world streams: start at `8090` (N streams)
  - POV agent streams: start at `9090` (N streams)

## Troubleshooting

- ImportError: `ModuleNotFoundError: No module named 'voxelsim'`
  - Ensure the extension is installed with `maturin develop -m voxelsim-py/Cargo.toml`, or that `PYTHONPATH` includes the built `voxelsim` extension directory.
- Missing `vxs.px4` submodule
  - Reinstall/build with `--features px4` if you need PX4 dynamics.
- Renderer not receiving POV
  - Verify `pov_count` matches the number of POV streams you intend to use and the renderer is running.
