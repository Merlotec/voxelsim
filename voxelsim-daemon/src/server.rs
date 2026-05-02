use std::collections::{VecDeque, vec_deque};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::backend::{self, ControlBackend, ControlStep, PythonBackend};
use crate::controller::ConnectionInterface;

use std::sync::mpsc;
use voxelsim::AgentStateUpdate;
use voxelsim::{
    Action, ActionIntent, Agent, MoveDir, VoxelGrid,
    agent::chase::{FixedLookaheadChaser, TrajectoryChaser},
};

#[derive(Debug)]
enum Command {
    Takeoff { alt_m: f32 },
    EngagePython { script: String },
    EngageManual,
    Disengage,
    Land,
    Kill,
    Quit,
}

fn parse_command(line: &str) -> Option<Command> {
    let parts: Vec<&str> = line.trim().splitn(2, ' ').collect();
    match parts.as_slice() {
        ["TAKEOFF", rest] => rest
            .parse::<f32>()
            .ok()
            .map(|a| Command::Takeoff { alt_m: a }),
        ["ENGAGE", script] => Some(Command::EngagePython {
            script: (*script).to_string(),
        }),
        ["ENGAGE_MANUAL"] => Some(Command::EngageManual),
        ["DISENGAGE"] => Some(Command::Disengage),
        ["LAND"] => Some(Command::Land),
        ["KILL"] => Some(Command::Kill),
        ["QUIT"] => Some(Command::Quit),
        _ => None,
    }
}

// Manual input receiver backed by a channel from the socket loop
struct SocketActionReceiver {
    rx: mpsc::Receiver<AgentStateUpdate>,
}

impl backend::manual::ActionReceiver for SocketActionReceiver {
    fn try_recv_signal(&self) -> Option<AgentStateUpdate> {
        self.rx.try_recv().ok()
    }
}

// Use trait objects for the active backend

fn parse_manual_action(line: &str, agent: &Agent) -> Option<AgentStateUpdate> {
    // Format: ACTION <urgency> <yaw> <moves_csv>, where moves are MoveDir codes
    let mut it = line.trim().split_whitespace();
    if it.next()? != "ACTION" {
        return None;
    }
    let urgency = it.next()?.parse::<f64>().ok()?;
    let yaw = it.next()?.parse::<f64>().ok()?;
    let moves_csv = it.next()?;
    let mut seq = Vec::new();
    for tok in moves_csv.split(',') {
        let code = tok.parse::<i32>().ok()?;
        let md = MoveDir::try_from(code).ok()?;
        seq.push(md);
    }
    let intent = ActionIntent::new(urgency, yaw, seq);
    Some(AgentStateUpdate::Replace(VecDeque::from([intent])))
}

pub fn run_server(addr: &str) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    println!("Listening for commands on {}", addr);

    // Shared flight state
    let conn =
        Arc::new(ConnectionInterface::connect("udpin:0.0.0.0:14540").expect("connect mavlink"));
    let agent = Arc::new(Mutex::new(Agent::new(0)));
    let world = Arc::new(Mutex::new(VoxelGrid::new()));

    for stream in listener.incoming() {
        let stream = stream?;
        handle_client(stream, conn.clone(), agent.clone(), world.clone())
            .unwrap_or_else(|e| eprintln!("Client handling error: {}", e));
    }
    Ok(())
}

pub fn handle_client(
    mut stream: TcpStream,
    conn: Arc<ConnectionInterface>,
    agent: Arc<Mutex<Agent>>,
    world: Arc<Mutex<VoxelGrid>>,
) -> std::io::Result<()> {
    let peer = stream.peer_addr()?;
    stream.write_all(b"Hello from voxelsim-daemon.\n")?;
    stream.write_all(
        b"Commands: TAKEOFF <alt>, ENGAGE <script>, ENGAGE_MANUAL, DISENGAGE, LAND, KILL, QUIT\n",
    )?;
    stream.write_all(b"Manual input: ACTION <urgency> <yaw> <move_codes_csv>\n")?;

    stream.set_read_timeout(Some(Duration::from_millis(50)))?;
    let mut buf = Vec::<u8>::new();
    let mut engaged = false;
    let mut script_cache: Option<String> = None;
    let mut backend_opt: Option<Box<dyn ControlBackend>> = None;
    let mut tx_manual_opt: Option<mpsc::Sender<AgentStateUpdate>> = None;
    let mut chaser = FixedLookaheadChaser::default();
    let mut last_tick = Instant::now();

    loop {
        // Try read a line with timeout
        let mut tmp = [0u8; 1024];
        match stream.read(&mut tmp) {
            Ok(0) => { /* no data */ }
            Ok(n) => {
                buf.extend_from_slice(&tmp[..n]);
                while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                    let line = String::from_utf8_lossy(&buf[..pos]).to_string();
                    buf.drain(..=pos);
                    if let Some(cmd) = parse_command(&line) {
                        match cmd {
                            Command::Takeoff { alt_m } => {
                                takeoff(&conn, alt_m)?;
                                writeln!(stream, "OK TAKEOFF {}", alt_m)?;
                            }
                            Command::EngagePython { script } => {
                                // Build backend now so subsequent ticks are cheap
                                match PythonBackend::from_script(&script) {
                                    Ok(b) => {
                                        backend_opt = Some(Box::new(b) as Box<dyn ControlBackend>);
                                        engaged = true;
                                        script_cache = Some(script);
                                        writeln!(stream, "OK ENGAGED")?;
                                    }
                                    Err(e) => {
                                        writeln!(stream, "ERR ENGAGE: {}", e)?;
                                    }
                                }
                            }
                            Command::EngageManual => {
                                let (tx, rx) = mpsc::channel();
                                tx_manual_opt = Some(tx);
                                let receiver = SocketActionReceiver { rx };
                                let manual = backend::manual::ManualBackend::new(receiver);
                                backend_opt = Some(Box::new(manual) as Box<dyn ControlBackend>);
                                engaged = true;
                                writeln!(stream, "OK ENGAGED MANUAL")?;
                            }
                            Command::Disengage => {
                                engaged = false;
                                hover(&conn)?;
                                writeln!(stream, "OK DISENGAGED")?;
                            }
                            Command::Land => {
                                engaged = false;
                                land(&conn)?;
                                writeln!(stream, "OK LANDING")?;
                            }
                            Command::Kill => {
                                engaged = false;
                                kill(&conn)?;
                                writeln!(stream, "OK KILLED")?;
                            }
                            Command::Quit => {
                                engaged = false;
                                writeln!(stream, "BYE")?;
                                return Ok(());
                            }
                        }
                    } else {
                        // If in manual mode, allow ACTION lines to push actions
                        if let Some(tx) = &tx_manual_opt {
                            if let Ok(ag) = agent.lock() {
                                if let Some(action) = parse_manual_action(&line, &*ag) {
                                    let _ = tx.send(action);
                                    writeln!(stream, "OK ACTION QUEUED")?;
                                    continue;
                                }
                            }
                        }
                        writeln!(stream, "ERR Unknown command")?;
                    }
                }
            }
            Err(ref e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut => {}
            Err(e) => return Err(e),
        }

        // Offboard tick ~10 Hz
        if engaged {
            if backend_opt.is_some() {
                // compute dt
                let now = Instant::now();
                let dt = (now - last_tick).as_secs_f64().max(0.05);
                last_tick = now;

                // Update action
                if let Some(backend) = backend_opt.as_mut() {
                    let ag_snapshot = agent.lock().unwrap().clone();
                    let wg_snapshot = world.lock().unwrap().clone();
                    if let ControlStep {
                        update: Some(update),
                        min_sleep,
                    } = backend.update_action(&ag_snapshot, &wg_snapshot)
                    {
                        let mut ag = agent.lock().unwrap();
                        ag.update_state(update);
                    }
                }

                // Step chaser
                let (pos_vx, vel_vx, yaw) = {
                    let ag = agent.lock().unwrap();
                    let tgt = chaser.step_chase(&ag, dt);
                    match tgt.progress {
                        voxelsim::agent::chase::ActionProgress::ProgressTo(s, trim) => {
                            drop(ag);
                            let mut agm = agent.lock().unwrap();
                            if let voxelsim::agent::AgentState::Action(ref mut act) = agm.state {
                                let _ = act.update_progress(s, trim);
                            }
                        }
                        voxelsim::agent::chase::ActionProgress::Complete(state) => {
                            drop(ag);
                            let mut agm = agent.lock().unwrap();
                            agm.state = state;
                        }
                        _ => {}
                    }
                    (tgt.pos, tgt.vel, tgt.yaw as f32)
                };

                // Convert from voxelsim's nalgebra to this crate's nalgebra
                let _ = conn.send_waypoint(pos_vx, vel_vx, yaw);
            }
        }

        // pace loop
        thread::sleep(Duration::from_millis(20));
    }
}

fn takeoff(conn: &ConnectionInterface, alt_m: f32) -> std::io::Result<()> {
    // Arm
    let _ = conn.send_armed(true);
    // Hold a position at given altitude by streaming a few setpoints
    for _ in 0..30 {
        let pos = nalgebra::Vector3::new(0.0, 0.0, alt_m as f64);
        let vel = nalgebra::Vector3::new(0.0, 0.0, 0.0);
        let _ = conn.send_waypoint(pos, vel, 0.0);
        thread::sleep(Duration::from_millis(100));
    }
    Ok(())
}

fn hover(conn: &ConnectionInterface) -> std::io::Result<()> {
    // Send a few zero-velocity setpoints to keep the vehicle hovering
    for _ in 0..20 {
        let pos = nalgebra::Vector3::new(0.0, 0.0, 0.0);
        let vel = nalgebra::Vector3::new(0.0, 0.0, 0.0);
        let _ = conn.send_waypoint(pos, vel, 0.0);
        thread::sleep(Duration::from_millis(100));
    }
    Ok(())
}

fn land(conn: &ConnectionInterface) -> std::io::Result<()> {
    // Descend with a gentle setpoint for a short period, then disarm
    for step in (0..=20).rev() {
        let z = step as f64 * 0.1; // descend to 0
        let pos = nalgebra::Vector3::new(0.0, 0.0, z);
        let vel = nalgebra::Vector3::new(0.0, 0.0, -0.3);
        let _ = conn.send_waypoint(pos, vel, 0.0);
        thread::sleep(Duration::from_millis(200));
    }
    let _ = conn.send_armed(false);
    Ok(())
}

fn kill(conn: &ConnectionInterface) -> std::io::Result<()> {
    let _ = conn.send_kill();
    Ok(())
}

// old threaded backend loop removed; the server loop now drives offboard updates inline
