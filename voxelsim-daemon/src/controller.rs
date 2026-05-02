use std::{error::Error, io};

use mavlink::{
    MavHeader,
    ardupilotmega::{
        COMMAND_LONG_DATA, HEARTBEAT_DATA, LED_CONTROL_DATA, MavCmd, MavFrame, MavMessage,
        PositionTargetTypemask, SET_POSITION_TARGET_LOCAL_NED_DATA,
    },
    error::MessageWriteError,
};
use nalgebra::Vector3;
// Cartesian setpoints use scalar inputs

mod ctrl {
    pub const LED_CONTROL_PATTERN_OFF: u8 = 0;
    pub const LED_CONTROL_PATTERN_CUSTOM: u8 = 255;
}

struct GeoRef {
    origin_lat_deg: f64,
    origin_lon_deg: f64,
    origin_alt_m: f64,
    meters_per_unit: f64, // scale for normalized/world units -> meters
}

impl Default for GeoRef {
    fn default() -> Self {
        Self {
            origin_lat_deg: 0.0,
            origin_lon_deg: 0.0,
            origin_alt_m: 0.0,
            meters_per_unit: 1.0,
        }
    }
}

pub struct ConnectionInterface {
    conn: Box<dyn mavlink::MavConnection<MavMessage>>,
    target_system: u8,
    target_component: u8,
    georef: GeoRef,
}

impl ConnectionInterface {
    pub fn connect(address: &str) -> io::Result<Self> {
        let conn = mavlink::connect(address)?;
        Ok(Self {
            conn,
            target_system: 1,
            target_component: 1,
            georef: GeoRef::default(),
        })
    }

    pub fn set_georef(
        &mut self,
        origin_lat_deg: f64,
        origin_lon_deg: f64,
        origin_alt_m: f64,
        meters_per_unit: f64,
    ) {
        self.georef = GeoRef {
            origin_lat_deg,
            origin_lon_deg,
            origin_alt_m,
            meters_per_unit,
        };
    }

    pub fn send_heartbeat(&self) -> Result<usize, MessageWriteError> {
        self.conn.send(
            &MavHeader::default(),
            &MavMessage::HEARTBEAT(HEARTBEAT_DATA::DEFAULT),
        )
    }

    pub fn send_led(&self, colour: [u8; 3], blink_rate: u8) -> Result<usize, MessageWriteError> {
        // build a custom‐pattern LED_CONTROL message
        let mut bytes = [0u8; 24];
        bytes[0..2].copy_from_slice(&colour);
        bytes[3] = blink_rate; // 2 Hz blink rate
        self.conn.send(
            &MavHeader::default(),
            &MavMessage::LED_CONTROL(LED_CONTROL_DATA {
                target_system: 1,
                target_component: 1,
                instance: 255,
                pattern: ctrl::LED_CONTROL_PATTERN_CUSTOM,
                custom_len: 4,
                custom_bytes: bytes,
            }),
        )
    }

    pub fn send_armed(&self, armed: bool) -> Result<usize, MessageWriteError> {
        // PX4 arming/disarming via COMMAND_LONG with MAV_CMD_COMPONENT_ARM_DISARM
        let msg = MavMessage::COMMAND_LONG(COMMAND_LONG_DATA {
            target_system: self.target_system,
            target_component: self.target_component,
            command: MavCmd::MAV_CMD_COMPONENT_ARM_DISARM,
            confirmation: 0,
            // param1: 1 = arm, 0 = disarm
            param1: if armed { 1.0 } else { 0.0 },
            // param2: 21196 to force in PX4 (0.0 for normal); keep 0.0 by default
            param2: 0.0,
            param3: 0.0,
            param4: 0.0,
            param5: 0.0,
            param6: 0.0,
            param7: 0.0,
        });
        self.conn.send(&MavHeader::default(), &msg)
    }

    pub fn send_kill(&self) -> Result<usize, MessageWriteError> {
        // Flight termination: immediately stop motors. Use with extreme caution.
        let msg = MavMessage::COMMAND_LONG(COMMAND_LONG_DATA {
            target_system: self.target_system,
            target_component: self.target_component,
            command: MavCmd::MAV_CMD_DO_FLIGHTTERMINATION,
            confirmation: 0,
            param1: 1.0, // 1 = terminate
            param2: 0.0,
            param3: 0.0,
            param4: 0.0,
            param5: 0.0,
            param6: 0.0,
            param7: 0.0,
        });
        self.conn.send(&MavHeader::default(), &msg)
    }

    pub fn send_waypoint(
        &self,
        pos: Vector3<f64>,
        vel: Vector3<f64>,
        yaw_world_rad: f32,
    ) -> Result<usize, MessageWriteError> {
        // Cartesian setpoint via SET_POSITION_TARGET_LOCAL_NED with position, velocity, and yaw.
        // World frame: X=East, Y=North, Z=Up. PX4 LOCAL_NED: X=North, Y=East, Z=Down.

        let scale = self.georef.meters_per_unit;
        let x_m = pos.x * scale;
        let y_m = pos.y * scale;
        let z_m = pos.z * scale;
        let vx_m = vel.x * scale;
        let vy_m = vel.y * scale;
        let vz_m = vel.z * scale;

        // ENU (world) to NED mapping expected by PX4 LOCAL_NED
        let x_ned = y_m; // north
        let y_ned = x_m; // east
        let z_ned = -z_m; // down

        let vx_ned = vy_m; // north
        let vy_ned = vx_m; // east
        let vz_ned = -vz_m; // down

        // Convert yaw from ENU (0=East, CCW) to NED (0=North, CW)
        let yaw_ned = std::f32::consts::FRAC_PI_2 - yaw_world_rad;

        // Include pos, vel, yaw; ignore acceleration and yaw_rate
        let type_mask = PositionTargetTypemask::POSITION_TARGET_TYPEMASK_AX_IGNORE
            | PositionTargetTypemask::POSITION_TARGET_TYPEMASK_AY_IGNORE
            | PositionTargetTypemask::POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | PositionTargetTypemask::POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE;

        let msg = MavMessage::SET_POSITION_TARGET_LOCAL_NED(SET_POSITION_TARGET_LOCAL_NED_DATA {
            time_boot_ms: 0,
            target_system: self.target_system,
            target_component: self.target_component,
            coordinate_frame: MavFrame::MAV_FRAME_LOCAL_NED,
            type_mask,
            x: x_ned as f32,
            y: y_ned as f32,
            z: z_ned as f32,
            vx: vx_ned as f32,
            vy: vy_ned as f32,
            vz: vz_ned as f32,
            afx: 0.0,
            afy: 0.0,
            afz: 0.0,
            yaw: yaw_ned,
            yaw_rate: 0.0,
        });

        self.conn.send(&MavHeader::default(), &msg)
    }
}
