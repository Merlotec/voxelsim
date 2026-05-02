#pragma once
#include <stdint.h>

// Minimal local position setpoint; sufficient for PositionControl::getLocalPositionSetpoint signature.
struct vehicle_local_position_setpoint_s {
    uint64_t timestamp;
    float x;
    float y;
    float z;
    float yaw;
    float yawspeed;
    float vx;
    float vy;
    float vz;
    float acceleration[3];
    float thrust[3];
};

