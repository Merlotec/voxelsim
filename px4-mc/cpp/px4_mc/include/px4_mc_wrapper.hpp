#pragma once

#include <cstdint>

extern "C" {

struct Px4McOpaque;

struct Px4McInput {
    float pos[3];
    float vel[3];
    float att_q[4]; // w, x, y, z
    float rates[3];
    float target_pos[3];
    float target_vel[3];
    float dt; // simulation timestep [s]
    float yaw_sp;      // desired yaw [rad], NAN to hold
    float yawspeed_sp; // desired yaw rate [rad/s], NAN if none
};

struct Px4McOutput {
    float torque[3];
    float thrust;
};

// Bulk controller settings for convenient FFI configuration
struct Px4McSettings {
    float dt;
    float mass;

    float att_p[3];
    float att_yaw_weight;
    float att_rate_limit[3];

    float pos_p[3];
    float vel_p[3];
    float vel_i[3];
    float vel_d[3];

    float vel_lim_xy;
    float vel_up;
    float vel_down;

    float thr_min;
    float thr_max;
    float thr_xy_margin;
    float tilt_max_rad;
    float hover_thrust;

    float rate_p[3];
    float rate_i[3];
    float rate_d[3];
    float rate_int_lim[3];

    // Behavior toggles
    uint8_t decouple_horiz_vert_accel; // bool (0/1)

    // Scale normalized PX4 torque [-1,1] to physical torque [N*m]
    // Applied per-axis after RateControl::update
    float torque_scale_nm[3];
};

Px4McOpaque* px4_mc_create(const Px4McSettings* settings);
void px4_mc_destroy(Px4McOpaque* ptr);
void px4_mc_update(Px4McOpaque* ptr, const Px4McInput* in, Px4McOutput* out);
void px4_mc_apply_settings(Px4McOpaque* ptr, const Px4McSettings* settings);

// Parameter setters
void px4_mc_set_att_gains(Px4McOpaque* ptr, const float p[3], float yaw_weight);
void px4_mc_set_att_rate_limit(Px4McOpaque* ptr, const float rate_limit[3]);
void px4_mc_set_pos_gains(Px4McOpaque* ptr, const float p[3]);
void px4_mc_set_vel_gains(Px4McOpaque* ptr, const float p[3], const float i[3], const float d[3]);
void px4_mc_set_vel_limits(Px4McOpaque* ptr, float vel_horizontal, float vel_up, float vel_down);
void px4_mc_set_thrust_limits(Px4McOpaque* ptr, float min_thr, float max_thr);
void px4_mc_set_thr_xy_margin(Px4McOpaque* ptr, float margin);
void px4_mc_set_tilt_limit(Px4McOpaque* ptr, float tilt_rad);
void px4_mc_set_hover_thrust(Px4McOpaque* ptr, float hover);
void px4_mc_set_rate_p(Px4McOpaque* ptr, const float p[3]);
void px4_mc_set_dt(Px4McOpaque* ptr, float dt);
void px4_mc_set_mass(Px4McOpaque* ptr, float mass);

}
