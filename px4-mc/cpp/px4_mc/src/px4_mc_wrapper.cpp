#include "px4_mc_wrapper.hpp"

#include <memory>
#include <cstring>
#include <cmath>


#include <matrix/matrix/math.hpp>
#include <uORB/topics/trajectory_setpoint.h>
#include <uORB/topics/vehicle_attitude_setpoint.h>
#include <uORB/topics/vehicle_local_position_setpoint.h>
#include <modules/mc_att_control/AttitudeControl/AttitudeControl.hpp>
#include <modules/mc_pos_control/PositionControl/PositionControl.hpp>
#include <lib/rate_control/rate_control.hpp>


// This wrapper aggregates PX4 controllers (position, attitude, rate)
// and exposes a simple C interface. If PX4 sources are not compiled
// into this library, the update call returns zeros.

namespace {

struct Px4McImpl {
    float dt;
    float mass;

    AttitudeControl att{};
    PositionControl pos{};
    RateControl rate{};

    // Tunables (approximate defaults; adjust as needed)
    matrix::Vector3f att_p{6.f, 6.f, 2.f};
    float att_yaw_weight{0.5f};
    matrix::Vector3f att_rate_limit{3.0f, 3.0f, 1.5f};

    matrix::Vector3f pos_p{4.f, 4.f, 6.f};
    matrix::Vector3f vel_p{6.f, 6.f, 4.f};
    matrix::Vector3f vel_i{0.5f, 0.5f, 0.5f};
    matrix::Vector3f vel_d{0.1f, 0.1f, 0.0f};
    float vel_lim_xy{8.f};
    float vel_up{4.f};
    float vel_down{2.f};
    float thr_min{0.05f};
    float thr_max{0.9f};
    float thr_xy_margin{0.2f};
    float tilt_max_rad{0.6f};
    float hover_thrust{0.5f}; // normalized [0,1]

    // Simple fallback rate gains if RateControl class not available
    matrix::Vector3f rate_p{0.15f, 0.15f, 0.1f};
    matrix::Vector3f rate_i{0.05f, 0.05f, 0.03f};
    matrix::Vector3f rate_d{0.001f, 0.001f, 0.0f};
    matrix::Vector3f rate_int_lim{0.3f, 0.3f, 0.2f};

    // Scale normalized torque [-1,1] to physical [N*m]
    matrix::Vector3f torque_scale_nm{0.8f, 0.8f, 0.3f};

    // Simple velocity derivative estimator for PositionControl D term
    matrix::Vector3f prev_vel{0.f, 0.f, 0.f};
    bool prev_vel_valid{false};

    Px4McImpl(float dt_, float mass_) : dt(dt_), mass(mass_) {
        // Configure controllers with reasonable defaults
        att.setProportionalGain(att_p, att_yaw_weight);
        att.setRateLimit(att_rate_limit);

        pos.setPositionGains(pos_p);
        pos.setVelocityGains(vel_p, vel_i, vel_d);
        pos.setVelocityLimits(vel_lim_xy, vel_up, vel_down);
        pos.setThrustLimits(thr_min, thr_max);
        pos.setHorizontalThrustMargin(thr_xy_margin);
        pos.setTiltLimit(tilt_max_rad);
        pos.setHoverThrust(hover_thrust);

        // Initialize rate controller gains (tunable via setters)
        rate.setPidGains(rate_p, rate_i, rate_d);
        rate.setIntegratorLimit(rate_int_lim);
    }

    void update(const Px4McInput& in, Px4McOutput& out) {
        using matrix::Vector3f; using matrix::Quatf; using matrix::Eulerf;

        const float dt_local = (in.dt > 0.f) ? in.dt : dt;
        const Vector3f pos_xyz{in.pos[0], in.pos[1], in.pos[2]};
        const Vector3f vel_xyz{in.vel[0], in.vel[1], in.vel[2]};
        const Vector3f rates{in.rates[0], in.rates[1], in.rates[2]};
        const Vector3f pos_sp{in.target_pos[0], in.target_pos[1], in.target_pos[2]};
        const Vector3f vel_sp{in.target_vel[0], in.target_vel[1], in.target_vel[2]};
        const Quatf q{in.att_q[0], in.att_q[1], in.att_q[2], in.att_q[3]}; // w, x, y, z

        // Position control to get thrust and yaw setpoints
        PositionControlStates st{};
        st.position = pos_xyz;
        st.velocity = vel_xyz;
        // Provide velocity derivative (acceleration) for D term damping
        Vector3f vel_dot{0.f, 0.f, 0.f};
        if (prev_vel_valid && dt_local > 1e-6f) {
            vel_dot = (vel_xyz - prev_vel) / dt_local;
        }
        prev_vel = vel_xyz;
        prev_vel_valid = true;
        st.acceleration = vel_dot;
        st.yaw = Eulerf(q).psi();
        pos.setState(st);

        trajectory_setpoint_s traj{};
        traj.position[0] = pos_sp(0); traj.position[1] = pos_sp(1); traj.position[2] = pos_sp(2);
        traj.velocity[0] = vel_sp(0); traj.velocity[1] = vel_sp(1); traj.velocity[2] = vel_sp(2);
        traj.acceleration[0] = NAN; traj.acceleration[1] = NAN; traj.acceleration[2] = NAN;
        // Yaw setpoint: use provided yaw/yawspeed if finite, otherwise hold current yaw and zero yawspeed
        traj.yaw = std::isfinite(in.yaw_sp) ? in.yaw_sp : st.yaw;
        traj.yawspeed = std::isfinite(in.yawspeed_sp) ? in.yawspeed_sp : 0.f;
        pos.setInputSetpoint(traj);

        const bool ok = pos.update(dt_local);

        vehicle_attitude_setpoint_s att_sp{};
        pos.getAttitudeSetpoint(att_sp);

        // Extract attitude and thrust setpoints
        Quatf q_sp{att_sp.q_d[0], att_sp.q_d[1], att_sp.q_d[2], att_sp.q_d[3]};
        // Note: thrust_body is expected in NED body frame; take magnitude as scalar thrust
        const Vector3f thrust_body{att_sp.thrust_body[0], att_sp.thrust_body[1], att_sp.thrust_body[2]};

        // Attitude control to body rates setpoint
        att.setAttitudeSetpoint(q_sp, att_sp.yawspeed);
        Vector3f rates_sp = att.update(q);

        // Rate control to torques
        // Rate control to torques using PX4 RateControl core library
        const Vector3f torque_norm = rate.update(rates, rates_sp, Vector3f{}, dt_local, false);
        const Vector3f torque = torque_norm.emult(torque_scale_nm);
        out.torque[0] = torque(0);
        out.torque[1] = torque(1);
        out.torque[2] = torque(2);

        // Output PX4 body thrust z-component (FRD: z downwards, typically negative for lift)
        out.thrust = thrust_body(2);
        (void)ok; // placeholder: consumer may inspect success later
        return;
    }
};

} // namespace

extern "C" {

struct Px4McOpaque { std::unique_ptr<Px4McImpl> inner; };

Px4McOpaque* px4_mc_create(const Px4McSettings* s) {
    try {
        if (!s) return nullptr;
        auto handle = new Px4McOpaque{std::make_unique<Px4McImpl>(s->dt, s->mass)};
        // Apply full settings once constructed
        px4_mc_apply_settings(handle, s);
        return handle;
    } catch (...) {
        return nullptr;
    }
}

void px4_mc_destroy(Px4McOpaque* ptr) {
    if (ptr) {
        delete ptr;
    }
}

void px4_mc_update(Px4McOpaque* ptr, const Px4McInput* in, Px4McOutput* out) {
    if (!ptr || !in || !out) return;
    ptr->inner->update(*in, *out);
}

// Helpers
static inline matrix::Vector3f to_vec3(const float v[3]) { return {v[0], v[1], v[2]}; }

void px4_mc_set_att_gains(Px4McOpaque* ptr, const float p[3], float yaw_weight) {
    if (!ptr) return;
    ptr->inner->att_p = to_vec3(p);
    ptr->inner->att_yaw_weight = yaw_weight;
    ptr->inner->att.setProportionalGain(ptr->inner->att_p, ptr->inner->att_yaw_weight);
}

void px4_mc_set_att_rate_limit(Px4McOpaque* ptr, const float rate_limit[3]) {
    if (!ptr) return;
    ptr->inner->att_rate_limit = to_vec3(rate_limit);
    ptr->inner->att.setRateLimit(ptr->inner->att_rate_limit);
}

void px4_mc_set_pos_gains(Px4McOpaque* ptr, const float p[3]) {
    if (!ptr) return;
    ptr->inner->pos_p = to_vec3(p);
    ptr->inner->pos.setPositionGains(ptr->inner->pos_p);
}

void px4_mc_set_vel_gains(Px4McOpaque* ptr, const float p[3], const float i[3], const float d[3]) {
    if (!ptr) return;
    ptr->inner->vel_p = to_vec3(p);
    ptr->inner->vel_i = to_vec3(i);
    ptr->inner->vel_d = to_vec3(d);
    ptr->inner->pos.setVelocityGains(ptr->inner->vel_p, ptr->inner->vel_i, ptr->inner->vel_d);
}

void px4_mc_set_vel_limits(Px4McOpaque* ptr, float vel_horizontal, float vel_up, float vel_down) {
    if (!ptr) return;
    ptr->inner->vel_lim_xy = vel_horizontal;
    ptr->inner->vel_up = vel_up;
    ptr->inner->vel_down = vel_down;
    ptr->inner->pos.setVelocityLimits(ptr->inner->vel_lim_xy, ptr->inner->vel_up, ptr->inner->vel_down);
}

void px4_mc_set_thrust_limits(Px4McOpaque* ptr, float min_thr, float max_thr) {
    if (!ptr) return;
    ptr->inner->thr_min = min_thr;
    ptr->inner->thr_max = max_thr;
    ptr->inner->pos.setThrustLimits(ptr->inner->thr_min, ptr->inner->thr_max);
}

void px4_mc_set_thr_xy_margin(Px4McOpaque* ptr, float margin) {
    if (!ptr) return;
    ptr->inner->thr_xy_margin = margin;
    ptr->inner->pos.setHorizontalThrustMargin(ptr->inner->thr_xy_margin);
}

void px4_mc_set_tilt_limit(Px4McOpaque* ptr, float tilt_rad) {
    if (!ptr) return;
    ptr->inner->tilt_max_rad = tilt_rad;
    ptr->inner->pos.setTiltLimit(ptr->inner->tilt_max_rad);
}

void px4_mc_set_hover_thrust(Px4McOpaque* ptr, float hover) {
    if (!ptr) return;
    ptr->inner->hover_thrust = hover;
    ptr->inner->pos.setHoverThrust(ptr->inner->hover_thrust);
}

void px4_mc_set_rate_p(Px4McOpaque* ptr, const float p[3]) {
    if (!ptr) return;
    ptr->inner->rate_p = to_vec3(p);
}

void px4_mc_set_dt(Px4McOpaque* ptr, float dt) {
    if (!ptr) return;
    ptr->inner->dt = dt;
}

void px4_mc_set_mass(Px4McOpaque* ptr, float mass) {
    if (!ptr) return;
    ptr->inner->mass = mass;
}

void px4_mc_apply_settings(Px4McOpaque* ptr, const Px4McSettings* s) {
    if (!ptr || !s) return;
    // Basic
    ptr->inner->dt = s->dt;
    ptr->inner->mass = s->mass;
    // Attitude
    px4_mc_set_att_gains(ptr, s->att_p, s->att_yaw_weight);
    px4_mc_set_att_rate_limit(ptr, s->att_rate_limit);
    // Position/Velocity
    px4_mc_set_pos_gains(ptr, s->pos_p);
    px4_mc_set_vel_gains(ptr, s->vel_p, s->vel_i, s->vel_d);
    px4_mc_set_vel_limits(ptr, s->vel_lim_xy, s->vel_up, s->vel_down);
    px4_mc_set_thrust_limits(ptr, s->thr_min, s->thr_max);
    px4_mc_set_thr_xy_margin(ptr, s->thr_xy_margin);
    px4_mc_set_tilt_limit(ptr, s->tilt_max_rad);
    px4_mc_set_hover_thrust(ptr, s->hover_thrust);
    // Behavior toggles
    ptr->inner->pos.decoupleHorizontalAndVecticalAcceleration(s->decouple_horiz_vert_accel != 0);
    // Rate (PID)
    ptr->inner->rate_p = to_vec3(s->rate_p);
    ptr->inner->rate_i = to_vec3(s->rate_i);
    ptr->inner->rate_d = to_vec3(s->rate_d);
    ptr->inner->rate_int_lim = to_vec3(s->rate_int_lim);
    ptr->inner->rate.setPidGains(ptr->inner->rate_p, ptr->inner->rate_i, ptr->inner->rate_d);
    ptr->inner->rate.setIntegratorLimit(ptr->inner->rate_int_lim);

    // Torque scaling
    ptr->inner->torque_scale_nm = matrix::Vector3f{s->torque_scale_nm[0], s->torque_scale_nm[1], s->torque_scale_nm[2]};
}

}
