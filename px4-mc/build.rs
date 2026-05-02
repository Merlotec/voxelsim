use std::env;
use std::path::PathBuf;

fn main() {
    // PX4_SRC_DIR can point at a live PX4-Autopilot checkout.
    // If unset, CMake falls back to the bundled vendor tree.
    let mut cfg = cmake::Config::new("cpp/px4_mc");

    // Rebuild triggers when environment or local sources change
    println!("cargo:rerun-if-env-changed=PX4_SRC_DIR");
    println!("cargo:rerun-if-changed=cpp/px4_mc/CMakeLists.txt");
    println!("cargo:rerun-if-changed=cpp/px4_mc/px4_sources.cmake");
    println!("cargo:rerun-if-changed=cpp/px4_mc/src/px4_mc_wrapper.cpp");
    println!("cargo:rerun-if-changed=cpp/px4_mc/include/px4_mc_wrapper.hpp");
    println!("cargo:rerun-if-changed=cpp/px4_mc/include/px4_boardconfig.h");
    println!("cargo:rerun-if-changed=cpp/px4_mc/include/matrix/math.hpp");
    println!("cargo:rerun-if-changed=cpp/px4_mc/include/uORB/topics/trajectory_setpoint.h");
    println!("cargo:rerun-if-changed=cpp/px4_mc/include/uORB/topics/vehicle_attitude_setpoint.h");
    println!(
        "cargo:rerun-if-changed=cpp/px4_mc/include/uORB/topics/vehicle_local_position_setpoint.h"
    );
    println!("cargo:rerun-if-changed=cpp/px4_mc/include/uORB/topics/rate_ctrl_status.h");
    // Rebuild if any vendored source changes
    println!("cargo:rerun-if-changed=cpp/px4_mc/vendor");

    if let Ok(px4_src) = env::var("PX4_SRC_DIR") {
        cfg.define("PX4_SRC_DIR", &px4_src);
    }

    // Build the C++ static library with CMake
    let dst = cfg.build();

    // Link the produced static lib
    let mut libdir = PathBuf::from(dst);
    libdir.push("lib");
    println!("cargo:rustc-link-search=native={}", libdir.display());
    println!("cargo:rustc-link-lib=static=px4_mc");

    // Link the C++ stdlib depending on target
    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("apple-darwin") {
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("windows") {
        // MSVC links C++ runtime automatically
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
