pub mod dynamics;
pub mod terrain;

// Re-export core types for Rust consumers
pub use terrain::*;

// Python bindings - only compile when python feature is enabled
#[cfg(feature = "python")]
pub mod py;
