use ::voxelsim as voxelsim_core;
use pyo3::prelude::*;

#[pymodule]
fn voxelsim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    voxelsim_core::py::voxelsim_core(m)?;
    voxelsim_compute::py::voxelsim_compute(m)?;
    voxelsim_simulator::py::voxelsim_simulator(m)?;
    Ok(())
}
