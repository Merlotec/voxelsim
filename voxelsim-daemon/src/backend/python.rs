use std::ffi::CString;
use std::{convert::TryFrom, time::Duration};

use pyo3::{
    prelude::*,
    types::{PyAny, PyDict, PyList, PyModule},
};
use voxelsim::{Action, ActionIntent, Agent, MoveDir, VoxelGrid};

use crate::backend::{ControlBackend, ControlStep};

pub struct PythonBackend {
    module: Py<PyModule>,
    update_action_fn: Py<PyAny>,
}

impl PythonBackend {
    pub fn from_script(script: &str) -> PyResult<Self> {
        // Ensure the Python interpreter is initialized for use on any thread
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            // Register voxelsim_core module so scripts can `import voxelsim_core` and return Action
            let core = PyModule::new(py, "voxelsim_core")?;
            voxelsim::py::voxelsim_core(&core)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("voxelsim_core", core)?;

            // Compile the provided script as a module
            let code = CString::new(script).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("script contains NUL")
            })?;
            let file = CString::new("<embedded>").unwrap();
            let name = CString::new("voxelsim_backend").unwrap();
            let module = PyModule::from_code(py, &code, &file, &name)?;
            let func = module.getattr("update_action")?;
            Ok(Self {
                module: module.unbind(),
                update_action_fn: func.unbind(),
            })
        })
    }
}

impl ControlBackend for PythonBackend {
    fn update_action(&mut self, agent: &Agent, fw: &VoxelGrid) -> ControlStep {
        let update = Python::with_gil(|py| {
            let func = self.update_action_fn.bind(py);
            // TODO: the fw clone may be very inefficient.
            let ret = func.call((agent.clone(), fw.clone()), None).ok()?;

            if let Ok(a) = ret.extract::<Option<voxelsim::py::PyAgentStateUpdate>>() {
                return a.clone();
            }

            None
        });

        ControlStep {
            update: update.map(|x| x.into()),
            min_sleep: Duration::from_millis(100),
        }
    }
}
