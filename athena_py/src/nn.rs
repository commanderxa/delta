pub mod linear;
pub mod module;

use pyo3::prelude::*;

use crate::nn::linear::PyLinear;

pub fn register_submodule(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let nn = PyModule::new(py, "_athena.nn")?;
    nn.add_class::<PyLinear>()?;
    parent.add_submodule(&nn)?;
    Ok(())
}
