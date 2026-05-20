pub mod criterions;
pub mod functional;
pub mod parameter;

use pyo3::prelude::*;

use crate::nn::parameter::PyParameter;

pub fn register_submodule(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let nn = PyModule::new(py, "_delta.nn")?;
    nn.add_class::<PyParameter>()?;
    functional::register_submodule(py, &nn)?;
    criterions::register_submodule(py, &nn)?;
    parent.add_submodule(&nn)?;
    Ok(())
}
