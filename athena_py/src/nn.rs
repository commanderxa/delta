pub mod linear;

use pyo3::prelude::*;

use crate::nn::linear::PyLinear;

pub fn register_nn_submodule(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let nn = PyModule::new(py, "nn")?;
    nn.add_class::<PyLinear>()?;
    parent.add_submodule(&nn)?;
    Ok(())
}
