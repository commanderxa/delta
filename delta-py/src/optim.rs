pub mod sgd;

use pyo3::prelude::*;

use crate::optim::sgd::PySGD;

pub fn register_submodule(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let optim = PyModule::new(py, "_delta.optim")?;
    optim.add_class::<PySGD>()?;
    parent.add_submodule(&optim)?;
    Ok(())
}
