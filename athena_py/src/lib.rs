mod linalg;
mod nn;
mod operations;
mod optim;
mod tensor;

use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "_athena")]
fn _athena(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    tensor::register_submodule(_py, m)?;
    linalg::register_submodule(_py, m)?;
    operations::register_submodule(_py, m)?;
    nn::register_submodule(_py, m)?;
    Ok(())
}
