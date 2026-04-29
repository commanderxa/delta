mod linalg;
mod nn;
mod operations;
mod optim;
mod tensor;
mod tensor_init;

use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "_delta")]
fn _delta(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    linalg::register_submodule(_py, m)?;
    nn::register_submodule(_py, m)?;
    operations::register_submodule(_py, m)?;
    optim::register_submodule(_py, m)?;
    tensor::register_submodule(_py, m)?;
    tensor_init::register_submodule(_py, m)?;
    Ok(())
}
