pub mod linalg;
pub mod nn;
pub mod operations;
pub mod tensor;

use pyo3::prelude::*;

use crate::{nn::register_nn_submodule, tensor::PyTensor};

#[pymodule]
#[pyo3(name = "_athena")]
fn _athena(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(crate::tensor::tensor, m)?)?;
    m.add_function(wrap_pyfunction!(crate::linalg::matmul, m)?)?;
    m.add_function(wrap_pyfunction!(crate::linalg::cross, m)?)?;
    m.add_function(wrap_pyfunction!(crate::operations::sum, m)?)?;
    m.add_function(wrap_pyfunction!(crate::operations::mean, m)?)?;
    m.add_function(wrap_pyfunction!(crate::operations::cat, m)?)?;
    register_nn_submodule(_py, m)?;
    Ok(())
}
