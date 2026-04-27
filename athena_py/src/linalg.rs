use pyo3::prelude::*;

use crate::tensor::PyTensor;

#[pyfunction]
pub fn matmul(a: PyRef<'_, PyTensor>, b: PyRef<'_, PyTensor>) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: athena::linalg::matmul(a.inner.clone(), b.inner.clone()),
    })
}

#[pyfunction]
pub fn cross(a: PyRef<'_, PyTensor>, b: PyRef<'_, PyTensor>) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: athena::linalg::cross(a.inner.clone(), b.inner.clone()),
    })
}
