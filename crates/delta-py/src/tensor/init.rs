use pyo3::prelude::*;

use crate::tensor::PyTensor;

pub fn register_submodule(_: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_function(wrap_pyfunction!(tensor, parent)?)?;
    parent.add_function(wrap_pyfunction!(randn, parent)?)?;
    parent.add_function(wrap_pyfunction!(zeros, parent)?)?;
    parent.add_function(wrap_pyfunction!(zeros_like, parent)?)?;
    parent.add_function(wrap_pyfunction!(ones, parent)?)?;
    parent.add_function(wrap_pyfunction!(ones_like, parent)?)?;
    parent.add_function(wrap_pyfunction!(eye, parent)?)?;
    parent.add_function(wrap_pyfunction!(arange, parent)?)?;
    Ok(())
}

#[pyfunction]
pub fn tensor(obj: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    let (data, shape) = crate::tensor::extract_nested(obj)?;
    Ok(PyTensor {
        inner: delta::tensor(&data, &shape),
    })
}

#[pyfunction]
pub fn randn(shape: Vec<usize>) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: delta::randn(&shape),
    })
}

#[pyfunction]
pub fn zeros(shape: Vec<usize>) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: delta::zeros(&shape),
    })
}

#[pyfunction]
pub fn zeros_like(tensor: PyRef<'_, PyTensor>) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: delta::zeros_like(&tensor.inner),
    })
}

#[pyfunction]
pub fn ones(shape: Vec<usize>) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: delta::ones(&shape),
    })
}

#[pyfunction]
pub fn ones_like(tensor: PyRef<'_, PyTensor>) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: delta::ones_like(&tensor.inner),
    })
}

#[pyfunction]
pub fn eye(n: usize) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: delta::eye(n),
    })
}

#[pyfunction]
pub fn arange(start: f64, end: f64, step: f64) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: delta::arange(start, end, step),
    })
}
