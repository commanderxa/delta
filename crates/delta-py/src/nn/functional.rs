use pyo3::prelude::*;

use delta::nn::functional as F;

use crate::tensor::PyTensor;

pub fn register_submodule(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let functional = PyModule::new(py, "delta.nn.functional")?;
    functional.add_function(wrap_pyfunction!(relu, &functional)?)?;
    functional.add_function(wrap_pyfunction!(sigmoid, &functional)?)?;
    functional.add_function(wrap_pyfunction!(softmax, &functional)?)?;
    parent.add_submodule(&functional)?;
    Ok(())
}

#[pyfunction]
pub fn relu(x: PyRef<'_, PyTensor>) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: F::relu(x.inner.clone()),
    })
}

#[pyfunction]
pub fn sigmoid(x: PyRef<'_, PyTensor>) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: F::sigmoid(x.inner.clone()),
    })
}

#[pyfunction]
#[pyo3(signature = (x, dim=-1))]
pub fn softmax(x: PyRef<'_, PyTensor>, dim: isize) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: F::softmax(x.inner.clone(), dim),
    })
}
