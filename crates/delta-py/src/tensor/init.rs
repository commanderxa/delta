use delta::{DType, Device, get_default_dtype};
use pyo3::prelude::*;

use crate::tensor::{PyTensor, device::PyDevice, dtype::PyDType};

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
#[pyo3(signature = (obj, device=PyDevice::from(delta::cpu)))]
pub fn tensor(obj: &Bound<'_, PyAny>, device: PyDevice) -> PyResult<PyTensor> {
    let (data, shape) = crate::tensor::extract_nested(obj)?;
    Ok(PyTensor {
        inner: delta::tensor(&data, &shape, Device::from(device)),
    })
}

#[pyfunction]
#[pyo3(signature = (shape, device=PyDevice::from(delta::cpu)))]
pub fn randn(shape: Vec<usize>, device: PyDevice) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: delta::randn(&shape, Device::from(device)),
    })
}

#[pyfunction]
#[pyo3(signature = (shape, dtype=PyDType::from(get_default_dtype()), device=PyDevice::from(delta::cpu)))]
pub fn zeros(shape: Vec<usize>, dtype: PyDType, device: PyDevice) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: delta::zeros(&shape, DType::from(dtype), Device::from(device)),
    })
}

#[pyfunction]
#[pyo3(signature = (tensor, dtype=PyDType::from(get_default_dtype()), device=PyDevice::from(delta::cpu)))]
pub fn zeros_like(
    tensor: PyRef<'_, PyTensor>,
    dtype: PyDType,
    device: PyDevice,
) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: delta::zeros_like(&tensor.inner, DType::from(dtype), Device::from(device)),
    })
}

#[pyfunction]
#[pyo3(signature = (shape, dtype=PyDType::from(get_default_dtype()), device=PyDevice::from(delta::cpu)))]
pub fn ones(shape: Vec<usize>, dtype: PyDType, device: PyDevice) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: delta::ones(&shape, DType::from(dtype), Device::from(device)),
    })
}

#[pyfunction]
#[pyo3(signature = (tensor, dtype=PyDType::from(get_default_dtype()), device=PyDevice::from(delta::cpu)))]
pub fn ones_like(
    tensor: PyRef<'_, PyTensor>,
    dtype: PyDType,
    device: PyDevice,
) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: delta::ones_like(&tensor.inner, DType::from(dtype), Device::from(device)),
    })
}

#[pyfunction]
#[pyo3(signature = (n, dtype=PyDType::from(get_default_dtype()), device=PyDevice::from(delta::cpu)))]
pub fn eye(n: usize, dtype: PyDType, device: PyDevice) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: delta::eye(n, DType::from(dtype), Device::from(device)),
    })
}

#[pyfunction]
#[pyo3(signature = (start, end, step, dtype=PyDType::from(get_default_dtype()), device=PyDevice::from(delta::cpu)))]
pub fn arange(
    start: f32,
    end: f32,
    step: f32,
    dtype: PyDType,
    device: PyDevice,
) -> PyResult<PyTensor> {
    Ok(PyTensor {
        inner: delta::arange(start, end, step, DType::from(dtype), Device::from(device)),
    })
}
