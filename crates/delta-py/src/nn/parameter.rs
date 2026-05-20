use delta::nn::Parameter;
use pyo3::prelude::*;

use crate::tensor::PyTensor;

#[pyclass(
    name = "Parameter",
    module = "delta.nn",
    unsendable,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyParameter {
    pub(crate) inner: Parameter,
}

#[pymethods]
impl PyParameter {
    #[new]
    fn new(tensor: PyRef<'_, PyTensor>) -> Self {
        Self {
            inner: Parameter(tensor.inner.clone()),
        }
    }

    // Delegate tensor methods so it behaves like a Tensor in Python
    fn grad(&self) -> Option<PyTensor> {
        if let Some(t) = self.inner.grad() {
            Some(PyTensor { inner: t })
        } else {
            None
        }
    }

    fn item(&self) -> Vec<f64> {
        self.inner.0.data()
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.0.shape().clone()
    }

    #[getter]
    fn data(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.0.clone(),
        }
    }
}
