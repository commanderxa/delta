use pyo3::prelude::*;

use athena::{
    ivalue,
    nn::{Linear, Module},
};

use crate::tensor::PyTensor;

#[pyclass(name = "Linear", module = "athena.nn", unsendable, skip_from_py_object)]
#[derive(Clone)]
pub struct PyLinear {
    pub(crate) inner: Linear,
}

#[pymethods]
impl PyLinear {
    #[new]
    #[pyo3(signature = (in_features, out_features, bias=true))]
    fn new(in_features: usize, out_features: usize, bias: bool) -> PyResult<Self> {
        Ok(Self {
            inner: Linear::new(in_features, out_features, bias),
        })
    }

    #[getter]
    fn weights(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.weights.clone(),
        }
    }

    fn forward(&self, x: PyRef<'_, PyTensor>) -> PyTensor {
        let (args, kwargs) = ivalue![[x.inner.clone()]];
        PyTensor {
            inner: self.inner.forward(args, kwargs).unwrap_tensor(),
        }
    }

    fn __call__(&self, x: PyRef<'_, PyTensor>) -> PyTensor {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|t| PyTensor { inner: t })
            .collect()
    }
}
