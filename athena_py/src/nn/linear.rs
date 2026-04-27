use athena::module::Forward;
use pyo3::prelude::*;

use athena::nn::Linear;

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

    fn forward(&self, x: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.forward(x.inner.clone()),
        }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        self.forward(x)
    }
}
