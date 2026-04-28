use pyo3::prelude::*;

use athena::optim::Optim;
use athena::optim::sgd::SGD;

use crate::tensor::PyTensor;

#[pyclass(name = "SGD", module = "athena.optim", unsendable)]
pub struct PySGD {
    pub(crate) inner: SGD,
}

#[pymethods]
impl PySGD {
    #[new]
    #[pyo3(signature = (parameters, lr))]
    fn new(parameters: Vec<PyRef<'_, PyTensor>>, lr: f64) -> Self {
        let params = parameters.into_iter().map(|t| t.inner.clone()).collect();

        Self {
            inner: SGD::new(params, lr),
        }
    }
    #[getter]
    fn lr(&self) -> f64 {
        self.inner.lr()
    }

    #[getter]
    fn is_maximize(&self) -> bool {
        self.inner.is_maximize()
    }

    fn step(&self) {
        self.inner.step();
    }

    fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    fn change_lr(&mut self, gamma: f64) {
        self.inner.change_lr(gamma);
    }

    fn maximize(&mut self) {
        self.inner.maximize();
    }

    fn minimize(&mut self) {
        self.inner.minimize();
    }
}
