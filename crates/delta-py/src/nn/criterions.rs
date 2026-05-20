use pyo3::prelude::*;

use crate::tensor::PyTensor;
use delta::nn::{MSELoss, criterions::Reduction};

pub fn register_submodule(_py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PyMSELoss>()?;
    parent.add_class::<PyReduction>()?;
    Ok(())
}

#[pyclass(module = "delta.nn", name = "Reduction", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyReduction {
    SUM = 0,
    MEAN = 1,
}

impl From<PyReduction> for Reduction {
    fn from(value: PyReduction) -> Self {
        match value {
            PyReduction::SUM => Reduction::SUM,
            PyReduction::MEAN => Reduction::MEAN,
        }
    }
}

#[pyclass(
    name = "MSELoss",
    module = "delta.nn",
    unsendable,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyMSELoss {
    pub(crate) inner: MSELoss,
}

#[pymethods]
impl PyMSELoss {
    #[new]
    #[pyo3(signature = (reduction=PyReduction::MEAN))]
    fn new(reduction: Option<PyReduction>) -> Self {
        Self {
            inner: MSELoss::new(reduction.map(Into::into)),
        }
    }

    #[pyo3(name = "measure")]
    fn measure(&self, a: PyRef<'_, PyTensor>, b: PyRef<'_, PyTensor>) -> PyTensor {
        PyTensor {
            inner: self.inner.measure(a.inner.clone(), b.inner.clone()),
        }
    }

    fn __call__(&self, a: PyRef<'_, PyTensor>, b: PyRef<'_, PyTensor>) -> PyTensor {
        PyTensor {
            inner: self.inner.measure(a.inner.clone(), b.inner.clone()),
        }
    }

    fn __repr__(&self) -> String {
        "MSELoss()".to_string()
    }
}
