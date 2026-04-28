use crate::tensor::PyTensor;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple};

#[pyclass(name = "Module", module = "athena.nn", subclass)]
pub struct PyModule;

#[pymethods]
impl PyModule {
    #[new]
    fn new() -> Self {
        Self
    }

    // "abstract" — subclasses must override this
    fn forward(&self, py: Python<'_>, _x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(py.None())
    }

    // concrete — all modules inherit this
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        slf.call_method("forward", args, kwargs).map(|b| b.unbind())
    }

    // concrete — subclasses override
    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }

    fn __repr__(&self) -> String {
        "Module()".to_string()
    }
}
