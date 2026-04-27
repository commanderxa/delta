use athena::Tensor;
use pyo3::prelude::*;

use crate::tensor::PyTensor;

#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn sum(input: PyRef<'_, PyTensor>, dim: Option<usize>, keepdim: bool) -> PyTensor {
    PyTensor {
        inner: input.inner.sum(dim, keepdim),
    }
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn mean(input: PyRef<'_, PyTensor>, dim: Option<usize>, keepdim: bool) -> PyTensor {
    PyTensor {
        inner: input.inner.mean(dim, keepdim),
    }
}

#[pyfunction]
#[pyo3(signature = (tensors, dim=0))]
pub fn cat(tensors: Vec<PyRef<'_, PyTensor>>, dim: usize) -> PyTensor {
    let rust_tensors: Vec<Tensor> = tensors.iter().map(|t| t.inner.clone()).collect();
    PyTensor {
        inner: Tensor::cat(&rust_tensors, dim),
    }
}
