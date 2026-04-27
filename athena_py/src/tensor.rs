use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use athena::Tensor;

#[pyclass(name = "Tensor", module = "athena", unsendable, skip_from_py_object)]
#[derive(Clone)]
pub struct PyTensor {
    pub(crate) inner: Tensor,
}

#[pymethods]
impl PyTensor {
    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> Self {
        Self {
            inner: Tensor::zeros(&shape),
        }
    }

    #[allow(non_snake_case)]
    #[getter]
    fn T(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.t(),
        })
    }

    #[staticmethod]
    fn zeros_like(a: PyRef<'_, PyTensor>) -> Self {
        let shape = a.shape();
        Self {
            inner: Tensor::zeros(&shape),
        }
    }

    #[staticmethod]
    fn ones(shape: Vec<usize>) -> Self {
        Self {
            inner: Tensor::ones(&shape),
        }
    }

    #[staticmethod]
    fn ones_like(a: PyRef<'_, PyTensor>) -> Self {
        let shape = a.shape();
        Self {
            inner: Tensor::ones(&shape),
        }
    }

    #[staticmethod]
    fn randn(shape: Vec<usize>) -> Self {
        Self {
            inner: Tensor::randn(&shape),
        }
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.shape.len()
    }

    #[getter]
    fn length(&self) -> usize {
        self.inner.length()
    }

    fn storage(&self) -> Vec<f64> {
        self.inner.storage()
    }

    fn item(&self) -> Vec<f64> {
        self.inner.item()
    }

    #[getter]
    fn grad(&self) -> PyResult<Option<Vec<f64>>> {
        Ok(self.inner.grad())
    }

    #[pyo3(signature = (*shape))]
    fn reshape(&self, shape: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let obj = if shape.len() == 1 {
            shape.get_item(0)?
        } else {
            shape.as_any().clone()
        };

        let shape = parse_shape(&obj, self.inner.length())?;

        Ok(Self {
            inner: self.inner.reshape(&shape),
        })
    }

    #[pyo3(signature = (dim=None))]
    fn squeeze(&self, dim: Option<Vec<usize>>) -> PyResult<Self> {
        let dim = dim.unwrap_or_default();
        Ok(Self {
            inner: self.inner.squeeze(&dim),
        })
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    fn backward(&self) -> () {
        self.inner.backward();
    }

    #[staticmethod]
    #[pyo3(signature = (tensors, dim=0))]
    fn cat(tensors: Vec<PyRef<'_, PyTensor>>, dim: Option<usize>) -> PyResult<Self> {
        let rust_tensors: Vec<Tensor> = tensors.iter().map(|t| t.inner.clone()).collect();
        let dim = dim.unwrap_or_default();
        Ok(Self {
            inner: Tensor::cat(&rust_tensors, dim),
        })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    fn sum(&self, dim: Option<usize>, keepdim: bool) -> Self {
        Self {
            inner: self.inner.sum(dim, keepdim),
        }
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    fn mean(&self, dim: Option<usize>, keepdim: bool) -> Self {
        Self {
            inner: self.inner.mean(dim, keepdim),
        }
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(value) = other.extract::<f64>() {
            Ok(Self {
                inner: self.inner.clone() + value,
            })
        } else if let Ok(other_tensor) = other.extract::<PyRef<'_, PyTensor>>() {
            Ok(Self {
                inner: self.inner.clone() + other_tensor.inner.clone(),
            })
        } else {
            Err(PyTypeError::new_err(
                "unsupported operand type(s) for +: 'athena.Tensor' and given type",
            ))
        }
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__add__(other)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(value) = other.extract::<f64>() {
            Ok(Self {
                inner: self.inner.clone() + value,
            })
        } else if let Ok(other_tensor) = other.extract::<PyRef<'_, PyTensor>>() {
            Ok(Self {
                inner: self.inner.clone() - other_tensor.inner.clone(),
            })
        } else {
            Err(PyTypeError::new_err(
                "unsupported operand type(s) for +: 'athena.Tensor' and given type",
            ))
        }
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__sub__(other)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(value) = other.extract::<f64>() {
            Ok(Self {
                inner: self.inner.clone() + value,
            })
        } else if let Ok(other_tensor) = other.extract::<PyRef<'_, PyTensor>>() {
            Ok(Self {
                inner: self.inner.clone() * other_tensor.inner.clone(),
            })
        } else {
            Err(PyTypeError::new_err(
                "unsupported operand type(s) for +: 'athena.Tensor' and given type",
            ))
        }
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(other)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(value) = other.extract::<f64>() {
            Ok(Self {
                inner: self.inner.clone() + value,
            })
        } else if let Ok(other_tensor) = other.extract::<PyRef<'_, PyTensor>>() {
            Ok(Self {
                inner: self.inner.clone() / other_tensor.inner.clone(),
            })
        } else {
            Err(PyTypeError::new_err(
                "unsupported operand type(s) for +: 'athena.Tensor' and given type",
            ))
        }
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__truediv__(other)
    }

    fn __neg__(&self) -> Self {
        Self {
            inner: -self.inner.clone(),
        }
    }
}

#[pyfunction]
pub fn tensor(obj: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    let (data, shape) = extract_nested(obj)?;
    Ok(PyTensor {
        inner: Tensor::tensor(&data, &shape),
    })
}

fn extract_nested(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, Vec<usize>)> {
    if let Ok(value) = obj.extract::<f64>() {
        return Ok((vec![value], vec![]));
    }

    if let Ok(list) = obj.cast::<PyList>() {
        return extract_sequence(list.iter().collect());
    }

    if let Ok(tuple) = obj.cast::<PyTuple>() {
        return extract_sequence(tuple.iter().collect());
    }

    Err(PyTypeError::new_err(
        "tensor() expects a number or a nested list/tuple of numbers",
    ))
}

fn extract_sequence(items: Vec<Bound<'_, PyAny>>) -> PyResult<(Vec<f64>, Vec<usize>)> {
    let len = items.len();

    if len == 0 {
        return Ok((Vec::new(), vec![0]));
    }

    let mut flat = Vec::new();
    let mut inner_shape: Option<Vec<usize>> = None;

    for item in items {
        let (child_flat, child_shape) = extract_nested(&item)?;

        match &inner_shape {
            None => inner_shape = Some(child_shape.clone()),
            Some(expected) if *expected == child_shape => {}
            Some(expected) => {
                return Err(PyValueError::new_err(format!(
                    "ragged nested sequence: expected inner shape {:?}, got {:?}",
                    expected, child_shape
                )));
            }
        }

        flat.extend(child_flat);
    }

    let mut shape = vec![len];
    if let Some(child_shape) = inner_shape {
        shape.extend(child_shape);
    }

    Ok((flat, shape))
}

fn parse_shape(obj: &Bound<'_, PyAny>, total_len: usize) -> PyResult<Vec<usize>> {
    let dims: Vec<isize> = if let Ok(v) = obj.extract::<isize>() {
        vec![v]
    } else if let Ok(v) = obj.extract::<Vec<isize>>() {
        v
    } else if let Ok(tuple) = obj.cast::<PyTuple>() {
        tuple.extract::<Vec<isize>>()?
    } else if let Ok(list) = obj.cast::<PyList>() {
        list.extract::<Vec<isize>>()?
    } else {
        return Err(PyValueError::new_err(
            "reshape expects an int, tuple, or list of ints",
        ));
    };

    if dims.is_empty() {
        return Err(PyValueError::new_err("reshape shape cannot be empty"));
    }

    let mut out = Vec::with_capacity(dims.len());
    let mut infer_idx: Option<usize> = None;
    let mut known_product = 1usize;

    for (i, &d) in dims.iter().enumerate() {
        if d == -1 {
            if infer_idx.is_some() {
                return Err(PyValueError::new_err("only one dimension can be inferred"));
            }
            infer_idx = Some(i);
            out.push(0);
        } else if d < 0 {
            return Err(PyValueError::new_err(
                "reshape dimensions must be >= 0, except -1",
            ));
        } else {
            let u = d as usize;
            known_product = known_product
                .checked_mul(u)
                .ok_or_else(|| PyValueError::new_err("shape product overflow"))?;
            out.push(u);
        }
    }

    if let Some(i) = infer_idx {
        if known_product == 0 || !total_len.is_multiple_of(known_product) {
            return Err(PyValueError::new_err(format!(
                "cannot infer shape {:?} for tensor of length {}",
                dims, total_len
            )));
        }
        out[i] = total_len / known_product;
    } else if known_product != total_len {
        return Err(PyValueError::new_err(format!(
            "cannot reshape tensor of length {} into shape {:?}",
            total_len, dims
        )));
    }

    Ok(out)
}
