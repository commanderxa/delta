use delta::DType;
use pyo3::prelude::*;

pub fn register_submodule(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let device = PyModule::new(py, "_delta.tensor.device")?;
    device.add_class::<PyDType>()?;
    device.add("float8", PyDType::Float8)?;
    device.add("float16", PyDType::Float16)?;
    device.add("bfloat16", PyDType::BFloat16)?;
    device.add("float32", PyDType::Float32)?;
    device.add("float64", PyDType::Float64)?;
    device.add("int8", PyDType::Int8)?;
    device.add("int16", PyDType::Int16)?;
    device.add("int32", PyDType::Int32)?;
    device.add("int64", PyDType::Int64)?;
    device.add("bool", PyDType::Bool)?;
    parent.add_submodule(&device)?;
    Ok(())
}

#[pyclass(name = "DType", module = "delta", from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyDType {
    Float8,
    Float16,
    BFloat16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Bool,
}

#[pymethods]
impl PyDType {
    fn __repr__(&self) -> &'static str {
        match self {
            PyDType::Float8 => "delta.float8",
            PyDType::Float16 => "delta.float16",
            PyDType::BFloat16 => "delta.bfloat16",
            PyDType::Float32 => "delta.float32",
            PyDType::Float64 => "delta.float64",
            PyDType::Int8 => "delta.int8",
            PyDType::Int16 => "delta.int16",
            PyDType::Int32 => "delta.int32",
            PyDType::Int64 => "delta.int64",
            PyDType::Bool => "delta.bool",
        }
    }

    fn __str__(&self) -> &'static str {
        self.__repr__()
    }
}

impl From<DType> for PyDType {
    fn from(value: DType) -> Self {
        match value {
            DType::Float8 => PyDType::Float8,
            DType::Float16 => PyDType::Float16,
            DType::BFloat16 => PyDType::BFloat16,
            DType::Float32 => PyDType::Float32,
            DType::Float64 => PyDType::Float64,
            DType::Int8 => PyDType::Int8,
            DType::Int16 => PyDType::Int16,
            DType::Int32 => PyDType::Int32,
            DType::Int64 => PyDType::Int64,
            DType::Bool => PyDType::Bool,
        }
    }
}

impl From<PyDType> for DType {
    fn from(value: PyDType) -> Self {
        match value {
            PyDType::Float8 => DType::Float8,
            PyDType::Float16 => DType::Float16,
            PyDType::BFloat16 => DType::BFloat16,
            PyDType::Float32 => DType::Float32,
            PyDType::Float64 => DType::Float64,
            PyDType::Int8 => DType::Int8,
            PyDType::Int16 => DType::Int16,
            PyDType::Int32 => DType::Int32,
            PyDType::Int64 => DType::Int64,
            PyDType::Bool => DType::Bool,
        }
    }
}
