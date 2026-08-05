use delta_core::{cast::Cast, dtype::DType, f8, repr::TensorRepr};
use half::{bf16, f16};

use crate::repr::CPUStorageRepr;

#[derive(Clone, Debug, PartialEq)]
pub enum CPUStorage {
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F8(Vec<f8>),
    F16(Vec<f16>),
    BF16(Vec<bf16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Bool(Vec<bool>),
}

impl CPUStorage {

    pub fn new<T: TensorRepr>(data: &[T]) -> Self {
        match T::DTYPE {
            DType::Float8 => CPUStorage::F8(data.iter().map(|x| (*x).cast()).collect()),
            DType::Float16 => CPUStorage::F16(data.iter().map(|x| (*x).cast()).collect()),
            DType::BFloat16 => CPUStorage::BF16(data.iter().map(|x| (*x).cast()).collect()),
            DType::Float32 => CPUStorage::F32(data.iter().map(|x| (*x).cast()).collect()),
            DType::Float64 => CPUStorage::F64(data.iter().map(|x| (*x).cast()).collect()),
            DType::Int8 => CPUStorage::I8(data.iter().map(|x| (*x).cast()).collect()),
            DType::Int16 => CPUStorage::I16(data.iter().map(|x| (*x).cast()).collect()),
            DType::Int32 => CPUStorage::I32(data.iter().map(|x| (*x).cast()).collect()),
            DType::Int64 => CPUStorage::I64(data.iter().map(|x| (*x).cast()).collect()),
            DType::Bool => CPUStorage::Bool(data.iter().map(|x| (*x).cast()).collect()),
        }
    }

    pub fn fill<T: TensorRepr>(&mut self, value: T) {
        match self {
            CPUStorage::F8(v) => v.fill(Cast::<f8>::cast(value)),
            CPUStorage::F16(v) => v.fill(Cast::<f16>::cast(value)),
            CPUStorage::BF16(v) => v.fill(Cast::<bf16>::cast(value)),
            CPUStorage::F32(v) => v.fill(Cast::<f32>::cast(value)),
            CPUStorage::F64(v) => v.fill(Cast::<f64>::cast(value)),
            CPUStorage::I8(v) => v.fill(Cast::<i8>::cast(value)),
            CPUStorage::I16(v) => v.fill(Cast::<i16>::cast(value)),
            CPUStorage::I32(v) => v.fill(Cast::<i32>::cast(value)),
            CPUStorage::I64(v) => v.fill(Cast::<i64>::cast(value)),
            CPUStorage::Bool(v) => v.fill(Cast::<bool>::cast(value)),
        }
    }

    pub fn cast_to<U: CPUStorageRepr>(&self) -> CPUStorage {
        match self {
            CPUStorage::F8(v) => {
                U::into_storage(&v.iter().map(|&x| U::cast_from(x)).collect::<Vec<_>>())
            }
            CPUStorage::F16(v) => {
                U::into_storage(&v.iter().map(|&x| U::cast_from(x)).collect::<Vec<_>>())
            }
            CPUStorage::BF16(v) => {
                U::into_storage(&v.iter().map(|&x| U::cast_from(x)).collect::<Vec<_>>())
            }
            CPUStorage::F32(v) => {
                U::into_storage(&v.iter().map(|&x| U::cast_from(x)).collect::<Vec<_>>())
            }
            CPUStorage::F64(v) => {
                U::into_storage(&v.iter().map(|&x| U::cast_from(x)).collect::<Vec<_>>())
            }
            CPUStorage::I8(v) => {
                U::into_storage(&v.iter().map(|&x| U::cast_from(x)).collect::<Vec<_>>())
            }
            CPUStorage::I16(v) => {
                U::into_storage(&v.iter().map(|&x| U::cast_from(x)).collect::<Vec<_>>())
            }
            CPUStorage::I32(v) => {
                U::into_storage(&v.iter().map(|&x| U::cast_from(x)).collect::<Vec<_>>())
            }
            CPUStorage::I64(v) => {
                U::into_storage(&v.iter().map(|&x| U::cast_from(x)).collect::<Vec<_>>())
            }
            CPUStorage::Bool(v) => {
                U::into_storage(&v.iter().map(|&x| U::cast_from(x)).collect::<Vec<_>>())
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            CPUStorage::I8(v) => v.len(),
            CPUStorage::I16(v) => v.len(),
            CPUStorage::I32(v) => v.len(),
            CPUStorage::I64(v) => v.len(),
            CPUStorage::F8(v) => v.len(),
            CPUStorage::F16(v) => v.len(),
            CPUStorage::BF16(v) => v.len(),
            CPUStorage::F32(v) => v.len(),
            CPUStorage::F64(v) => v.len(),
            CPUStorage::Bool(v) => v.len(),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            CPUStorage::I8(_) => DType::Int8,
            CPUStorage::I16(_) => DType::Int16,
            CPUStorage::I32(_) => DType::Int32,
            CPUStorage::I64(_) => DType::Int64,
            CPUStorage::F8(_) => DType::Float8,
            CPUStorage::F16(_) => DType::Float16,
            CPUStorage::BF16(_) => DType::BFloat16,
            CPUStorage::F32(_) => DType::Float32,
            CPUStorage::F64(_) => DType::Float64,
            CPUStorage::Bool(_) => DType::Bool,
        }
    }
}
