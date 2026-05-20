use delta_core::cast::Cast;

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

    #[cfg(feature = "cuda")]
    pub fn to_cuda(&self) -> CUDAStorage {
        match self {
            CPUStorage::I8(v) => CUDAStorage::I8(array_to_cuda_slice(v)),
            CPUStorage::I16(v) => CUDAStorage::I16(array_to_cuda_slice(v)),
            CPUStorage::I32(v) => CUDAStorage::I32(array_to_cuda_slice(v)),
            CPUStorage::I64(v) => CUDAStorage::I64(array_to_cuda_slice(v)),
            CPUStorage::F8(v) => CUDAStorage::F8(array_to_cuda_slice(v)),
            CPUStorage::F16(v) => CUDAStorage::F16(array_to_cuda_slice(v)),
            CPUStorage::BF16(v) => CUDAStorage::BF16(array_to_cuda_slice(v)),
            CPUStorage::F32(v) => CUDAStorage::F32(array_to_cuda_slice(v)),
            CPUStorage::F64(v) => CUDAStorage::F64(array_to_cuda_slice(v)),
            CPUStorage::Bool(v) => CUDAStorage::Bool(array_to_cuda_slice(v)),
        }
    }

    pub fn cast_to<U: StorageRepr>(&self) -> CPUStorage {
        match self {
            CPUStorage::F8(v) => {
                U::into_cpu_storage(&v.iter().map(|&x| U::cast_from(x)).collect::<Vec<_>>())
            }
            CPUStorage::F16(v) => {
                U::into_cpu_storage(&v.iter().map(|&x| U::cast_from(x)).collect::<Vec<_>>())
            }
            CPUStorage::BF16(v) => {
                U::into_cpu_storage(&v.iter().map(|&x| U::cast_from(x)).collect::<Vec<_>>())
            }
            CPUStorage::F32(v) => {
                U::into_cpu_storage(&v.iter().map(|&x| U::cast_from(x)).collect::<Vec<_>>())
            }
            CPUStorage::F64(v) => {
                U::into_cpu_storage(&v.iter().map(|&x| U::cast_from(x)).collect::<Vec<_>>())
            }
            CPUStorage::I8(v) => {
                U::into_cpu_storage(&v.iter().map(|&x| U::cast_from(x)).collect::<Vec<_>>())
            }
            CPUStorage::I16(v) => {
                U::into_cpu_storage(&v.iter().map(|&x| U::cast_from(x)).collect::<Vec<_>>())
            }
            CPUStorage::I32(v) => {
                U::into_cpu_storage(&v.iter().map(|&x| U::cast_from(x)).collect::<Vec<_>>())
            }
            CPUStorage::I64(v) => {
                U::into_cpu_storage(&v.iter().map(|&x| U::cast_from(x)).collect::<Vec<_>>())
            }
            CPUStorage::Bool(v) => {
                U::into_cpu_storage(&v.iter().map(|&x| U::cast_from(x)).collect::<Vec<_>>())
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