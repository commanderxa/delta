use cudarc::driver::CudaSlice;
use delta_core::{dtype::DType, f8};
use delta_cpu::storage::CPUStorage;
use half::{bf16, f16};

use crate::{cuda_slice_to_array, repr::CUDAStorageRepr};

#[derive(Clone, Debug)]
pub enum CUDAStorage {
    I8(CudaSlice<i8>),
    I16(CudaSlice<i16>),
    I32(CudaSlice<i32>),
    I64(CudaSlice<i64>),
    F8(CudaSlice<f8>),
    F16(CudaSlice<f16>),
    BF16(CudaSlice<bf16>),
    F32(CudaSlice<f32>),
    F64(CudaSlice<f64>),
    Bool(CudaSlice<bool>),
}

impl CUDAStorage {
    pub fn fill(&mut self, _: f32) {
        todo!()
    }

    pub fn to_cpu(&self) -> CPUStorage {
        match self {
            CUDAStorage::I8(v) => CPUStorage::I8(cuda_slice_to_array(v)),
            CUDAStorage::I16(v) => CPUStorage::I16(cuda_slice_to_array(v)),
            CUDAStorage::I32(v) => CPUStorage::I32(cuda_slice_to_array(v)),
            CUDAStorage::I64(v) => CPUStorage::I64(cuda_slice_to_array(v)),
            CUDAStorage::F8(v) => CPUStorage::F8(cuda_slice_to_array(v)),
            CUDAStorage::F16(v) => CPUStorage::F16(cuda_slice_to_array(v)),
            CUDAStorage::BF16(v) => CPUStorage::BF16(cuda_slice_to_array(v)),
            CUDAStorage::F32(v) => CPUStorage::F32(cuda_slice_to_array(v)),
            CUDAStorage::F64(v) => CPUStorage::F64(cuda_slice_to_array(v)),
            CUDAStorage::Bool(v) => CPUStorage::Bool(cuda_slice_to_array(v)),
        }
    }

    pub fn cast_to<U: CUDAStorageRepr>(&self) -> CUDAStorage {
        match self {
            CUDAStorage::F8(v) => U::into_cuda_storage(
                &cuda_slice_to_array(v)
                    .iter()
                    .map(|&x| U::cast_from(x))
                    .collect::<Vec<_>>(),
            ),
            CUDAStorage::F16(v) => U::into_cuda_storage(
                &cuda_slice_to_array(v)
                    .iter()
                    .map(|&x| U::cast_from(x))
                    .collect::<Vec<_>>(),
            ),
            CUDAStorage::BF16(v) => U::into_cuda_storage(
                &cuda_slice_to_array(v)
                    .iter()
                    .map(|&x| U::cast_from(x))
                    .collect::<Vec<_>>(),
            ),
            CUDAStorage::F32(v) => U::into_cuda_storage(
                &cuda_slice_to_array(v)
                    .iter()
                    .map(|&x| U::cast_from(x))
                    .collect::<Vec<_>>(),
            ),
            CUDAStorage::F64(v) => U::into_cuda_storage(
                &cuda_slice_to_array(v)
                    .iter()
                    .map(|&x| U::cast_from(x))
                    .collect::<Vec<_>>(),
            ),
            CUDAStorage::I8(v) => U::into_cuda_storage(
                &cuda_slice_to_array(v)
                    .iter()
                    .map(|&x| U::cast_from(x))
                    .collect::<Vec<_>>(),
            ),
            CUDAStorage::I16(v) => U::into_cuda_storage(
                &cuda_slice_to_array(v)
                    .iter()
                    .map(|&x| U::cast_from(x))
                    .collect::<Vec<_>>(),
            ),
            CUDAStorage::I32(v) => U::into_cuda_storage(
                &cuda_slice_to_array(v)
                    .iter()
                    .map(|&x| U::cast_from(x))
                    .collect::<Vec<_>>(),
            ),
            CUDAStorage::I64(v) => U::into_cuda_storage(
                &cuda_slice_to_array(v)
                    .iter()
                    .map(|&x| U::cast_from(x))
                    .collect::<Vec<_>>(),
            ),
            CUDAStorage::Bool(v) => U::into_cuda_storage(
                &cuda_slice_to_array(v)
                    .iter()
                    .map(|&x| U::cast_from(x))
                    .collect::<Vec<_>>(),
            ),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            CUDAStorage::I8(v) => v.len(),
            CUDAStorage::I16(v) => v.len(),
            CUDAStorage::I32(v) => v.len(),
            CUDAStorage::I64(v) => v.len(),
            CUDAStorage::F8(v) => v.len(),
            CUDAStorage::F16(v) => v.len(),
            CUDAStorage::BF16(v) => v.len(),
            CUDAStorage::F32(v) => v.len(),
            CUDAStorage::F64(v) => v.len(),
            CUDAStorage::Bool(v) => v.len(),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            CUDAStorage::I8(_) => DType::Int8,
            CUDAStorage::I16(_) => DType::Int16,
            CUDAStorage::I32(_) => DType::Int32,
            CUDAStorage::I64(_) => DType::Int64,
            CUDAStorage::F8(_) => DType::Float8,
            CUDAStorage::F16(_) => DType::Float16,
            CUDAStorage::BF16(_) => DType::BFloat16,
            CUDAStorage::F32(_) => DType::Float32,
            CUDAStorage::F64(_) => DType::Float64,
            CUDAStorage::Bool(_) => DType::Bool,
        }
    }
}

impl PartialEq for CUDAStorage {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (a, b) => a == b,
        }
    }
}