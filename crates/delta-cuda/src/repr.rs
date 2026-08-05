use cudarc::driver::CudaSlice;
use half::{bf16, f16};

use delta_core::{dtype::DType, f8};
use delta_cpu::{repr::CPUStorageRepr, storage::CPUStorage};

use crate::storage::CUDAStorage;
use crate::{array_to_cuda_slice, cuda_slice_to_array};

pub trait CUDAStorageRepr: Sized + Copy + Clone + CPUStorageRepr {
    const DTYPE: DType;

    // Storage construction
    fn into_cuda_storage(data: &[Self]) -> CUDAStorage;
    // Storage access
    fn storage_as_slice(s: &CUDAStorage) -> Option<&CudaSlice<Self>>;
    fn storage_as_slice_mut(s: &mut CUDAStorage) -> Option<&mut CudaSlice<Self>>;
    // function for other storage types
    fn from_cpu_storage(storage: &CPUStorage) -> CUDAStorage;
    fn into_cpu_storage(storage: &CUDAStorage) -> CPUStorage;
}

macro_rules! impl_storage_repr {
    ($($t:ty => $cpu_variant:ident, $cuda_variant:ident, $dtype:expr),+ $(,)?) => {
        $(
            impl CUDAStorageRepr for $t {
                const DTYPE: DType = $dtype;

                fn into_cuda_storage(data: &[Self]) -> CUDAStorage {
                    let data = array_to_cuda_slice(data);
                    CUDAStorage::$cuda_variant(data)
                }

                fn storage_as_slice(s: &CUDAStorage) -> Option<&CudaSlice<Self>> {
                    if let CUDAStorage::$cuda_variant(v) = s { Some(v) } else { None }
                }

                fn storage_as_slice_mut(s: &mut CUDAStorage) -> Option<&mut CudaSlice<Self>> {
                    if let CUDAStorage::$cuda_variant(v) = s { Some(v) } else { None }
                }

                fn from_cpu_storage(s: &CPUStorage) -> CUDAStorage {
                    Self::into_cuda_storage(<Self as CPUStorageRepr>::storage_as_slice(s).unwrap())
                }

                fn into_cpu_storage(s: &CUDAStorage) -> CPUStorage {
                    Self::into_storage(&cuda_slice_to_array(<Self as CUDAStorageRepr>::storage_as_slice(s).unwrap()))
                }
            }
        )+
    };
}

impl_storage_repr! {
    i8   => I8,   I8,   DType::Int8,
    i16  => I16,  I16,  DType::Int16,
    i32  => I32,  I32,  DType::Int32,
    i64  => I64,  I64,  DType::Int64,
    f16  => F16,  F16,  DType::Float16,
    bf16 => BF16, BF16, DType::BFloat16,
    f8   => F8,   F8,   DType::Float8,
    f32  => F32,  F32,  DType::Float32,
    f64  => F64,  F64,  DType::Float64,
    bool => Bool, Bool, DType::Bool,
}
