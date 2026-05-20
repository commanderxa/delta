#[cfg(feature = "cuda")]
use crate::cuda::{array_to_cuda_slice, cuda_slice_to_array};
use crate::{
    DType, f8,
    tensor::{
        cast::{Cast, CastFrom},
        repr::TensorRepr,
    },
};
#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;
use half::{bf16, f16};

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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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

    pub fn cast_to<U: StorageRepr>(&self) -> CUDAStorage {
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

#[cfg(feature = "cuda")]
impl PartialEq for CUDAStorage {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (a, b) => a == b,
        }
    }
}

pub trait StorageRepr:
    Sized
    + Copy
    + Clone
    + CastFrom<i8>
    + CastFrom<i16>
    + CastFrom<i32>
    + CastFrom<i64>
    + CastFrom<f8>
    + CastFrom<f16>
    + CastFrom<bf16>
    + CastFrom<f32>
    + CastFrom<f64>
    + CastFrom<bool>
{
    const DTYPE: DType;

    // Storage construction
    fn into_cpu_storage(data: &[Self]) -> CPUStorage;
    #[cfg(feature = "cuda")]
    fn into_cuda_storage(data: &[Self]) -> CUDAStorage;
    #[cfg(feature = "cuda")]
    fn into_cuda_storage_from_cuda(data: CudaSlice<Self>) -> CUDAStorage;

    // Storage access
    fn cpu_storage_as_slice(s: &CPUStorage) -> Option<&[Self]>;
    fn cpu_storage_as_slice_mut(s: &mut CPUStorage) -> Option<&mut [Self]>;
    #[cfg(feature = "cuda")]
    fn cuda_storage_as_slice(s: &CUDAStorage) -> Option<&CudaSlice<Self>>;
    #[cfg(feature = "cuda")]
    fn cuda_storage_as_slice_mut(s: &mut CUDAStorage) -> Option<&mut CudaSlice<Self>>;

    #[cfg(feature = "cuda")]
    fn cuda_storage_from_cpu(storage: &CPUStorage) -> CUDAStorage;
    #[cfg(feature = "cuda")]
    fn cpu_storage_from_cuda(storage: &CUDAStorage) -> CPUStorage;
}

macro_rules! impl_storage_repr {
    ($($t:ty => $cpu_variant:ident, $cuda_variant:ident, $dtype:expr),+ $(,)?) => {
        $(
            impl StorageRepr for $t {
                const DTYPE: DType = $dtype;

                fn into_cpu_storage(data: &[Self]) -> CPUStorage {
                    CPUStorage::$cpu_variant(data.to_vec())
                }

                fn cpu_storage_as_slice(s: &CPUStorage) -> Option<&[Self]> {
                    if let CPUStorage::$cpu_variant(v) = s { Some(v) } else { None }
                }

                fn cpu_storage_as_slice_mut(s: &mut CPUStorage) -> Option<&mut [Self]> {
                    if let CPUStorage::$cpu_variant(v) = s { Some(v) } else { None }
                }

                #[cfg(feature = "cuda")]
                fn into_cuda_storage(data: &[Self]) -> CUDAStorage {
                    let data = array_to_cuda_slice(data);
                    CUDAStorage::$cuda_variant(data)
                }

                #[cfg(feature = "cuda")]
                fn into_cuda_storage_from_cuda(data: CudaSlice<Self>) -> CUDAStorage {
                    CUDAStorage::$cuda_variant(data)
                }

                #[cfg(feature = "cuda")]
                fn cuda_storage_as_slice(s: &CUDAStorage) -> Option<&CudaSlice<Self>> {
                    if let CUDAStorage::$cuda_variant(v) = s { Some(v) } else { None }
                }

                #[cfg(feature = "cuda")]
                fn cuda_storage_as_slice_mut(s: &mut CUDAStorage) -> Option<&mut CudaSlice<Self>> {
                    if let CUDAStorage::$cuda_variant(v) = s { Some(v) } else { None }
                }

                #[cfg(feature = "cuda")]
                fn cuda_storage_from_cpu(s: &CPUStorage) -> CUDAStorage {
                    Self::into_cuda_storage(Self::cpu_storage_as_slice(s).unwrap())
                }

                #[cfg(feature = "cuda")]
                fn cpu_storage_from_cuda(s: &CUDAStorage) -> CPUStorage {
                    Self::into_cpu_storage(&crate::cuda::cuda_slice_to_array(Self::cuda_storage_as_slice(s).unwrap()))
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
