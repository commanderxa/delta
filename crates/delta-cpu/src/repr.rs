use delta_core::{cast::CastFrom, dtype::DType, f8};
use half::{bf16, f16};

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
