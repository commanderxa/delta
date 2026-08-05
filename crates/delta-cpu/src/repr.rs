use delta_core::{dtype::DType, f8, repr::TensorRepr};
use half::{bf16, f16};

use crate::storage::CPUStorage;

pub trait CPUStorageRepr: Sized + Copy + Clone + TensorRepr {
    const DTYPE: DType;

    // Storage construction
    fn into_storage(data: &[Self]) -> CPUStorage;
    // Storage access
    fn storage_as_slice(s: &CPUStorage) -> Option<&[Self]>;
    fn storage_as_slice_mut(s: &mut CPUStorage) -> Option<&mut [Self]>;
}

macro_rules! impl_storage_repr {
    ($($t:ty => $cpu_variant:ident, $cuda_variant:ident, $dtype:expr),+ $(,)?) => {
        $(
            impl CPUStorageRepr for $t {
                const DTYPE: DType = $dtype;

                fn into_storage(data: &[Self]) -> CPUStorage {
                    CPUStorage::$cpu_variant(data.to_vec())
                }

                fn storage_as_slice(s: &CPUStorage) -> Option<&[Self]> {
                    if let CPUStorage::$cpu_variant(v) = s { Some(v) } else { None }
                }

                fn storage_as_slice_mut(s: &mut CPUStorage) -> Option<&mut [Self]> {
                    if let CPUStorage::$cpu_variant(v) = s { Some(v) } else { None }
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
