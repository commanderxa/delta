use std::fmt::{Debug, Display};
use std::ops::Range;

use half::{bf16, f16};
use num_traits::{Float, ToPrimitive};

use crate::cast::{Cast, CastFrom};
use crate::dtype::DType;
use crate::f8;

// pub trait StorageRepr:
//     Sized
//     + Copy
//     + Clone
//     + CastFrom<i8>
//     + CastFrom<i16>
//     + CastFrom<i32>
//     + CastFrom<i64>
//     + CastFrom<f8>
//     + CastFrom<f16>
//     + CastFrom<bf16>
//     + CastFrom<f32>
//     + CastFrom<f64>
//     + CastFrom<bool>
//     + CastFrom<Self> {
//     const DTYPE: DType;

//     // Storage construction
//     fn into_storage(data: &[Self]) -> CPUStorage;
//     // Storage access
//     fn storage_as_slice(s: &CPUStorage) -> Option<&[Self]>;
//     fn storage_as_slice_mut(s: &mut CPUStorage) -> Option<&mut [Self]>;
//     // function for other storage types
//     fn from_cpu_storage(storage: &CPUStorage) -> CUDAStorage;
//     fn into_cpu_storage(storage: &CUDAStorage) -> CPUStorage;
// }

pub trait TensorRepr:
    'static
    + Copy
    + Clone
    + Debug
    + Display
    + PartialEq
    + PartialOrd
    + Sized
    + Cast<Self>
    + Cast<bool>
    + Cast<f8>
    + Cast<f16>
    + Cast<bf16>
    + Cast<f32>
    + Cast<f64>
    + Cast<i8>
    + Cast<i16>
    + Cast<i32>
    + Cast<i64>
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
    + CastFrom<Self>
{
    const DTYPE: DType;
    fn dtype() -> DType;
    fn zero() -> Self;
    fn one() -> Self;
    fn max() -> Self;
}

macro_rules! impl_tensor_element {
    ($ty:ty, $dtype:expr, $zero:expr, $one:expr, $max:expr) => {
        impl TensorRepr for $ty {
            const DTYPE: DType = $dtype;

            fn dtype() -> DType {
                $dtype
            }

            fn zero() -> Self {
                $zero
            }

            fn one() -> Self {
                $one
            }

            fn max() -> Self {
                $max
            }
        }
    };
}

impl_tensor_element!(i8, DType::Int8, 0, 1, i8::MAX);
impl_tensor_element!(i16, DType::Int16, 0, 1, i16::MAX);
impl_tensor_element!(i32, DType::Int32, 0, 1, i32::MAX);
impl_tensor_element!(i64, DType::Int64, 0, 1, i64::MAX);
impl_tensor_element!(
    f8,
    DType::Float8,
    f8::from_f32(0.0),
    f8::from_f32(1.0),
    f8::MAX
);
impl_tensor_element!(
    f16,
    DType::Float16,
    f16::from_f32(0.0),
    f16::from_f32(1.0),
    f16::MAX
);
impl_tensor_element!(
    bf16,
    DType::BFloat16,
    bf16::from_f32(0.0),
    bf16::from_f32(1.0),
    bf16::MAX
);
impl_tensor_element!(f32, DType::Float32, 0.0, 1.0, f32::MAX);
impl_tensor_element!(f64, DType::Float64, 0.0, 1.0, f64::MAX);
impl_tensor_element!(bool, DType::Bool, false, true, true);

pub trait NumTensorRepr:
    TensorRepr
    + std::ops::Add<Output = Self>
    + std::ops::AddAssign
    + std::ops::Sub<Output = Self>
    + std::ops::SubAssign
    + std::ops::Mul<Output = Self>
    + std::ops::MulAssign
    + std::ops::Div<Output = Self>
    + std::ops::DivAssign
    + std::ops::Neg<Output = Self>
    + ToPrimitive
{
}

impl NumTensorRepr for f8 {}
impl NumTensorRepr for f16 {}
impl NumTensorRepr for bf16 {}
impl NumTensorRepr for f32 {}
impl NumTensorRepr for f64 {}
impl NumTensorRepr for i8 {}
impl NumTensorRepr for i16 {}
impl NumTensorRepr for i32 {}
impl NumTensorRepr for i64 {}

pub trait FloatTensorRepr: NumTensorRepr + Float {
    fn neg_infinity() -> Self;
    fn random_range(range: Range<Self>) -> Self;
}

impl FloatTensorRepr for f8 {
    fn neg_infinity() -> Self {
        f8::NEG_INFINITY
    }

    fn random_range(range: Range<Self>) -> Self {
        f8::from_f32(rand::random_range(
            <Self as Cast<f32>>::cast(range.start)..<Self as Cast<f32>>::cast(range.end),
        ))
    }
}
impl FloatTensorRepr for f16 {
    fn neg_infinity() -> Self {
        f16::NEG_INFINITY
    }

    fn random_range(range: Range<Self>) -> Self {
        f16::from_f32(rand::random_range(
            <Self as Cast<f32>>::cast(range.start)..<Self as Cast<f32>>::cast(range.end),
        ))
    }
}
impl FloatTensorRepr for bf16 {
    fn neg_infinity() -> Self {
        bf16::NEG_INFINITY
    }

    fn random_range(range: Range<Self>) -> Self {
        bf16::from_f32(rand::random_range(
            <Self as Cast<f32>>::cast(range.start)..<Self as Cast<f32>>::cast(range.end),
        ))
    }
}
impl FloatTensorRepr for f32 {
    fn neg_infinity() -> Self {
        f32::NEG_INFINITY
    }

    fn random_range(range: Range<Self>) -> Self {
        rand::random_range(
            <Self as Cast<f32>>::cast(range.start)..<Self as Cast<f32>>::cast(range.end),
        )
    }
}
impl FloatTensorRepr for f64 {
    fn neg_infinity() -> Self {
        f64::NEG_INFINITY
    }

    fn random_range(range: Range<Self>) -> Self {
        rand::random_range(
            <Self as Cast<f64>>::cast(range.start)..<Self as Cast<f64>>::cast(range.end),
        )
    }
}
