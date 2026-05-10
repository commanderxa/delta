use std::fmt::{Debug, Display};
use std::ops::Range;

#[cfg(feature = "cuda")]
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use half::{bf16, f16};
use num_traits::{Float};

use crate::tensor::cast::Cast;
use crate::tensor::storage_impl::StorageRepr;
use crate::{DType, f8};

#[cfg(not(feature = "cuda"))]
pub trait TensorRepr:
    'static
    + Copy
    + Clone
    + Debug
    + Display
    + PartialEq
    + Sized
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + Cast<Self>
    + PartialOrd
    + StorageRepr
{
    fn dtype() -> DType;
    fn zero() -> Self;
    fn one() -> Self;
    fn max() -> Self;
}

#[cfg(feature = "cuda")]
pub trait TensorRepr:
    'static
    + Copy
    + Clone
    + Debug
    + Display
    + PartialEq
    + Sized
    + std::ops::Add<Output = Self>
    + std::ops::AddAssign
    + std::ops::Sub<Output = Self>
    + std::ops::SubAssign
    + std::ops::Mul<Output = Self>
    + std::ops::MulAssign
    + std::ops::Div<Output = Self>
    + std::ops::DivAssign
    + std::ops::Neg<Output = Self>
    + PartialOrd
    + Cast<Self>
    + StorageRepr
    + DeviceRepr
    + ValidAsZeroBits
{
    fn dtype() -> DType;
    fn zero() -> Self;
    fn one() -> Self;
    fn max() -> Self;
}

macro_rules! impl_tensor_element {
    ($ty:ty, $dtype:expr, $zero:expr, $one:expr, $max:expr) => {
        impl TensorRepr for $ty {
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
// impl_tensor_element!(bool, DType::Bool, false, true, true);

pub trait FloatTensorRepr: TensorRepr + Float {
    fn neg_infinity() -> Self;
    fn random_range(range: Range<Self>) -> Self;
}

impl FloatTensorRepr for f8 {
    fn neg_infinity() -> Self {
        f8::NEG_INFINITY
    }

    fn random_range(range: Range<Self>) -> Self {
        f8::from_f32(rand::random_range(
            <Self as Cast<f32>>::cast(range.start)
                ..<Self as Cast<f32>>::cast(range.end),
        ))
    }
}
impl FloatTensorRepr for f16 {
    fn neg_infinity() -> Self {
        f16::NEG_INFINITY
    }

    fn random_range(range: Range<Self>) -> Self {
        f16::from_f32(rand::random_range(
            <Self as Cast<f32>>::cast(range.start)
                ..<Self as Cast<f32>>::cast(range.end),
        ))
    }
}
impl FloatTensorRepr for bf16 {
    fn neg_infinity() -> Self {
        bf16::NEG_INFINITY
    }

    fn random_range(range: Range<Self>) -> Self {
        bf16::from_f32(rand::random_range(
            <Self as Cast<f32>>::cast(range.start)
                ..<Self as Cast<f32>>::cast(range.end),
        ))
    }
}
impl FloatTensorRepr for f32 {
    fn neg_infinity() -> Self {
        f32::NEG_INFINITY
    }

    fn random_range(range: Range<Self>) -> Self {
        rand::random_range(
            <Self as Cast<f32>>::cast(range.start)
                ..<Self as Cast<f32>>::cast(range.end),
        )
    }
}
impl FloatTensorRepr for f64 {
    fn neg_infinity() -> Self {
        f64::NEG_INFINITY
    }

    fn random_range(range: Range<Self>) -> Self {
        rand::random_range(
            <Self as Cast<f64>>::cast(range.start)
                ..<Self as Cast<f64>>::cast(range.end),
        )
    }
}
