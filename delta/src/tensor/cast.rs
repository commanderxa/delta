use half::{bf16, f16};
use num_traits::AsPrimitive;

use crate::f8;

pub trait Cast<U> {
    fn cast(self) -> U;
}

macro_rules! impl_cast_primitives {
    // Native: use AsPrimitive / as-cast
    ($from:ty => $($to:ty),+ $(;)?) => {
        $(
            impl Cast<$to> for $from {
                #[inline]
                fn cast(self) -> $to { self.as_() }
            }
        )*
    };
    // Exotic from: route through f32
    (via_f32: $from:ty => $($to:ty),+ $(;)?) => {
        $(
            impl Cast<$to> for $from {
                #[inline]
                fn cast(self) -> $to { self.to_f32().as_() }
            }
        )*
    };
}

impl_cast_primitives! {f8 => i8, i16, i32, i64, f8, f32, f64}
impl_cast_primitives! {f16 => i8, i16, i32, i64, f16, bf16, f32, f64}
impl_cast_primitives! {bf16 => i8, i16, i32, i64, f16, bf16, f32, f64}
impl_cast_primitives! {via_f32: f8 => f16, bf16}
impl_cast_primitives! {via_f32: f16 => f8}
impl_cast_primitives! {via_f32: bf16 => f8}

impl_cast_primitives! {f32 => i8, i16, i32, i64, f8, f16, bf16, f32, f64}
impl_cast_primitives! {f64 => i8, i16, i32, i64, f8, f16, bf16, f32, f64}

impl_cast_primitives! {i8 => i8, i16, i32, i64, f8, f16, bf16, f32, f64}
impl_cast_primitives! {i16 => i8, i16, i32, i64, f8, f16, bf16, f32, f64}
impl_cast_primitives! {i32 => i8, i16, i32, i64, f8, f16, bf16, f32, f64}
impl_cast_primitives! {i64 => i8, i16, i32, i64, f8, f16, bf16, f32, f64}
