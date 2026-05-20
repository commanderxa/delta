use half::{bf16, f16};
use num_traits::AsPrimitive;

#[allow(non_camel_case_types)]
pub type f8 = float8::F8E4M3;

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

    (via_i32: $from:ty => $($to:ty),+ $(;)?) => {
        $(
            impl Cast<$to> for $from {
                #[inline]
                fn cast(self) -> $to { (self as i32).as_() }
            }
        )*
    };

    (to_bool: $($from:ty),+ $(;)?) => {
        $(
            impl Cast<bool> for $from {
                #[inline]
                fn cast(self) -> bool {
                    self != <$from as num_traits::Zero>::zero()
                }
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

impl_cast_primitives! {via_i32: bool => i8, i16, i32, i64, f8, f16, bf16, f32, f64}
impl_cast_primitives! {to_bool: i8, i16, i32, i64, f8, f16, bf16, f32, f64}

impl Cast<bool> for bool {
    #[inline]
    fn cast(self) -> bool {
        self != false
    }
}

pub trait CastFrom<T>: Sized {
    fn cast_from(val: T) -> Self;
}

impl<T, U> CastFrom<U> for T
where
    U: Cast<T>,
{
    fn cast_from(val: U) -> T {
        Cast::cast(val)
    }
}
