use half::{f16, bf16};

use crate::f8;

pub trait PromoteInto<T> {
    fn promote_into(self) -> T;
}

impl PromoteInto<i8> for i8  { fn promote_into(self) -> i8 { self } }
impl PromoteInto<i8> for i16  { fn promote_into(self) -> i8 { self as i8 } }
impl PromoteInto<i8> for i32  { fn promote_into(self) -> i8 { self as i8 } }
impl PromoteInto<i8> for i64  { fn promote_into(self) -> i8 { self as i8 } }
impl PromoteInto<i8> for f8   { fn promote_into(self) -> i8 { self.to_f32() as i8 } }
impl PromoteInto<i8> for f16  { fn promote_into(self) -> i8 { self.to_f32() as i8 } }
impl PromoteInto<i8> for bf16 { fn promote_into(self) -> i8 { self.to_f32() as i8 } }
impl PromoteInto<i8> for f32  { fn promote_into(self) -> i8 { self as i8 } }
impl PromoteInto<i8> for f64   { fn promote_into(self) -> i8 { self as i8 } }

impl PromoteInto<i16> for i8  { fn promote_into(self) -> i16 { self as i16 } }
impl PromoteInto<i16> for i16  { fn promote_into(self) -> i16 { self } }
impl PromoteInto<i16> for i32  { fn promote_into(self) -> i16 { self as i16 } }
impl PromoteInto<i16> for i64  { fn promote_into(self) -> i16 { self as i16 } }
impl PromoteInto<i16> for f8   { fn promote_into(self) -> i16 { self.to_f32() as i16 } }
impl PromoteInto<i16> for f16  { fn promote_into(self) -> i16 { self.to_f32() as i16 } }
impl PromoteInto<i16> for bf16 { fn promote_into(self) -> i16 { self.to_f32() as i16 } }
impl PromoteInto<i16> for f32  { fn promote_into(self) -> i16 { self as i16 } }
impl PromoteInto<i16> for f64   { fn promote_into(self) -> i16 { self as i16 } }

impl PromoteInto<i32> for i8  { fn promote_into(self) -> i32 { self as i32 } }
impl PromoteInto<i32> for i16  { fn promote_into(self) -> i32 { self as i32 } }
impl PromoteInto<i32> for i32  { fn promote_into(self) -> i32 { self } }
impl PromoteInto<i32> for i64  { fn promote_into(self) -> i32 { self as i32 } }
impl PromoteInto<i32> for f8   { fn promote_into(self) -> i32 { self.to_f32() as i32 } }
impl PromoteInto<i32> for f16  { fn promote_into(self) -> i32 { self.to_f32() as i32 } }
impl PromoteInto<i32> for bf16 { fn promote_into(self) -> i32 { self.to_f32() as i32 } }
impl PromoteInto<i32> for f32  { fn promote_into(self) -> i32 { self as i32 } }
impl PromoteInto<i32> for f64   { fn promote_into(self) -> i32 { self as i32 } }

impl PromoteInto<i64> for i8  { fn promote_into(self) -> i64 { self as i64 } }
impl PromoteInto<i64> for i16  { fn promote_into(self) -> i64 { self as i64 } }
impl PromoteInto<i64> for i32  { fn promote_into(self) -> i64 { self as i64 } }
impl PromoteInto<i64> for i64  { fn promote_into(self) -> i64 { self } }
impl PromoteInto<i64> for f8   { fn promote_into(self) -> i64 { self.to_f32() as i64 } }
impl PromoteInto<i64> for f16  { fn promote_into(self) -> i64 { self.to_f32() as i64 } }
impl PromoteInto<i64> for bf16 { fn promote_into(self) -> i64 { self.to_f32() as i64 } }
impl PromoteInto<i64> for f32  { fn promote_into(self) -> i64 { self as i64 } }
impl PromoteInto<i64> for f64   { fn promote_into(self) -> i64 { self as i64 } }

impl PromoteInto<f8> for i8   { fn promote_into(self) -> f8 { f8::from_f32(self as f32) } }
impl PromoteInto<f8> for i16  { fn promote_into(self) -> f8 { f8::from_f32(self as f32) } }
impl PromoteInto<f8> for i32  { fn promote_into(self) -> f8 { f8::from_f32(self as f32) } }
impl PromoteInto<f8> for i64  { fn promote_into(self) -> f8 { f8::from_f32(self as f32) } }
impl PromoteInto<f8> for f8   { fn promote_into(self) -> f8 { self } }
impl PromoteInto<f8> for f16  { fn promote_into(self) -> f8 { f8::from_f32(self.to_f32()) } }
impl PromoteInto<f8> for bf16 { fn promote_into(self) -> f8 { f8::from_f32(self.to_f32()) } }
impl PromoteInto<f8> for f32  { fn promote_into(self) -> f8 { f8::from_f32(self) } }
impl PromoteInto<f8> for f64  { fn promote_into(self) -> f8 { f8::from_f32(self as f32) } }

impl PromoteInto<f16> for i8   { fn promote_into(self) -> f16 { f16::from_f32(self as f32) } }
impl PromoteInto<f16> for i16  { fn promote_into(self) -> f16 { f16::from_f32(self as f32) } }
impl PromoteInto<f16> for i32  { fn promote_into(self) -> f16 { f16::from_f32(self as f32) } }
impl PromoteInto<f16> for i64  { fn promote_into(self) -> f16 { f16::from_f32(self as f32) } }
impl PromoteInto<f16> for f8  { fn promote_into(self) -> f16 { f16::from_f32(self.to_f32()) } }
impl PromoteInto<f16> for f16  { fn promote_into(self) -> f16 { self } }
impl PromoteInto<f16> for bf16 { fn promote_into(self) -> f16 { f16::from_f32(self.to_f32()) } }
impl PromoteInto<f16> for f32  { fn promote_into(self) -> f16 { f16::from_f32(self) } }
impl PromoteInto<f16> for f64  { fn promote_into(self) -> f16 { f16::from_f32(self as f32) } }

impl PromoteInto<bf16> for i8   { fn promote_into(self) -> bf16 { bf16::from_f32(self as f32) } }
impl PromoteInto<bf16> for i16  { fn promote_into(self) -> bf16 { bf16::from_f32(self as f32) } }
impl PromoteInto<bf16> for i32  { fn promote_into(self) -> bf16 { bf16::from_f32(self as f32) } }
impl PromoteInto<bf16> for i64  { fn promote_into(self) -> bf16 { bf16::from_f32(self as f32) } }
impl PromoteInto<bf16> for f8  { fn promote_into(self) -> bf16 { bf16::from_f32(self.to_f32()) } }
impl PromoteInto<bf16> for f16 { fn promote_into(self) -> bf16 { bf16::from_f32(self.to_f32()) } }
impl PromoteInto<bf16> for bf16  { fn promote_into(self) -> bf16 { self } }
impl PromoteInto<bf16> for f32  { fn promote_into(self) -> bf16 { bf16::from_f32(self) } }
impl PromoteInto<bf16> for f64  { fn promote_into(self) -> bf16 { bf16::from_f32(self as f32) } }

impl PromoteInto<f32> for i8   { fn promote_into(self) -> f32 { self as f32 } }
impl PromoteInto<f32> for i16  { fn promote_into(self) -> f32 { self as f32 } }
impl PromoteInto<f32> for i32  { fn promote_into(self) -> f32 { self as f32 } }
impl PromoteInto<f32> for i64  { fn promote_into(self) -> f32 { self as f32 } }
impl PromoteInto<f32> for f8   { fn promote_into(self) -> f32 { self.to_f32() } }
impl PromoteInto<f32> for f16  { fn promote_into(self) -> f32 { self.to_f32() } }
impl PromoteInto<f32> for bf16 { fn promote_into(self) -> f32 { self.to_f32() } }
impl PromoteInto<f32> for f32  { fn promote_into(self) -> f32 { self } }
impl PromoteInto<f32> for f64  { fn promote_into(self) -> f32 { self as f32 } }

impl PromoteInto<f64> for i8   { fn promote_into(self) -> f64 { self as f64 } }
impl PromoteInto<f64> for i16  { fn promote_into(self) -> f64 { self as f64 } }
impl PromoteInto<f64> for i32  { fn promote_into(self) -> f64 { self as f64 } }
impl PromoteInto<f64> for i64  { fn promote_into(self) -> f64 { self as f64 } }
impl PromoteInto<f64> for f8   { fn promote_into(self) -> f64 { self.to_f64() } }
impl PromoteInto<f64> for f16  { fn promote_into(self) -> f64 { self.to_f64() } }
impl PromoteInto<f64> for bf16 { fn promote_into(self) -> f64 { self.to_f64() } }
impl PromoteInto<f64> for f32  { fn promote_into(self) -> f64 { self as f64 } }
impl PromoteInto<f64> for f64  { fn promote_into(self) -> f64 { self } }