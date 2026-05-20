use crate::DEFAULT_DTYPE;

#[allow(non_camel_case_types)]
pub type f8 = float8::F8E4M3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DType {
    Float8,
    Float16,
    BFloat16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Bool,
}

impl DType {
    pub fn itemsize(&self) -> usize {
        match self {
            DType::Float8 => 1,
            DType::Float16 => 2,
            DType::BFloat16 => 2,
            DType::Float32 => 4,
            DType::Float64 => 8,
            DType::Int8 => 1,
            DType::Int16 => 2,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::Bool => 1,
        }
    }

    pub fn rank(&self) -> u8 {
        if matches!(self, DType::Bool) {
            1
        } else if matches!(
            self,
            DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64
        ) {
            2
        } else {
            3
        }
    }

    pub fn is_float(&self) -> bool {
        matches!(
            self,
            DType::Float8 | DType::Float16 | DType::BFloat16 | DType::Float32 | DType::Float64
        )
    }

    pub fn is_int(&self) -> bool {
        matches!(
            self,
            DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64
        )
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, DType::Bool)
    }
}

#[allow(non_upper_case_globals)]
pub const float8: DType = DType::Float8;
#[allow(non_upper_case_globals)]
pub const float16: DType = DType::Float16;
#[allow(non_upper_case_globals)]
pub const bfloat16: DType = DType::BFloat16;
#[allow(non_upper_case_globals)]
pub const float32: DType = DType::Float32;
#[allow(non_upper_case_globals)]
pub const float64: DType = DType::Float64;
#[allow(non_upper_case_globals)]
pub const int8: DType = DType::Int8;
#[allow(non_upper_case_globals)]
pub const int16: DType = DType::Int16;
#[allow(non_upper_case_globals)]
pub const int32: DType = DType::Int32;
#[allow(non_upper_case_globals)]
pub const int64: DType = DType::Int64;
#[allow(non_upper_case_globals)]
pub const bool: DType = DType::Bool;

pub fn get_default_dtype() -> DType {
    *DEFAULT_DTYPE.read().unwrap()
}

pub fn set_default_dtype(dtype: DType) {
    *DEFAULT_DTYPE.write().unwrap() = dtype;
}
