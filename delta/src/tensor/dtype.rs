#[allow(non_camel_case_types)]
pub type f8 = float8::F8E4M3;

#[derive(Debug, Clone)]
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
}
