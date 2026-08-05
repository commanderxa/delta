use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use delta_core::cast::Cast;
use half::{bf16, f16};

use crate::{
    Storage, Tensor, f8,
    op::Op,
    tensor::{repr::NumTensorRepr},
};

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        Self::multicast_op(self, rhs, Op::Add)
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::multicast_op(self, rhs, Op::Sub)
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::multicast_op(self, rhs, Op::Mul)
    }
}

impl Div for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.pow(-1)
    }
}

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        match &mut self.inner.borrow_mut().data {
            Storage::CPU(data) => match data {
                super::storage_impl::CPUStorage::I8(v) => {
                    for x in v.iter_mut() {
                        *x = -*x;
                    }
                }
                super::storage_impl::CPUStorage::I16(v) => {
                    for x in v.iter_mut() {
                        *x = -*x;
                    }
                }
                super::storage_impl::CPUStorage::I32(v) => {
                    for x in v.iter_mut() {
                        *x = -*x;
                    }
                }
                super::storage_impl::CPUStorage::I64(v) => {
                    for x in v.iter_mut() {
                        *x = -*x;
                    }
                }
                super::storage_impl::CPUStorage::F8(v) => {
                    for x in v.iter_mut() {
                        *x = -*x;
                    }
                }
                super::storage_impl::CPUStorage::F16(v) => {
                    for x in v.iter_mut() {
                        *x = -*x;
                    }
                }
                super::storage_impl::CPUStorage::BF16(v) => {
                    for x in v.iter_mut() {
                        *x = -*x;
                    }
                }
                super::storage_impl::CPUStorage::F32(v) => {
                    for x in v.iter_mut() {
                        *x = -*x;
                    }
                }
                super::storage_impl::CPUStorage::F64(v) => {
                    for x in v.iter_mut() {
                        *x = -*x;
                    }
                }
                super::storage_impl::CPUStorage::Bool(v) => {
                    for x in v.iter_mut() {
                        *x = !*x;
                    }
                }
            },
            #[cfg(feature = "cuda")]
            Storage::CUDA(_) => todo!("CUDA neg"),
        }
        self
    }
}

impl<T: NumTensorRepr> Add<T> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: T) -> Self::Output {
        match self.dtype() {
            crate::DType::Float8 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: f8| x + Cast::<f8>::cast(rhs));
            }
            crate::DType::Float16 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: f16| x + Cast::<f16>::cast(rhs));
            },
            crate::DType::BFloat16 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: bf16| x + Cast::<bf16>::cast(rhs));
            },
            crate::DType::Float32 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: f32| x + Cast::<f32>::cast(rhs));
            },
            crate::DType::Float64 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: f64| x + Cast::<f64>::cast(rhs));
            },
            crate::DType::Int8 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: i8| x + Cast::<i8>::cast(rhs));
            },
            crate::DType::Int16 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: i16| x + Cast::<i16>::cast(rhs));
            },
            crate::DType::Int32 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: i32| x + Cast::<i32>::cast(rhs));
            },
            crate::DType::Int64 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: i64| x + Cast::<i64>::cast(rhs));
            },
            crate::DType::Bool => todo!(),
        }
        self
    }
}

impl<T: NumTensorRepr> AddAssign<T> for Tensor {
    fn add_assign(&mut self, rhs: T) {
        match &mut self.inner.borrow_mut().data {
            Storage::CPU(data) => {
                if let Some(v) = T::cpu_storage_as_slice_mut(data) {
                    for x in v.iter_mut() {
                        *x = *x + Cast::cast(rhs);
                    }
                }
            }
            #[cfg(feature = "cuda")]
            Storage::CUDA(_) => todo!("CUDA add_assign"),
        }
    }
}

impl<T: NumTensorRepr> Sub<T> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: T) -> Self::Output {
        match self.dtype() {
            crate::DType::Float8 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: f8| x - Cast::<f8>::cast(rhs));
            }
            crate::DType::Float16 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: f16| x - Cast::<f16>::cast(rhs));
            }
            crate::DType::BFloat16 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: bf16| x - Cast::<bf16>::cast(rhs));
            }
            crate::DType::Float32 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: f32| x - Cast::<f32>::cast(rhs));
            }
            crate::DType::Float64 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: f64| x - Cast::<f64>::cast(rhs));
            }
            crate::DType::Int8 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: i8| x - Cast::<i8>::cast(rhs));
            }
            crate::DType::Int16 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: i16| x - Cast::<i16>::cast(rhs));
            }
            crate::DType::Int32 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: i32| x - Cast::<i32>::cast(rhs));
            }
            crate::DType::Int64 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: i64| x - Cast::<i64>::cast(rhs));
            }
            crate::DType::Bool => todo!(),
        }
        self
    }
}

impl<T: NumTensorRepr> SubAssign<T> for Tensor {
    fn sub_assign(&mut self, rhs: T) {
        match &mut self.inner.borrow_mut().data {
            Storage::CPU(data) => {
                if let Some(v) = T::cpu_storage_as_slice_mut(data) {
                    for x in v.iter_mut() {
                        *x = *x - Cast::cast(rhs); // requires T: Add<Output=T>
                    }
                }
            }
            #[cfg(feature = "cuda")]
            Storage::CUDA(_) => todo!("CUDA add_assign"),
        }
    }
}

impl<T: NumTensorRepr> Mul<T> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: T) -> Self::Output {
        match self.dtype() {
            crate::DType::Float8 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: f8| x * Cast::<f8>::cast(rhs));
            }
            crate::DType::Float16 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: f16| x * Cast::<f16>::cast(rhs));
            }
            crate::DType::BFloat16 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: bf16| x * Cast::<bf16>::cast(rhs));
            }
            crate::DType::Float32 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: f32| x * Cast::<f32>::cast(rhs));
            }
            crate::DType::Float64 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: f64| x * Cast::<f64>::cast(rhs));
            }
            crate::DType::Int8 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: i8| x * Cast::<i8>::cast(rhs));
            }
            crate::DType::Int16 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: i16| x * Cast::<i16>::cast(rhs));
            }
            crate::DType::Int32 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: i32| x * Cast::<i32>::cast(rhs));
            }
            crate::DType::Int64 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: i64| x * Cast::<i64>::cast(rhs));
            }
            crate::DType::Bool => todo!(),
        }
        self
    }
}

impl<T: NumTensorRepr> MulAssign<T> for Tensor {
    fn mul_assign(&mut self, rhs: T) {
        match &mut self.inner.borrow_mut().data {
            Storage::CPU(data) => {
                if let Some(v) = T::cpu_storage_as_slice_mut(data) {
                    for x in v.iter_mut() {
                        *x = *x * Cast::cast(rhs); // requires T: Add<Output=T>
                    }
                }
            }
            #[cfg(feature = "cuda")]
            Storage::CUDA(_) => todo!("CUDA add_assign"),
        }
    }
}

impl<T: NumTensorRepr> Div<T> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: T) -> Self::Output {
        match self.dtype() {
            crate::DType::Float8 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: f8| x / Cast::<f8>::cast(rhs));
            }
            crate::DType::Float16 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: f16| x / Cast::<f16>::cast(rhs));
            }
            crate::DType::BFloat16 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: bf16| x / Cast::<bf16>::cast(rhs));
            }
            crate::DType::Float32 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: f32| x / Cast::<f32>::cast(rhs));
            }
            crate::DType::Float64 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: f64| x / Cast::<f64>::cast(rhs));
            }
            crate::DType::Int8 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: i8| x / Cast::<i8>::cast(rhs));
            }
            crate::DType::Int16 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: i16| x / Cast::<i16>::cast(rhs));
            }
            crate::DType::Int32 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: i32| x / Cast::<i32>::cast(rhs));
            }
            crate::DType::Int64 => {
                self.inner
                    .borrow_mut()
                    .data
                    .map_inplace(|x: i64| x / Cast::<i64>::cast(rhs));
            }
            crate::DType::Bool => todo!(),
        }
        self
    }
}

impl<T: NumTensorRepr> DivAssign<T> for Tensor {
    fn div_assign(&mut self, rhs: T) {
        match &mut self.inner.borrow_mut().data {
            Storage::CPU(data) => {
                if let Some(v) = T::cpu_storage_as_slice_mut(data) {
                    for x in v.iter_mut() {
                        *x = *x / Cast::cast(rhs); // requires T: Add<Output=T>
                    }
                }
            }
            #[cfg(feature = "cuda")]
            Storage::CUDA(_) => todo!("CUDA add_assign"),
        }
    }
}
