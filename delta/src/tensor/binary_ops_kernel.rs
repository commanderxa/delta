use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{Storage, Tensor, op::Op, tensor::repr::TensorRepr};

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
        let lhs_f = self.cast::<f32>();
        let rhs_f = rhs.cast::<f32>().pow(-1);
        // dispatch to a float-typed element-wise div kernel
        (lhs_f * rhs_f).cast::<f32>()
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
                        *x != *x;
                    }
                }
            },
            #[cfg(feature = "cuda")]
            Storage::CUDA(_) => todo!("CUDA neg"),
        }
        self
    }
}

impl<T: TensorRepr> Add<T> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: T) -> Self::Output {
        self.inner.borrow_mut().data.map_inplace(|x: T| x + rhs);
        self
    }
}

impl<T: TensorRepr> AddAssign<T> for Tensor {
    fn add_assign(&mut self, rhs: T) {
        match &mut self.inner.borrow_mut().data {
            Storage::CPU(data) => {
                if let Some(v) = T::cpu_storage_as_slice_mut(data) {
                    for x in v.iter_mut() {
                        *x = *x + rhs;
                    }
                }
            }
            #[cfg(feature = "cuda")]
            Storage::CUDA(_) => todo!("CUDA add_assign"),
        }
    }
}

impl<T: TensorRepr> Sub<T> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: T) -> Self::Output {
        self.inner.borrow_mut().data.map_inplace(|x: T| x - rhs);
        self
    }
}

impl<T: TensorRepr> SubAssign<T> for Tensor {
    fn sub_assign(&mut self, rhs: T) {
        match &mut self.inner.borrow_mut().data {
            Storage::CPU(data) => {
                if let Some(v) = T::cpu_storage_as_slice_mut(data) {
                    for x in v.iter_mut() {
                        *x = *x - rhs; // requires T: Add<Output=T>
                    }
                }
            }
            #[cfg(feature = "cuda")]
            Storage::CUDA(_) => todo!("CUDA add_assign"),
        }
    }
}

impl<T: TensorRepr> Mul<T> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: T) -> Self::Output {
        self.inner.borrow_mut().data.map_inplace(|x: T| x * rhs);
        self
    }
}

impl<T: TensorRepr> MulAssign<T> for Tensor {
    fn mul_assign(&mut self, rhs: T) {
        match &mut self.inner.borrow_mut().data {
            Storage::CPU(data) => {
                if let Some(v) = T::cpu_storage_as_slice_mut(data) {
                    for x in v.iter_mut() {
                        *x = *x * rhs; // requires T: Add<Output=T>
                    }
                }
            }
            #[cfg(feature = "cuda")]
            Storage::CUDA(_) => todo!("CUDA add_assign"),
        }
    }
}

impl<T: TensorRepr> Div<T> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: T) -> Self::Output {
        self.inner.borrow_mut().data.map_inplace(|x: T| x / rhs);
        self
    }
}

impl<T: TensorRepr> DivAssign<T> for Tensor {
    fn div_assign(&mut self, rhs: T) {
        match &mut self.inner.borrow_mut().data {
            Storage::CPU(data) => {
                if let Some(v) = T::cpu_storage_as_slice_mut(data) {
                    for x in v.iter_mut() {
                        *x = *x / rhs; // requires T: Add<Output=T>
                    }
                }
            }
            #[cfg(feature = "cuda")]
            Storage::CUDA(_) => todo!("CUDA add_assign"),
        }
    }
}
