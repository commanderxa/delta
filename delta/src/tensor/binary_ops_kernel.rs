use std::ops::{Add, Div, Mul, Neg, Sub};

use half::{bf16, f16};

use crate::{
    Tensor, f8,
    op::Op,
    tensor::{
        element::{TensorElement, TensorFloat, TensorNum},
        promote_primitives::PromoteInto,
    },
};

impl<T: TensorElement> Add for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        Self::multicast_op(self, rhs, Op::Add)
    }
}

impl<T: TensorElement> Sub for Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::multicast_op(self, rhs, Op::Sub)
    }
}

impl<T: TensorElement> Mul for Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::multicast_op(self, rhs, Op::Mul)
    }
}

impl<T: TensorElement> Div for Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: Self) -> Self::Output {
        let lhs_f = self.cast::<f32>();
        let rhs_f = rhs.cast::<f32>().pow(-1);
        // dispatch to a float-typed element-wise div kernel
        (lhs_f * rhs_f).cast::<T>()
    }
}

impl<T: TensorNum> Neg for Tensor<T> {
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        self.inner.borrow_mut().data.map_inplace(|x| -x);
        self
    }
}

macro_rules! impl_tensor_scalar_ops {
    ($($scalar:ty),+) => {
        $(
            impl<T: TensorNum> Add<$scalar> for Tensor<T>
            where
                $scalar: PromoteInto<T>
            {
                type Output = Tensor<T>;
                fn add(self, rhs: $scalar) -> Self::Output {
                    let rhs: T = rhs.promote_into();
                    self.inner.borrow_mut().data.map_inplace(|x| x + rhs);
                    self
                }
            }

            impl<T: TensorNum> Add<Tensor<T>> for $scalar
            where
                $scalar: PromoteInto<T>
            {
                type Output = Tensor<T>;
                fn add(self, rhs: Tensor<T>) -> Self::Output {
                    rhs + self  // reuse above
                }
            }

            impl<T: TensorNum> Sub<$scalar> for Tensor<T>
            where
                $scalar: PromoteInto<T>
            {
                type Output = Tensor<T>;
                fn sub(self, rhs: $scalar) -> Self::Output {
                    let rhs: T = rhs.promote_into();
                    self.inner.borrow_mut().data.map_inplace(|x| x - rhs);
                    self
                }
            }

            impl<T: TensorNum> Sub<Tensor<T>> for $scalar
            where
                $scalar: PromoteInto<T>
            {
                type Output = Tensor<T>;
                fn sub(self, rhs: Tensor<T>) -> Self::Output {
                    rhs - self  // reuse above
                }
            }

            impl<T: TensorNum> Mul<$scalar> for Tensor<T>
            where
                $scalar: PromoteInto<T>
            {
                type Output = Tensor<T>;
                fn mul(self, rhs: $scalar) -> Self::Output {
                    let rhs: T = rhs.promote_into();
                    self.inner.borrow_mut().data.map_inplace(|x| x * rhs);
                    self
                }
            }

            impl<T: TensorNum> Mul<Tensor<T>> for $scalar
            where
                $scalar: PromoteInto<T>
            {
                type Output = Tensor<T>;
                fn mul(self, rhs: Tensor<T>) -> Self::Output {
                    rhs * self  // reuse above
                }
            }
            impl<T: TensorNum> Div<$scalar> for Tensor<T>
            where
                $scalar: PromoteInto<T>
            {
                type Output = Tensor<T>;
                fn div(self, rhs: $scalar) -> Self::Output {
                    let rhs: T = rhs.promote_into();
                    self.inner.borrow_mut().data.map_inplace(|x| x / rhs);
                    self
                }
            }

            impl<T: TensorNum> Div<Tensor<T>> for $scalar
            where
                $scalar: PromoteInto<T>
            {
                type Output = Tensor<T>;
                fn div(self, rhs: Tensor<T>) -> Self::Output {
                    rhs / self  // reuse above
                }
            }
        )+
    };
}

// impl_tensor_scalar_ops!(i8, i16, i32, i64, f8, f16, bf16, f32, f64);

impl<T: TensorFloat> Add<T> for Tensor<T> {
    type Output = Tensor<T>;
    fn add(self, rhs: T) -> Self::Output {
        self.inner.borrow_mut().data.map_inplace(|x| x + rhs);
        self
    }
}

impl<T: TensorFloat> Sub<T> for Tensor<T> {
    type Output = Tensor<T>;
    fn sub(self, rhs: T) -> Self::Output {
        self.inner.borrow_mut().data.map_inplace(|x| x - rhs);
        self
    }
}

impl<T: TensorFloat> Mul<T> for Tensor<T> {
    type Output = Tensor<T>;
    fn mul(self, rhs: T) -> Self::Output {
        self.inner.borrow_mut().data.map_inplace(|x| x * rhs);
        self
    }
}

impl<T: TensorFloat> Div<T> for Tensor<T> {
    type Output = Tensor<T>;
    fn div(self, rhs: T) -> Self::Output {
        self.inner.borrow_mut().data.map_inplace(|x| x / rhs);
        self
    }
}
