use half::{bf16, f16};
use num_traits::AsPrimitive;

use delta_tensor::{FloatTensorRepr, Tensor, TensorImpl, TensorRepr, f8, op::Op};

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub enum Reduction {
    SUM,
    MEAN,
}

#[derive(Clone)]
pub struct MSELoss {
    reduction: Option<Reduction>,
}

impl MSELoss {
    pub fn new(reduction: Option<Reduction>) -> Self {
        Self {
            reduction: reduction,
        }
    }

    pub fn measure(&self, a: Tensor, b: Tensor) -> Tensor {
        let dtype = a.dtype();
        let t = (a - b).pow(2) * 0.5;
        let t_len = t.length();
        let inner = match dtype {
            delta_tensor::float8 => {
                let a = t.data();
                let mut s = f8::zero();
                if let Some(reduction) = self.reduction {
                    s = a.iter().fold(f8::zero(), |acc, x: &f8| acc + *x);
                    if reduction == Reduction::MEAN {
                        s = s / t_len.as_();
                    }
                }
                TensorImpl::from_op(vec![s], &[1], vec![t.clone()], Op::MSE(t_len), t.device())
            }
            delta_tensor::float16 => {
                let a = t.data();
                let mut s = f16::zero();
                if let Some(reduction) = self.reduction {
                    s = a.iter().fold(f16::zero(), |acc, x: &f16| acc + *x);
                    if reduction == Reduction::MEAN {
                        s = s / f16::from_f64(t_len as f64);
                    }
                }
                TensorImpl::from_op(vec![s], &[1], vec![t.clone()], Op::MSE(t_len), t.device())
            }
            delta_tensor::bfloat16 => {
                let a = t.data();
                let mut s = bf16::zero();
                if let Some(reduction) = self.reduction {
                    s = a.iter().fold(bf16::zero(), |acc, x: &bf16| acc + *x);
                    if reduction == Reduction::MEAN {
                        s = s / bf16::from_f64(t_len as f64);
                    }
                }
                TensorImpl::from_op(vec![s], &[1], vec![t.clone()], Op::MSE(t_len), t.device())
            }
            delta_tensor::float32 => {
                let a = t.data();
                let mut s = f32::zero();
                if let Some(reduction) = self.reduction {
                    s = a.iter().fold(f32::zero(), |acc, x: &f32| acc + *x);
                    if reduction == Reduction::MEAN {
                        s = s / t_len as f32;
                    }
                }
                TensorImpl::from_op(vec![s], &[1], vec![t.clone()], Op::MSE(t_len), t.device())
            }
            delta_tensor::float64 => {
                let a = t.data();
                let mut s = f64::zero();
                if let Some(reduction) = self.reduction {
                    s = a.iter().fold(f64::zero(), |acc, x: &f64| acc + *x);
                    if reduction == Reduction::MEAN {
                        s = s / t_len as f64;
                    }
                }
                TensorImpl::from_op(vec![s], &[1], vec![t.clone()], Op::MSE(t_len), t.device())
            }
            delta_tensor::int8 => todo!(),
            delta_tensor::int16 => todo!(),
            delta_tensor::int32 => todo!(),
            delta_tensor::int64 => todo!(),
            delta_tensor::bool => todo!(),
        };
        Tensor::new(inner)
    }
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new(Some(Reduction::MEAN))
    }
}

#[derive(Clone)]
pub struct CrossEntropyLoss {
    reduction: Option<Reduction>,
}

impl CrossEntropyLoss {
    pub fn new(reduction: Option<Reduction>) -> Self {
        Self {
            reduction: reduction,
        }
    }

    pub fn measure<T: FloatTensorRepr>(&self, a: Tensor, b: Tensor) -> Tensor {
        todo!()
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new(Some(Reduction::MEAN))
    }
}
