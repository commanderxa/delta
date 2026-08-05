use std::fmt::Debug;

use crate::{DType, Tensor, promote_tensors};
use delta_core::cast::Cast;

pub trait OpBase {
    fn forward(&self, input: &[Tensor]) -> Tensor;
    fn backward(&self, inputs: &[Tensor]);
}

fn binary_ops_prep(a: Tensor, b: Tensor) -> (Tensor, Tensor) {
    let mut a = a;
    let mut b = b;
    // check whether to expand any of variables
    if a.shape() != b.shape() {
        // if `a` tensor is bigger => expand `b`
        // else expand `a`
        if a.length() > b.length() {
            b = b.expand(&a.shape());
        } else {
            a = a.expand(&b.shape());
        }
    }
    promote_tensors!(&mut a, &mut b);
    return (a, b);
}

#[derive(Clone, Debug)]
pub struct Add;

impl Add {
    fn new()  -> Self {
        Self
    }
}

impl OpBase for Add {
    fn forward(&self, input: &[Tensor]) -> Tensor {
        let a = input[0].clone();
        let b = input[1].clone();
        let (a, b) = binary_ops_prep(a, b);
        let device = a.device();
        match a.dtype() {
            DType::Float8 => todo!(),
            DType::Float16 => todo!(),
            DType::BFloat16 => todo!(),
            DType::Float32 => {
                let mut mask = vec![0; a.shape().len()];
                let mut data = vec![Cast::cast(0.); a.length()];
                // iterate over storage data
                for d in data.iter_mut() {
                    // compute index of past position of data
                    let a_i = a.data::<f32>()[a
                        .stride()
                        .iter()
                        .zip(&mask)
                        .map(|(a, b)| a * b)
                        .sum::<usize>()];
                    let b_i: f32 = b.data()[b
                        .stride()
                        .iter()
                        .zip(&mask)
                        .map(|(a, b)| a * b)
                        .sum::<usize>()];
                    // write the result for particular element based on the operation
                    *d = a_i + b_i;
                    for j in (0..a.shape().len()).rev() {
                        if a.shape()[j] - 1 == mask[j] {
                            continue;
                        }
                        mask[j] += 1;
                        for k in ((j + 1)..a.shape().len()).rev() {
                            mask[k] = 0;
                        }
                        break;
                    }
                }
                Tensor::from_op(data, &a.shape(), vec![a, b], Op::Add, device)
            }
            DType::Float64 => todo!(),
            DType::Int8 => todo!(),
            DType::Int16 => todo!(),
            DType::Int32 => todo!(),
            DType::Int64 => todo!(),
            DType::Bool => todo!(),
        }
    }

    fn backward(&self, inputs: &[Tensor]) {
        let t = inputs[0].inner.borrow();
        let grad = t.grad.clone().unwrap();
        t.prev[0].add_to_grad(grad.clone());
        t.prev[1].add_to_grad(grad);
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug, PartialEq)]
/// Operations that are available to apply to `Value`.
pub enum Op {
    Add,
    Sub,
    Mul,
    Sum {
        dim: Option<usize>,
        keepdim: bool,
    },
    Mean {
        dim: Option<usize>,
        keepdim: bool,
        count: usize,
    },
    Pow(i32),
    Exp(Tensor),
    MatMul,
    Cross,
    // Squeeze,
    // Unsqueeze,
    ReLU,
    Sigmoid(Tensor),
    Softmax(Tensor, usize),
    MSE(usize),
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::Add => write!(f, "Add"),
            Op::Sub => write!(f, "Sub"),
            Op::Mul => write!(f, "Mul"),
            Op::Sum { dim: _, keepdim: _ } => write!(f, "Sum"),
            Op::Mean {
                dim: _,
                keepdim: _,
                count: _,
            } => write!(f, "Mean"),
            Op::Pow(n) => write!(f, "Pow({n})"),
            Op::Exp(_) => write!(f, "Exp"),
            Op::MatMul => write!(f, "MatMul"),
            Op::Cross => write!(f, "Cross"),
            Op::ReLU => write!(f, "ReLU"),
            Op::Sigmoid(n) => write!(f, "Sigmoid({n})"),
            Op::Softmax(n, dim) => write!(f, "Softmax({n},{dim})"),
            Op::MSE(n) => write!(f, "MSE({n})"),
        }
    }
}
