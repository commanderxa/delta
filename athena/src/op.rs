use crate::Tensor;

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
