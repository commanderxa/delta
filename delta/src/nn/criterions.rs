use crate::{
    Tensor, TensorImpl,
    op::Op,
    tensor::repr::{FloatTensorRepr, TensorRepr},
};

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

    pub fn measure<T: FloatTensorRepr>(&self, a: Tensor, b: Tensor) -> Tensor {
        let t = (a - b).pow(2) * 0.5;
        let a = t.data();
        let t_len = t.length();
        let mut s = <T as TensorRepr>::zero();
        if let Some(reduction) = self.reduction {
            s = a.iter().fold(<T as TensorRepr>::zero(), |acc, x| acc + *x);
            if reduction == Reduction::MEAN {
                s = s / T::from(t_len).unwrap();
            }
        }
        let inner = TensorImpl::from_op(vec![s], &[1], vec![t.clone()], Op::MSE(t_len), t.device());
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
