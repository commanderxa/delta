use crate::{Tensor, TensorImpl, op::Op, tensor::{cast::Cast, element::{TensorElement, TensorFloat, TensorNum}}};

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

    pub fn measure<T: TensorNum + Cast<U>, U: TensorFloat>(&self, a: Tensor<T>, b: Tensor<T>) -> Tensor<U> {
        let t = (a - b).cast::<U>().pow(2) * U::from(0.5).unwrap();
        let a = t.data();
        let t_len = t.length();
        let u_t_len = U::from(t_len).unwrap();
        let mut s = <U as TensorElement>::zero();
        if let Some(reduction) = self.reduction {
            s = a.iter().fold(<U as TensorElement>::zero(), |acc, x| acc + *x);
            if reduction == Reduction::MEAN {
                s = s / u_t_len;
            }
        }
        let inner = TensorImpl::from_op(
            vec![s],
            &[1],
            vec![t.clone()],
            Op::MSE(t_len),
            t.device,
        );
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

    pub fn measure<T: TensorNum>(&self, a: Tensor<T>, b: Tensor<T>) -> Tensor<T> {
        todo!()
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new(Some(Reduction::MEAN))
    }
}
