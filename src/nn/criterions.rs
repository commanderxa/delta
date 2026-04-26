use crate::{Tensor, op::Op, tensor_data::TensorData};

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
        let t = (a - b).pow(2) * 0.5 as f64;
        let a = t.item();
        let t_len = t.length() as f64;
        let mut s = 0.0;
        if let Some(reduction) = self.reduction {
            s = a.iter().sum::<f64>();
            if reduction == Reduction::MEAN {
                s /= t_len;
            }
        }
        let inner = TensorData::from_op(vec![s], vec![t], Op::MSE(t_len as usize));
        Tensor::new(inner, &[1])
    }
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new(Some(Reduction::MEAN))
    }
}
