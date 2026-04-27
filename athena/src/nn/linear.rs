use crate::{
    Tensor, linalg,
    module::{Forward, Module},
};

/// # `Linear` Layer
///
/// Contains of:
/// - weights
/// - bias
///
/// Linear layer performs: `x @ W + b`,
/// where:
/// - `x` is input
/// - `W` is weights
/// - `b` is bias
#[derive(Clone)]
pub struct Linear {
    pub weights: Tensor,
}

impl Linear {
    pub fn new(mut in_features: usize, out_features: usize, bias: bool) -> Self {
        if bias {
            in_features += 1;
        }
        let _weights = Tensor::randn(&[in_features, out_features]);
        Self { weights: _weights }
    }
}

impl Module for Linear {
    fn module_name(&self) -> String {
        "Linear".to_owned()
    }

    fn parameters(&self) -> Vec<Tensor> {
        let parameters = vec![self.weights.clone()];
        parameters
    }
}

impl Forward for Linear {
    fn forward(&self, x: Tensor) -> Tensor {
        let weights = self.weights.clone();
        let mut ones_shape = x.shape();
        let _ = ones_shape.pop();
        ones_shape.push(1);
        let x = Tensor::cat(&[x, Tensor::ones(&ones_shape)], 1);
        let x = linalg::matmul(x, weights);
        x
    }
}
