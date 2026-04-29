use std::collections::HashMap;

use crate::{
    Tensor,
    ivalue::IValue,
    linalg,
    nn::{Module, Parameter},
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
// #[derive(Module, Clone)]
pub struct Linear {
    // #[param]
    pub weights: Parameter,
}

impl Linear {
    pub fn new(mut in_features: usize, out_features: usize, bias: bool) -> Self {
        if bias {
            in_features += 1;
        }
        let _weights = Parameter(crate::randn(&[in_features, out_features]));
        Self { weights: _weights }
    }
}

impl Module for Linear {
    fn module_name(&self) -> String {
        "Linear".to_owned()
    }

    fn parameters(&self) -> Vec<Parameter> {
        let parameters = vec![self.weights.clone()];
        parameters
    }

    fn forward(&self, args: Vec<IValue>, _kwargs: HashMap<String, IValue>) -> IValue {
        let x = match &args[0] {
            IValue::Tensor(t) => t.clone(),
            _ => panic!("Linear expects a Tensor as first argument"),
        };
        let weights = self.weights.clone();
        let mut ones_shape = x.shape();
        let _ = ones_shape.pop();
        ones_shape.push(1);
        let x = Tensor::cat(&[x, crate::ones(&ones_shape)], 1);
        let x = linalg::matmul(x, weights.0);
        IValue::Tensor(x)
    }
}
