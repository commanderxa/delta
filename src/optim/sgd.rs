use crate::Tensor;

use super::Optim;

/// # SGD algorithm
///
/// Stochastic Gradient Descent for updating model parameters.
///
/// It has:
/// - parameters of the model
/// - learning rate
#[derive(Clone)]
pub struct SGD {
    parameters: Vec<Tensor>,
    lr: f64,
    maximize: bool,
}

impl SGD {
    pub fn new(parameters: Vec<Tensor>, lr: f64) -> Self {
        Self {
            parameters,
            lr,
            maximize: false,
        }
    }

    pub fn maximize(&mut self) {
        self.maximize = true;
    }

    pub fn minimize(&mut self) {
        self.maximize = false;
    }
}

impl Optim for SGD {
    fn step(&self) {
        for i in 0..self.parameters.len() {
            let data: Vec<f64> = self.parameters[i]
                .grad()
                .unwrap()
                .iter()
                .zip(self.parameters[i].item())
                .map(|(a, b)| {
                    // w_i = w_(i-1) - lr * grad
                    // b = w(i-1)
                    // a = grad
                    b - self.lr * a
                })
                .collect();
            self.parameters[i].set_data(data);
        }
    }

    fn zero_grad(&self) {
        for i in 0..self.parameters.len() {
            self.parameters[i].inner.borrow_mut().zero_grad();
        }
    }

    fn change_lr(&mut self, gamma: f64) {
        self.lr *= gamma
    }
}
