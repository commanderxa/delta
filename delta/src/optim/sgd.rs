use crate::{nn::Parameter, tensor::element::TensorFloat};

use super::Optim;

/// # SGD algorithm
///
/// Stochastic Gradient Descent for updating model parameters.
///
/// It has:
/// - parameters of the model
/// - learning rate
#[derive(Clone)]
pub struct SGD<T: TensorFloat> {
    parameters: Vec<Parameter<T>>,
    lr: T,
    maximize: bool,
}

impl<T: TensorFloat> SGD<T> {
    pub fn lr(&self) -> T {
        self.lr
    }

    pub fn is_maximize(&self) -> bool {
        self.maximize
    }

    pub fn parameters(&self) -> &[Parameter<T>] {
        &self.parameters
    }

    pub fn new(parameters: Vec<Parameter<T>>, lr: T) -> Self {
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

impl<T: TensorFloat> Optim<T> for SGD<T> {
    fn step(&self) {
        for i in 0..self.parameters.len() {
            let data: Vec<T> = self.parameters[i]
                .grad()
                .unwrap()
                .data()
                .iter()
                .zip(self.parameters[i].data())
                .map(|(a, b)| {
                    // w_i = w_(i-1) - lr * grad
                    // b = w(i-1)
                    // a = grad
                    b - self.lr * *a
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

    fn change_lr(&mut self, gamma: T) {
        self.lr = self.lr * gamma
    }
}
