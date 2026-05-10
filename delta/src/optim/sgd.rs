use crate::{nn::Parameter, tensor::repr::FloatTensorRepr};

use super::Optim;

/// # SGD algorithm
///
/// Stochastic Gradient Descent for updating model parameters.
///
/// It has:
/// - parameters of the model
/// - learning rate
#[derive(Clone)]
pub struct SGD<T: FloatTensorRepr> {
    parameters: Vec<Parameter>,
    lr: T,
    maximize: bool,
}

impl<T: FloatTensorRepr> SGD<T> {
    pub fn lr(&self) -> T {
        self.lr.clone()
    }

    pub fn is_maximize(&self) -> bool {
        self.maximize
    }

    pub fn parameters(&self) -> &[Parameter] {
        &self.parameters
    }

    pub fn new(parameters: Vec<Parameter>, lr: T) -> Self {
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

impl<T: FloatTensorRepr> Optim<T> for SGD<T> {
    fn step(&self) {
        for i in 0..self.parameters.len() {
            let data: Vec<T> = self.parameters[i]
                .grad()
                .unwrap()
                .data::<T>()
                .iter()
                .zip(self.parameters[i].data::<T>())
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
            self.parameters[i].inner.borrow_mut().zero_grad::<T>();
        }
    }

    fn change_lr(&mut self, gamma: T) {
        self.lr = self.lr * gamma
    }
}
