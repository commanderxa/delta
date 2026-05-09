pub mod lr_scheduler;
pub mod sgd;

// Short paths for algorithms
pub use sgd::SGD;

use crate::tensor::element::TensorFloat;

/// Behavior of optimizers
pub trait Optim<T: TensorFloat> {
    /// Updates the parameters according to the gradients.
    fn step(&self);

    /// Sets gradients to zero.
    fn zero_grad(&self);

    /// changes the learning rate by gamma
    fn change_lr(&mut self, gamma: T);
}
