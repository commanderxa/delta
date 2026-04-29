use crate::Tensor;

/// X is the input data
pub type X = Tensor;
/// Y is the target data
pub type Y = Tensor;
/// Sample represents zipped input and target data
pub type Sample = (X, Y);
