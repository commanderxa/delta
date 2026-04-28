use std::collections::HashMap;

use crate::{Tensor, ivalue::IValue};

/// # `Module` Trait
///
/// Trait that defines specific behavior for modules that work with tensors.
///
/// `Module` includes:
/// - `module_name` - returns the name of module
/// - `forward` - performs inference in the module (forward propagation)
/// - `parameters` - returns all parameters that this module contains
pub trait Module {
    /// Returns the name of module
    fn module_name(&self) -> String;

    /// Returns the parameters of module
    fn parameters(&self) -> Vec<Tensor>;

    /// `forward` - performs inference in the module (forward propagation)
    fn forward(&self, args: Vec<IValue>, kwargs: HashMap<String, IValue>) -> IValue;
}
