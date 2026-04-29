use std::collections::HashMap;

use crate::{ivalue::IValue, nn};

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

    /// `forward` - performs inference in the module (forward propagation)
    fn forward(&self, args: Vec<IValue>, kwargs: HashMap<String, IValue>) -> IValue;

    /// Returns the submodules of module
    fn submodules(&self) -> Vec<&dyn Module> {
        vec![]
    }

    /// Returns the parameters of module
    fn parameters(&self) -> Vec<nn::Parameter> {
        vec![]
    }
}
