use crate::Tensor;

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
}

/// # `Forward` Trait
///
/// Trait that defines specific behavior for modules that work with tensors.
/// This trait has to be implemented for struct that implement `Module`.
///
/// `forward` - performs inference in the module (forward propagation)
pub trait Forward: Module {
    /// Forward (inference) function for module
    fn forward(&self, x: Tensor) -> Tensor;
}
