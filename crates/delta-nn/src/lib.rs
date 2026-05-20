pub mod criterions;
pub mod functional;
pub mod linear;
pub mod module;
pub mod parameter;

// define short paths
// layers
pub use linear::Linear;
// criterions
pub use criterions::MSELoss;
// module
pub use module::Module;
// parameter
pub use parameter::Parameter;
