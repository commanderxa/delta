pub mod criterions;
pub mod functional;
pub mod linear;
pub mod module;

// define short paths
// layers
pub use linear::Linear;
// criterions
pub use criterions::MSELoss;
// module
pub use module::Module;
