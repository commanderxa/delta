pub mod criterions;
pub mod functional;
pub mod linear;

// define short paths
// layers
pub use linear::Linear;
// criterions
pub use criterions::MSELoss;
