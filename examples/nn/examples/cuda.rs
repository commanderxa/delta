#[cfg(feature = "cuda")]
use delta::cuda;
use delta::nn::functional as F;
use delta::{
    ivalue,
    nn::{self, Module},
};

fn main() {
    #[cfg(feature = "cuda")]
    println!("Is CUDA available: {}.", cuda::is_available());
    let linear = nn::Linear::new(20, 10, true);
    let x = delta::randn(&[2, 20], delta::cuda);
    println!("Weights: {:?}", linear.weights.0);
    println!("IN:\n{x}");
    let (args, kwargs) = ivalue![[x]];
    let out = linear.forward(args, kwargs).unwrap_tensor();
    println!("OUT:\n{out}");
    let out = F::sigmoid(out);
    println!("OUT:\n{out}");
}
