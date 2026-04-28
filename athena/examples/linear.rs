use athena::{
    Tensor, ivalue,
    nn::{Linear, Module, functional as F},
};

fn main() {
    let linear = Linear::new(20, 10, true);
    let x = Tensor::randn(&[2, 20]);
    println!("Weights: {}", linear.weights);
    println!("IN:\n{x}");
    let (args, kwargs) = ivalue![[x]];
    let out = linear.forward(args, kwargs).unwrap_tensor();
    println!("OUT:\n{out}");
    let out = F::sigmoid(out);
    println!("OUT:\n{out}");
}
