use delta::{
    ivalue,
    nn::{Linear, Module, functional as F},
};

fn main() {
    let linear = Linear::new::<f32>(20, 10, true);
    let x = delta::randn::<f32>(&[2, 20], delta::device::Device::CPU);
    println!("Weights: {}", linear.weights.0);
    println!("IN:\n{x}");
    let (args, kwargs) = ivalue![[x]];
    let out = linear.forward(args, kwargs).unwrap_tensor();
    println!("OUT:\n{out}");
    let out = F::sigmoid(out);
    println!("OUT:\n{out}");
}
