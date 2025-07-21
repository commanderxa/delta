use cognius::{nn::functional as F, Tensor};

fn main() {
    let x = Tensor::tensor(&[0.24, 0.1, 0.5, 0.8, 1.2, 2.2], &[1, 2, 3]);
    println!("IN:\n{x}\n\n");
    let x = F::softmax(x, 2);
    println!("OUT:\n{x}");
}
