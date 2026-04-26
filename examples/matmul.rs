use athena::{linalg, Tensor};

fn main() {
    let a = Tensor::randn(&[2, 3, 2]);
    let b = Tensor::ones(&[2, 4]);
    println!("A:\n{}\n", a);
    println!("B:\n{}\n\n", b);
    let c = linalg::matmul(a, b);
    println!("C = A @ B\n{}", c);
}
