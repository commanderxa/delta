use delta::linalg;

fn main() {
    let a = delta::randn(&[2, 3, 2]);
    let b = delta::ones(&[2, 4]);
    println!("A:\n{}\n", a);
    println!("B:\n{}\n\n", b);
    let c = linalg::matmul(a, b);
    println!("C = A @ B\n{}", c);
}
