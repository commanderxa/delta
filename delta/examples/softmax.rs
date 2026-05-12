use delta::nn::functional as F;

fn main() {
    let x = delta::tensor!([[[0.24, 0.1, 0.5], [0.8, 1.2, 2.2]]]).cast(delta::float32);
    println!("IN:\n{x}");
    let x = F::softmax(x, 2);
    println!("\nSOFTMAX:\n{x}");
    let x = x.squeeze(&[]);
    let x = x.sum(Some(1), false);
    println!("\nSUM:\n{x}");
}
