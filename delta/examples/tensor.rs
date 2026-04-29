use delta::{Tensor, tensor};

fn main() {
    let a = tensor!([1., 2., 3.]);
    print_info(a);

    let b = tensor!([[1, 1, 1], [2, 2, 2], [3, 3, 3]]);
    print_info(b);
}

fn print_info(t: Tensor) -> () {
    println!("Tensor");
    println!("\tdata: {:?}", t.item());
    println!("\tshape: {:?}", t.shape);
    println!("\tsimple: {}", t);
    println!("\tfull: {:?}", t);
}
