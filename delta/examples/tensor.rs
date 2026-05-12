use delta::Tensor;

fn main() {
    let a = delta::tensor!([1.0, 2.0, 3.0], delta::int32);
    print_info(a);

    let b = delta::tensor!([[1, 1, 1], [2, 2, 2], [3, 3, 3]]).cast(delta::int32);
    print_info(b);
}

fn print_info(t: Tensor) -> () {
    // let t = t.cast(delta::float32);
    println!("Tensor");
    println!("\tdtype: {:?}", t.dtype());
    println!("\tdata: {:?}", t.data::<i32>());
    println!("\tshape: {:?}", t.shape());
    println!("\tsimple: {}", t);
    println!("\tfull: {:?}", t);
}
