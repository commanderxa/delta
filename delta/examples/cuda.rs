#[cfg(feature = "cuda")]
use delta::cuda;

fn main() {
    #[cfg(feature = "cuda")]
    println!("{}", cuda::is_available());
}
