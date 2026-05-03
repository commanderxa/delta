#[cfg(feature = "cuda")]
use cudarc::cublas::safe::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
#[cfg(feature = "cuda")]
use cudarc::cublas::sys::cublasOperation_t;

use crate::device::Device;
use crate::{Tensor, op::Op, tensor_impl::TensorImpl};

/// Matrix multiplication
///
/// Accepts:
/// * a: `Tensor`
/// * b: `Tensor`
///
/// The inner dimensions of the matrices must be the same.
pub fn matmul(a: Tensor, b: Tensor) -> Tensor {
    // shapes of the tensors
    let mut a_shape: Vec<usize> = a.shape();
    let b_shape: Vec<usize> = b.shape();
    // check wether the operation is able to be proceeded
    assert!(
        a_shape.len() > 1,
        "The shape of the tensor must not be scalar."
    );
    assert!(
        b_shape.len() > 1,
        "The shape of the tensor must not be scalar."
    );
    assert_eq!(
        a_shape.last().unwrap(),
        b_shape.first().unwrap(),
        "The shapes of the tensors must have the same inner dimension -> (M x K) @ (K x N), but you have tensors A: {:?} and B: {:?}",
        format!("({a_shape:?})").replace('[', "").replace(']', ""),
        format!("({b_shape:?})").replace('[', "").replace(']', ""),
    );

    let device = check_device!(&a, &b);

    // get batch dimensions if they exist
    let mut batches: Vec<usize> = vec![];
    for i in 0..(a_shape.len() - 2) {
        batches.push(a_shape[i]);
    }
    // remove batch dimensions from the A tensor shape
    a_shape.drain(0..batches.len());
    let batch_prod = batches.iter().product::<usize>();

    // literal notations for matrix dimensions for the conviniency and comprehension
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    // new shape
    let new_shape: Vec<usize> = batches.into_iter().chain(vec![m, n]).collect();

    device_op!(device,
        cpu => matmul_cpu(a, b, m, k, n, batch_prod, &new_shape),
        cuda => {
            let blas = CudaBlas::new(crate::cuda::current_stream()).unwrap();
            matmul_batched_cuda(&blas, a, b, batch_prod, m, k, n, &new_shape)
        }
    )
}

fn matmul_cpu(
    a: Tensor,
    b: Tensor,
    m: usize,
    k: usize,
    n: usize,
    batch_size: usize,
    new_shape: &[usize],
) -> Tensor {
    // data of the tensors, the tensor b is transposed
    let a_data = a.data();
    let b_data = b.t().data();

    let mut result = vec![0.0; batch_size * m * n];

    // iterate over the batch dimensions
    // `_b` is a batch dimension
    for _b in 0..batch_size {
        // iterate over the result tensor, it zips the slices of the left and
        // right tensors then it multiplies the two zipped values and returns
        // the slice back, after it sums the vector to obtain the value
        for i in 0..m {
            for j in 0..n {
                let b_data = &b_data[(j * k)..(j * k + k)];
                result[_b * m * n + i * n + j] = a_data
                    [(_b * m * k + i * k)..(_b * m * k + i * k + k)]
                    .iter()
                    .zip(b_data)
                    .map(|(&a, &b)| a * b)
                    .sum::<f64>();
            }
        }
    }
    // add batch dimensions to the new shape
    let inner = TensorImpl::from_op(result, vec![a, b], Op::MatMul, Device::CPU);
    Tensor::new(inner, &new_shape)
}

#[cfg(feature = "cuda")]
fn matmul_batched_cuda(
    blas: &CudaBlas,
    a: Tensor,
    b: Tensor,
    batch_size: usize,
    m: usize,
    k: usize,
    n: usize,
    new_shape: &[usize],
) -> Tensor {
    let mut a = a;
    let mut b = b;

    if &a.shape().len() > &b.shape().len() {
        let mut b_shape = new_shape[0..new_shape.len() - 2].to_vec();
        b_shape.extend(&b.shape());
        b = b.expand(&b_shape).contiguous();

    } else if &a.shape().len() < &b.shape().len() {
        let mut a_shape = new_shape[0..new_shape.len() - 2].to_vec();
        a_shape.extend(&a.shape());
        a = a.expand(&a_shape).contiguous();
    }

    let a_data = a.get_data_storage();
    let a_data = a_data.as_cuda();
    let b_data = b.get_data_storage();
    let b_data = b_data.as_cuda();

    let stream = crate::cuda::current_stream();
    let mut c = stream.alloc_zeros::<f64>(batch_size * m * n).unwrap();

    let cfg = StridedBatchedConfig {
        gemm: GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0f64,
            lda: n as i32,
            ldb: k as i32,
            beta: 0.0f64,
            ldc: n as i32,
        },
        batch_size: batch_size as i32,
        stride_a: (k * n) as i64,
        stride_b: (m * k) as i64,
        stride_c: (m * n) as i64,
    };

    unsafe {
        blas.gemm_strided_batched(cfg, b_data, a_data, &mut c)
            .unwrap();
    }

    let inner = TensorImpl::from_cuda(c, vec![a, b], Some(Op::MatMul));
    Tensor::new(inner, &new_shape)
}
