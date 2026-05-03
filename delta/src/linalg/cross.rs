use crate::{Tensor, op::Op, tensor_impl::TensorImpl};

/// Cross Product
///
/// Accepts:
/// * a: `Tensor`
/// * b: `Tensor`
///
/// Performs the cross product of 3-dimensional vectors.
pub fn cross(a: Tensor, b: Tensor) -> Tensor {
    // check the dimensions
    assert_eq!(
        a.shape().len(),
        b.shape().len(),
        "Shape length of Tensor `b` does not much shape length of Tensor `a`"
    );
    assert_eq!(
        a.shape()[a.shape().len() - 1],
        3,
        "Last dimension of Tensor `a` does not equal 3."
    );
    assert_eq!(
        b.shape()[b.shape().len() - 1],
        3,
        "Last dimension of Tensor `b` does not equal 3."
    );

    let device = check_device!(&a, &b);

    let mut a = a;
    let mut b = b;
    // check whether to expand any of variables
    if a.shape != b.shape {
        // if `a` tensor is bigger => expand `b`
        // else expand `a`
        if a.length() > b.length() {
            b = b.expand(&a.shape);
        } else {
            a = a.expand(&b.shape);
        }
    }

    // get data of the Tensors
    let _a = a.data();
    let _b = b.data();
    // result init
    let mut result = vec![0.0; a.length()];
    let mut i: usize = 0;
    while i < (a.length() - 2) {
        result[i] = (_a[i + 1] * _b[i + 2]) - (_a[i + 2] * _b[i + 1]);
        result[i + 1] = (_a[i + 2] * _b[i]) - (_a[i] * _b[i + 2]);
        result[i + 2] = (_a[i] * _b[i + 1]) - (_a[i + 1] * _b[i]);
        i += 3;
    }
    let shape = a.shape();
    // computation
    // construct a Tensor
    let inner = TensorImpl::from_op(result, vec![a, b], Op::Cross, device);
    Tensor::new(inner, &shape)
}
