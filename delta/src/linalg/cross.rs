use crate::{Tensor, TensorImpl, op::Op, promote_tensors, tensor::repr::TensorRepr};

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

    let device = crate::check_device!(&a, &b);

    let mut a = a;
    let mut b = b;
    promote_tensors!(&mut a, &mut b);

    let mut a = a;
    let mut b = b;
    // check whether to expand any of variables
    if a.shape() != b.shape() {
        // if `a` tensor is bigger => expand `b`
        // else expand `a`
        if a.length() > b.length() {
            b = b.expand(&a.shape());
        } else {
            a = a.expand(&b.shape());
        }
    }

    let inner = match a.dtype() {
        crate::DType::Float8 => todo!(),
        crate::DType::Float16 => todo!(),
        crate::DType::BFloat16 => todo!(),
        crate::DType::Float32 => {
            // get data of the Tensors
            let _a = a.data::<f32>();
            let _b = b.data::<f32>();
            // result init
            let mut result = vec![<f32 as TensorRepr>::zero(); a.length()];
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
            TensorImpl::from_op(result, &shape, vec![a, b], Op::Cross, device)
        }
        crate::DType::Float64 => todo!(),
        crate::DType::Int8 => todo!(),
        crate::DType::Int16 => todo!(),
        crate::DType::Int32 => todo!(),
        crate::DType::Int64 => todo!(),
        crate::DType::Bool => todo!(),
    };
    Tensor::new(inner)
}
