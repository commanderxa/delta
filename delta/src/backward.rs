use crate::{
    Tensor, linalg,
    op::Op,
    tensor::element::{TensorElement, TensorFloat},
};

/// Backward trait for backpropagation operation.
pub trait Backward<T: TensorFloat> {
    fn backward(&self, tensor: &Tensor<T>);
}

/// `Backward` trait implemeted for `Op` where each operation has it's own
/// backpropagation operation.
impl<T: TensorFloat> Backward<T> for Op<T> {
    fn backward(&self, tensor: &Tensor<T>) {
        match self {
            // Addition backward
            // 1.0 * grad for both previous tensors
            Op::Add => {
                let t = tensor.inner.borrow();
                let grad = t.grad.clone().unwrap();
                t._prev[0].add_to_grad(grad.clone());
                t._prev[1].add_to_grad(grad);
            }

            // Addition backward
            // 1.0 * grad for both previous tensors
            Op::Sub => {
                let t = tensor.inner.borrow();
                let grad = t.grad.clone().unwrap();
                t._prev[0].add_to_grad(grad.clone());
                t._prev[1].add_to_grad(grad);
            }

            // Multiplication backward
            // a * b
            // da = b * grad
            // db = a * grad
            Op::Mul => {
                let t = tensor.inner.borrow();
                let grad = t.grad.clone().unwrap();
                let l_item = t._prev[0].clone();
                let r_item = t._prev[1].clone();
                t._prev[0].add_to_grad(r_item * grad.clone());
                t._prev[1].add_to_grad(l_item * grad);
            }

            Op::Sum { dim, keepdim } => {
                let t = tensor.inner.borrow();
                let grad = t.grad.clone().unwrap().data();
                let prev = &t._prev[0];
                let prev_shape = prev.shape();

                let back = match dim {
                    None => crate::tensor(&vec![grad[0]; prev.length()], &prev_shape),
                    Some(dim) => {
                        let mut grad_tensor =
                            crate::tensor(&grad, &tensor.shape()).requires_grad(false);

                        if !*keepdim {
                            let mut shape = prev_shape.clone();
                            shape[*dim] = 1;
                            grad_tensor = grad_tensor.reshape(&shape);
                        }

                        grad_tensor.expand(&prev_shape)
                    }
                };

                prev.add_to_grad(back);
            }

            Op::Mean {
                dim,
                keepdim,
                count,
            } => {
                let t = tensor.inner.borrow();
                let grad = t.grad.clone().unwrap().data();
                let prev = &t._prev[0];
                let prev_shape = prev.shape();

                let back = match dim {
                    None => crate::tensor(&vec![grad[0]; prev.length()], &prev_shape),
                    Some(dim) => {
                        let mut grad_tensor =
                            crate::tensor(&grad, &tensor.shape()).requires_grad(false);

                        if !*keepdim {
                            let mut shape = prev_shape.clone();
                            shape[*dim] = 1;
                            grad_tensor = grad_tensor.reshape(&shape);
                        }

                        grad_tensor.expand(&prev_shape)
                    }
                };

                let scale = <T as TensorElement>::one() / T::from(*count).unwrap();
                for g in back.data().iter_mut() {
                    *g = *g * scale;
                }

                prev.add_to_grad(back);
            }

            // Power backward
            // d(x^n)/dx * grad = n * x^(n-1) * grad
            Op::Pow(n) => {
                let t = tensor.inner.borrow();
                let n = *n;
                let grad = t.grad.clone().unwrap();
                t._prev[0].add_to_grad(tensor.pow(n - 1) * grad);
            }

            // Exponent backward
            // d(e^x)/dx = e^x
            Op::Exp(_t) => {
                let t = tensor.inner.borrow();
                let grad = t.grad.clone().unwrap();
                t._prev[0].add_to_grad(_t.exp() * grad);
            }

            // Matrix Multiplication backward
            // da = dc @ b.T
            // db = a.T @ dc
            Op::MatMul => {
                let t = tensor.inner.borrow();
                let d_c = t.grad.clone().unwrap();
                let a = t._prev[0].t();
                let b = t._prev[1].t();
                t._prev[0].add_to_grad(linalg::matmul(d_c.clone(), b));
                t._prev[1].add_to_grad(linalg::matmul(a, d_c));
            }

            // Cross Section Multiplication backward
            Op::Cross => {
                todo!();
            }

            // ReLU backward
            // d(relu)/dx = { 1 if x > 0 else 0 }
            Op::ReLU => {
                let t = tensor.inner.borrow();
                let mut prev = t._prev[0].inner.borrow_mut();
                let input_data = prev.data.to_vec();
                let out_grad = t.grad.as_ref().unwrap().data();
                let grad: Vec<T> = input_data
                    .into_iter()
                    .zip(out_grad.into_iter())
                    .map(|(x, go)| {
                        if x > <T as TensorElement>::zero() {
                            go
                        } else {
                            <T as TensorElement>::zero()
                        }
                    })
                    .collect();

                prev.grad = Some(crate::tensor(&grad, &t.shape));
            }

            // Sigmoid backward
            // sigmoid function:        1 / (1 + exp(-x))
            // dx(sigmoid) function:    exp(-x) / (1 + exp(-x))^2
            Op::Sigmoid(x) => {
                let t = tensor.inner.borrow();
                let e_x = (-x.clone()).exp();
                let res = e_x.clone() / (e_x + <T as TensorElement>::one()).pow(2);
                let grad = t.grad.clone().unwrap();
                let dx = grad * res;
                t._prev[0].add_to_grad(dx);
            }

            Op::Softmax(x, _) => {
                let t = tensor.inner.borrow();
                let n = x.length();
                let s = x.data();
                let mut jacobian = vec![<T as TensorElement>::zero(); n * n];
                for i in 0..n {
                    for j in 0..n {
                        if i == j {
                            jacobian[i * n + j] = s[i] * (<T as TensorElement>::one() - s[i]);
                        } else {
                            jacobian[i * n + j] = -s[i] * s[j];
                        }
                    }
                }
                let a = crate::tensor(&jacobian, &[n, n]).t();
                t._prev[0].add_to_grad(a);
            }

            Op::MSE(n) => {
                let t = tensor.inner.borrow();
                let t_prev = t._prev[0].inner.borrow();
                let t_sub = t_prev._prev[0].inner.borrow();
                let grad = (t_sub._prev[0].clone() - t_sub._prev[1].clone()) * T::from(*n).unwrap();
                drop(t_sub);
                drop(t_prev);
                t._prev[0].add_to_grad(grad);
            }
        }
    }
}
