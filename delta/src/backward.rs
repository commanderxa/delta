use crate::{Tensor, linalg, op::Op, storage::Storage, tensor_impl::TensorImpl};

/// Backward trait for backpropagation operation.
pub trait Backward {
    fn backward(&self, tensor: &Tensor);
}

/// `Backward` trait implemeted for `Op` where each operation has it's own
/// backpropagation operation.
impl Backward for Op {
    fn backward(&self, tensor: &Tensor) {
        match self {
            // Addition backward
            // 1.0 * grad for both previous tensors
            Op::Add => {
                let t = tensor.inner.borrow();
                let grad = t.grad.clone().unwrap().to_vec();
                t._prev[0].add_to_grad(grad.clone());
                t._prev[1].add_to_grad(grad);
            }

            // Addition backward
            // 1.0 * grad for both previous tensors
            Op::Sub => {
                let t = tensor.inner.borrow();
                let grad = t.grad.clone().unwrap().to_vec();
                t._prev[0].add_to_grad(grad.clone());
                t._prev[1].add_to_grad(grad);
            }

            // Multiplication backward
            // a * b
            // da = b * grad
            // db = a * grad
            Op::Mul => {
                let t = tensor.inner.borrow();
                let grad = t.grad.clone().unwrap().to_vec();
                let l_item = t._prev[0].data();
                let r_item = t._prev[1].data();
                t._prev[0].add_to_grad(
                    r_item
                        .iter()
                        .zip(grad.clone())
                        .map(|(a, b)| a * b)
                        .collect(),
                );
                t._prev[1].add_to_grad(l_item.iter().zip(grad).map(|(a, b)| a * b).collect());
            }

            Op::Sum { dim, keepdim } => {
                let t = tensor.inner.borrow();
                let grad = t.grad.clone().unwrap().to_vec();
                let prev = &t._prev[0];
                let prev_shape = prev.shape.clone();

                let back = match dim {
                    None => {
                        vec![grad[0]; prev.length()]
                    }
                    Some(dim) => {
                        let mut grad_tensor =
                            crate::tensor(&grad, &tensor.shape).requires_grad(false);

                        if !*keepdim {
                            let mut shape = prev_shape.clone();
                            shape[*dim] = 1;
                            grad_tensor = grad_tensor.reshape(&shape);
                        }

                        grad_tensor.expand(&prev_shape).data()
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
                let grad = t.grad.clone().unwrap().to_vec();
                let prev = &t._prev[0];
                let prev_shape = prev.shape.clone();

                let mut back = match dim {
                    None => {
                        vec![grad[0]; prev.length()]
                    }
                    Some(dim) => {
                        let mut grad_tensor =
                            crate::tensor(&grad, &tensor.shape).requires_grad(false);

                        if !*keepdim {
                            let mut shape = prev_shape.clone();
                            shape[*dim] = 1;
                            grad_tensor = grad_tensor.reshape(&shape);
                        }

                        grad_tensor.expand(&prev_shape).data()
                    }
                };

                let scale = 1.0 / *count as f64;
                for g in back.iter_mut() {
                    *g *= scale;
                }

                prev.add_to_grad(back);
            }

            // Power backward
            // d(x^n)/dx * grad = n * x^(n-1) * grad
            Op::Pow(n) => {
                let t = tensor.inner.borrow();
                let n = *n;
                let grad = t.grad.clone().unwrap().to_vec();
                t._prev[0].add_to_grad(
                    t.data
                        .iter()
                        .zip(grad)
                        .map(|(x, g)| n as f64 * x.powi(n - 1) * g)
                        .collect(),
                );
            }

            // Exponent backward
            // d(e^x)/dx = e^x
            Op::Exp(_t) => {
                let t = tensor.inner.borrow();
                let grad = t.grad.clone().unwrap().to_vec();
                t._prev[0].add_to_grad(
                    _t.exp()
                        .data()
                        .iter()
                        .zip(grad)
                        .map(|(a, b)| a * b)
                        .collect(),
                );
            }

            // Matrix Multiplication backward
            // da = dc @ b.T
            // db = a.T @ dc
            Op::MatMul => {
                let t = tensor.inner.borrow();
                let grad = t.grad.clone().unwrap().to_vec();
                let d_c =
                    Tensor::new(TensorImpl::from_f64(grad), &tensor.shape).requires_grad(false);
                let a = t._prev[0].t();
                let b = t._prev[1].t();
                t._prev[0].add_to_grad(linalg::matmul(d_c.clone(), b).data());
                t._prev[1].add_to_grad(linalg::matmul(a, d_c).data());
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
                let out_grad = t.grad.as_ref().unwrap().to_vec();
                let grad: Vec<f64> = input_data
                    .into_iter()
                    .zip(out_grad.into_iter())
                    .map(|(x, go)| if x > 0.0 { go } else { 0.0 })
                    .collect();

                prev.grad = Some(Storage::new(grad, prev.data.device()));
            }

            // Sigmoid backward
            // sigmoid function:        1 / (1 + exp(-x))
            // dx(sigmoid) function:    exp(-x) / (1 + exp(-x))^2
            Op::Sigmoid(x) => {
                let t = tensor.inner.borrow();
                let e_x = (-x.clone()).exp();
                let res = e_x.clone() / (e_x + 1.0 as f64).pow(2);
                let grad = t.grad.clone().unwrap();
                let dx = grad.iter().zip(res.data()).map(|(a, b)| a * b).collect();
                t._prev[0].add_to_grad(dx);
            }

            Op::Softmax(x, _) => {
                let t = tensor.inner.borrow();
                let n = x.length();
                let s = x.data();
                let mut jacobian = vec![0.0; n * n];
                for i in 0..n {
                    for j in 0..n {
                        if i == j {
                            jacobian[i * n + j] = s[i] * (1.0 - s[i]);
                        } else {
                            jacobian[i * n + j] = -s[i] * s[j];
                        }
                    }
                }
                let a = crate::tensor(&jacobian, &[n, n]).t();
                t._prev[0].add_to_grad(a.data());
            }

            Op::MSE(n) => {
                let t = tensor.inner.borrow();
                let t_prev = t._prev[0].inner.borrow();
                let t_sub = t_prev._prev[0].inner.borrow();
                let grad = (t_sub._prev[0].clone() - t_sub._prev[1].clone()) * n.to_owned();
                drop(t_sub);
                drop(t_prev);
                t._prev[0].add_to_grad(grad.data());
            }
        }
    }
}
