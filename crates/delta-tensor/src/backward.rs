use crate::{Tensor, linalg, op::Op};

use delta_core::cast::Cast;

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
                let grad = t.grad.clone().unwrap();
                t.prev[0].add_to_grad(grad.clone());
                t.prev[1].add_to_grad(grad);
            }

            // Addition backward
            // 1.0 * grad for both previous tensors
            Op::Sub => {
                let t = tensor.inner.borrow();
                let grad = t.grad.clone().unwrap();
                t.prev[0].add_to_grad(grad.clone());
                t.prev[1].add_to_grad(grad);
            }

            // Multiplication backward
            // a * b
            // da = b * grad
            // db = a * grad
            Op::Mul => {
                let t = tensor.inner.borrow();
                let grad = t.grad.clone().unwrap();
                let l_item = t.prev[0].clone();
                let r_item = t.prev[1].clone();
                t.prev[0].add_to_grad(r_item * grad.clone());
                t.prev[1].add_to_grad(l_item * grad);
            }

            Op::Sum { dim, keepdim } => {
                // let t = tensor.inner.borrow();
                // let grad = t.grad.clone().unwrap().data();
                // let prev = &t.prev[0];
                // let prev_shape = prev.shape();

                // let back = match dim {
                //     None => {
                //         crate::tensor(&vec![grad[0]; prev.length()], &prev_shape, prev.device())
                //     }
                //     Some(dim) => {
                //         let mut grad_tensor =
                //             crate::tensor(&grad, &tensor.shape(), tensor.device())
                //                 .requires_grad(false);

                //         if !*keepdim {
                //             let mut shape = prev_shape.clone();
                //             shape[*dim] = 1;
                //             grad_tensor = grad_tensor.reshape(&shape);
                //         }

                //         grad_tensor.expand(&prev_shape)
                //     }
                // };

                // prev.add_to_grad(back);
            }

            Op::Mean {
                dim,
                keepdim,
                count,
            } => {
                // let t = tensor.inner.borrow();
                // let grad = t.grad.clone().unwrap().data();
                // let prev = &t.prev[0];
                // let prev_shape = prev.shape();

                // let back = match dim {
                //     None => {
                //         crate::tensor(&vec![grad[0]; prev.length()], &prev_shape, prev.device())
                //     }
                //     Some(dim) => {
                //         let mut grad_tensor =
                //             crate::tensor(&grad, &tensor.shape(), tensor.device())
                //                 .requires_grad(false);

                //         if !*keepdim {
                //             let mut shape = prev_shape.clone();
                //             shape[*dim] = 1;
                //             grad_tensor = grad_tensor.reshape(&shape);
                //         }

                //         grad_tensor.expand(&prev_shape)
                //     }
                // };

                // let scale = Cast::cast(1.) / Cast::cast(*count as i64);
                // for g in back.data().iter_mut() {
                //     *g = *g * scale;
                // }

                // prev.add_to_grad(back);
            }

            // Power backward
            // d(x^n)/dx * grad = n * x^(n-1) * grad
            Op::Pow(n) => {
                let t = tensor.inner.borrow();
                let n = *n;
                let grad = t.grad.clone().unwrap();
                t.prev[0].add_to_grad(tensor.pow(n - 1) * grad);
            }

            // Exponent backward
            // d(e^x)/dx = e^x
            Op::Exp(_t) => {
                let t = tensor.inner.borrow();
                let grad = t.grad.clone().unwrap();
                t.prev[0].add_to_grad(_t.exp() * grad);
            }

            // Matrix Multiplication backward
            // da = dc @ b.T
            // db = a.T @ dc
            Op::MatMul => {
                let t = tensor.inner.borrow();
                let d_c = t.grad.clone().unwrap();
                let a = t.prev[0].t();
                let b = t.prev[1].t();
                t.prev[0].add_to_grad(linalg::matmul(d_c.clone(), b));
                t.prev[1].add_to_grad(linalg::matmul(a, d_c));
            }

            // Cross Section Multiplication backward
            Op::Cross => {
                todo!();
            }

            // ReLU backward
            // d(relu)/dx = { 1 if x > 0 else 0 }
            Op::ReLU => {
                // let t = tensor.inner.borrow();
                // let mut prev = t.prev[0].inner.borrow_mut();
                // let input_data = prev.data.as_cpu();
                // let out_grad = t.grad.as_ref().unwrap().data();
                // let grad: Vec<_> = input_data
                //     .into_iter()
                //     .zip(out_grad.into_iter())
                //     .map(|(x, go)| if x > Cast::cast(0) { go } else { Cast::cast(0) })
                //     .collect();

                // prev.grad = Some(crate::tensor(&grad, &t.shape, t.device()));
            }

            // Sigmoid backward
            // sigmoid function:        1 / (1 + exp(-x))
            // dx(sigmoid) function:    exp(-x) / (1 + exp(-x))^2
            Op::Sigmoid(x) => {
                // let t = tensor.inner.borrow();
                // let e_x = (-x.clone()).exp();
                // let res = e_x.clone() / (e_x + Cast::cast(1.)).pow(2);
                // let grad = t.grad.clone().unwrap();
                // let dx = grad * res;
                // t.prev[0].add_to_grad(dx);
            }

            Op::Softmax(x, _) => {
                // let t = tensor.inner.borrow();
                // let n = x.length();
                // let s = x.data();
                // let mut jacobian = vec![Cast::cast(0.); n * n];
                // for i in 0..n {
                //     for j in 0..n {
                //         if i == j {
                //             jacobian[i * n + j] = s[i] * (Cast::cast(1.) - s[i]);
                //         } else {
                //             jacobian[i * n + j] = -s[i] * s[j];
                //         }
                //     }
                // }
                // let a = crate::tensor(&jacobian, &[n, n], t.device()).t();
                // t.prev[0].add_to_grad(a);
            }

            Op::MSE(n) => {
                // let t = tensor.inner.borrow();
                // let tprev = t.prev[0].inner.borrow();
                // let t_sub = tprev.prev[0].inner.borrow();
                // let grad = (t_sub.prev[0].clone() - t_sub.prev[1].clone()) * Cast::cast(*n as i64);
                // drop(t_sub);
                // drop(tprev);
                // t.prev[0].add_to_grad(grad);
            }
        }
    }
}
