#[cfg(test)]
mod tests {
    use athena::{
        Tensor,
        module::{Forward, Module},
        nn::{self, Linear, functional as F},
        optim::{Optim, SGD},
        tensor,
    };

    #[test]
    fn zero_grad() {
        let a = Tensor::tensor(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
        let b = Tensor::ones(&[2, 3]);
        let optim = SGD::new(vec![a.clone(), b.clone()], 1e-3);
        let c = a.clone() * b.clone();
        c.backward();
        assert_ne!(a.grad().unwrap().iter().sum::<f64>(), 0.0);
        assert_ne!(b.grad().unwrap().iter().sum::<f64>(), 0.0);
        optim.step();
        optim.zero_grad();
        assert_eq!(a.grad().unwrap().iter().sum::<f64>(), 0.0);
        assert_eq!(b.grad().unwrap().iter().sum::<f64>(), 0.0);
    }

    #[test]
    fn optim_step() {
        let mlp = MLP::new([4, 1]);
        let optim = SGD::new(mlp.parameters(), 1e-1);
        optim.zero_grad();

        let x = Tensor::randn(&[10, 4]);
        let criterion = nn::MSELoss::default();

        let mut out = mlp.forward(x.clone());
        out = out.squeeze(&[]);
        let loss = criterion.measure(
            out.clone(),
            tensor::Tensor::tensor(&[1., 0., 1., 0., 1., 0., 1., 1., 0., 1.], &[10]),
        );

        loss.backward();

        let old_data = mlp
            .parameters()
            .clone()
            .iter()
            .map(|x| x.item())
            .collect::<Vec<Vec<f64>>>();

        optim.step();

        let new_data = mlp
            .parameters()
            .clone()
            .iter()
            .map(|x| x.item())
            .collect::<Vec<Vec<f64>>>();

        for i in 0..old_data.len() {
            println!("{:?}", old_data[i]);
            println!("{:?}\n", new_data[i]);
            assert_ne!(old_data[i], new_data[i])
        }
    }

    struct MLP {
        linear1: Linear,
    }

    impl MLP {
        pub fn new(features: [usize; 2]) -> Self {
            Self {
                linear1: Linear::new(features[0], features[1], true),
            }
        }
    }

    impl Module for MLP {
        fn module_name(&self) -> String {
            "MLP".to_owned()
        }

        fn parameters(&self) -> Vec<Tensor> {
            let parameters = self.linear1.parameters();
            parameters
        }
    }

    impl Forward for MLP {
        fn forward(&self, x: Tensor) -> Tensor {
            let x = self.linear1.forward(x);
            F::sigmoid(x)
        }
    }
}
