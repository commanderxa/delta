use athena::{
    module::{Forward, Module},
    nn::{functional as F, Linear, MSELoss},
    optim::{Optim, SGD},
    Tensor,
};

fn main() {
    let epochs = 10;
    let criterion = MSELoss::new(None);
    let mlp = MLP::new([2, 16, 1]);
    let optim = SGD::new(mlp.parameters(), 3e-3);

    let data = vec![
        Tensor::tensor(&[9., 3.], &[2]),
        Tensor::tensor(&[2., 3.], &[2]),
        Tensor::tensor(&[3., 7.], &[2]),
        Tensor::tensor(&[8., 6.], &[2]),
        Tensor::tensor(&[4., 4.], &[2]),
        Tensor::tensor(&[4., 1.], &[2]),
        Tensor::tensor(&[5., 2.], &[2]),
        Tensor::tensor(&[2., 5.], &[2]),
        Tensor::tensor(&[5., 6.], &[2]),
        Tensor::tensor(&[7., 2.], &[2]),
        Tensor::tensor(&[9., 1.], &[2]),
    ];
    let targets = vec![
        Tensor::tensor(&[0.], &[1]),
        Tensor::tensor(&[1.], &[1]),
        Tensor::tensor(&[1.], &[1]),
        Tensor::tensor(&[1.], &[1]),
        Tensor::tensor(&[1.], &[1]),
        Tensor::tensor(&[0.], &[1]),
        Tensor::tensor(&[0.], &[1]),
        Tensor::tensor(&[1.], &[1]),
        Tensor::tensor(&[1.], &[1]),
        Tensor::tensor(&[0.], &[1]),
        Tensor::tensor(&[0.], &[1]),
    ];

    let test_data = vec![
        Tensor::tensor(&[6., 6.], &[2]),
        Tensor::tensor(&[8., 2.], &[2]),
    ];
    let test_targets = vec![Tensor::tensor(&[1.], &[1]), Tensor::tensor(&[0.], &[1])];

    for epoch in 1..=epochs {
        let mut losses = 0.0;
        for (x, y) in data.iter().zip(targets.clone()) {
            optim.zero_grad();

            let x = x.clone().unsqueeze(0);
            let y = y.clone();
            let out = mlp.forward(x).squeeze(&[0]);
            let loss = criterion.measure(out, y);

            loss.backward();
            optim.step();

            losses += loss.item()[0];
        }
        println!("Epoch: {epoch}/{epochs} | loss: {:.10}", losses);
    }

    for (x, y) in test_data.iter().zip(test_targets.clone()) {
        let x = x.clone().unsqueeze(0);
        let out = mlp.forward(x.clone()).squeeze(&[0]);

        let loss = criterion.measure(out.clone(), y.clone());
        println!(
            "\nDATA: {0}\nTARGETS: {1}\nOUT:  {2}\nLOSS: {3:?}",
            x,
            y.item()[0],
            out.item()[0],
            loss.item()[0]
        );
        loss.backward();
    }
}

struct MLP {
    linear: Linear,
    linear1: Linear,
}

impl MLP {
    pub fn new(features: [usize; 3]) -> Self {
        Self {
            linear: Linear::new(features[0], features[1]),
            linear1: Linear::new(features[1], features[2]),
        }
    }
}

impl Module for MLP {
    fn module_name(&self) -> String {
        "MLP".to_owned()
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut parameters = self.linear.parameters();
        parameters.append(&mut self.linear1.parameters());
        parameters
    }
}

impl Forward for MLP {
    fn forward(&self, x: Tensor) -> Tensor {
        let x = self.linear.forward(x);
        let x = F::relu(x);
        let x = self.linear1.forward(x);
        F::sigmoid(x)
    }
}
