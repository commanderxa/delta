use athena::{
    module::{Forward, Module},
    nn::{Linear, MSELoss},
    optim::{Optim, SGD},
    Tensor,
};

fn main() {
    let epochs = 10;
    let criterion = MSELoss::new(None);
    let mlp = MLP::new([1, 1]);
    let optim = SGD::new(mlp.parameters(), 3e-3);

    let data = vec![
        Tensor::tensor(&[1.], &[1, 1]),
        Tensor::tensor(&[2.], &[1, 1]),
        Tensor::tensor(&[3.], &[1, 1]),
        Tensor::tensor(&[4.], &[1, 1]),
        Tensor::tensor(&[5.], &[1, 1]),
        Tensor::tensor(&[6.], &[1, 1]),
        Tensor::tensor(&[7.], &[1, 1]),
        Tensor::tensor(&[8.], &[1, 1]),
        Tensor::tensor(&[9.], &[1, 1]),
        Tensor::tensor(&[10.], &[1, 1]),
    ];
    let targets = vec![
        Tensor::tensor(&[2. * 1.], &[1, 1]),
        Tensor::tensor(&[2. * 2.], &[1, 1]),
        Tensor::tensor(&[2. * 3.], &[1, 1]),
        Tensor::tensor(&[2. * 4.], &[1, 1]),
        Tensor::tensor(&[2. * 5.], &[1, 1]),
        Tensor::tensor(&[2. * 6.], &[1, 1]),
        Tensor::tensor(&[2. * 7.], &[1, 1]),
        Tensor::tensor(&[2. * 8.], &[1, 1]),
        Tensor::tensor(&[2. * 9.], &[1, 1]),
        Tensor::tensor(&[2. * 10.], &[1, 1]),
    ];

    let test_data = vec![
        Tensor::tensor(&[11.], &[1, 1]),
        Tensor::tensor(&[-2.], &[1, 1]),
    ];
    let test_targets = vec![
        Tensor::tensor(&[22.], &[1, 1]),
        Tensor::tensor(&[-4.], &[1, 1]),
    ];

    for epoch in 1..=epochs {
        let mut losses = 0.0;
        for (x, y) in data.iter().zip(targets.clone()) {
            optim.zero_grad();

            let x = x.clone();
            let y = y.clone();
            let out = mlp.forward(x.clone());
            let loss = criterion.measure(out, y);

            loss.backward();
            optim.step();

            losses += loss.item()[0];
        }
        println!(
            "Epoch: {epoch}/{epochs} | loss: {:.10}",
            (losses / (epochs as f64 * 2.))
        );
    }

    let mut training_res = vec![];
    let mut test_res = vec![];

    for (x, y) in data.iter().zip(targets.clone()) {
        let out = mlp.forward(x.clone());
        let loss = criterion.measure(out.clone(), y.clone());
        loss.backward();
        training_res.push((out.item()[0], y.item()[0]))
    }

    for (x, y) in test_data.iter().zip(test_targets.clone()) {
        let out = mlp.forward(x.clone());
        let loss = criterion.measure(out.clone(), y.clone());
        loss.backward();
        test_res.push((out.item()[0], y.item()[0]))
    }

    println!("TRAINING RESULT: {:#?}", training_res);
    println!("TEST RESULT: {:#?}", test_res);
    println!("MODEL: {:?}", mlp.parameters());
}

struct MLP {
    linear1: Linear,
}

impl MLP {
    pub fn new(features: [usize; 2]) -> Self {
        Self {
            linear1: Linear::new(features[0], features[1]),
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
        x
    }
}
