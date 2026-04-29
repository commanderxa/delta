use std::collections::HashMap;

use delta::{
    ivalue,
    ivalue::IValue,
    nn::{self, Linear, MSELoss, Module, functional as F},
    optim::{Optim, SGD},
};

fn main() {
    let epochs = 1000;
    let criterion = MSELoss::default();
    let mlp = MLP::new([2, 1]);
    let optim = SGD::new(mlp.parameters(), 1e-1);

    let data = vec![
        delta::tensor(&[9., 3.], &[2]),
        delta::tensor(&[2., 3.], &[2]),
        delta::tensor(&[3., 7.], &[2]),
        delta::tensor(&[8., 6.], &[2]),
        delta::tensor(&[4., 4.], &[2]),
        delta::tensor(&[4., 1.], &[2]),
        delta::tensor(&[5., 2.], &[2]),
        delta::tensor(&[2., 5.], &[2]),
        delta::tensor(&[5., 6.], &[2]),
        delta::tensor(&[7., 2.], &[2]),
        delta::tensor(&[9., 1.], &[2]),
    ];
    let targets = vec![
        delta::tensor(&[0.], &[1]),
        delta::tensor(&[1.], &[1]),
        delta::tensor(&[1.], &[1]),
        delta::tensor(&[1.], &[1]),
        delta::tensor(&[1.], &[1]),
        delta::tensor(&[0.], &[1]),
        delta::tensor(&[0.], &[1]),
        delta::tensor(&[1.], &[1]),
        delta::tensor(&[1.], &[1]),
        delta::tensor(&[0.], &[1]),
        delta::tensor(&[0.], &[1]),
    ];

    let test_data = vec![
        delta::tensor(&[6., 6.], &[2]),
        delta::tensor(&[8., 2.], &[2]),
    ];
    let test_targets = vec![delta::tensor(&[1.], &[1]), delta::tensor(&[0.], &[1])];

    for epoch in 1..=epochs {
        let mut losses = 0.0;
        for (x, y) in data.iter().zip(targets.clone()) {
            optim.zero_grad();

            let x = x.clone().unsqueeze(0);
            let y = y.clone();
            let (args, kwargs) = ivalue![[x]];
            let out = mlp.forward(args, kwargs).unwrap_tensor().squeeze(&[0]);
            let loss = criterion.measure(out, y);

            loss.backward();
            optim.step();

            losses += loss.item()[0];
        }
        println!("Epoch: {epoch}/{epochs} | loss: {:.10}", losses);
    }

    for (x, y) in test_data.iter().zip(test_targets.clone()) {
        let x = x.clone().unsqueeze(0);
        let (args, kwargs) = ivalue![[x.clone()]];
        let out = mlp.forward(args, kwargs).unwrap_tensor().squeeze(&[0]);

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
}

impl MLP {
    pub fn new(features: [usize; 2]) -> Self {
        Self {
            linear: Linear::new(features[0], features[1], true),
        }
    }
}

impl Module for MLP {
    fn module_name(&self) -> String {
        "MLP".to_owned()
    }

    fn parameters(&self) -> Vec<nn::Parameter> {
        let parameters = self.linear.parameters();
        parameters
    }

    fn forward(&self, args: Vec<IValue>, kwargs: HashMap<String, IValue>) -> IValue {
        let x = self.linear.forward(args, kwargs).unwrap_tensor();
        IValue::Tensor(F::sigmoid(x))
    }
}
