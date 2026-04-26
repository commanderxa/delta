use crate::optim::Optim;

use super::Scheduler;

pub struct MultiStepLR {
    optimizer: Box<dyn Optim>,
    pub milestones: Vec<usize>,
    pub gamma: f64,
    count: usize,
}

impl MultiStepLR {
    pub fn new(optimizer: Box<dyn Optim>, milestones: &[usize], gamma: f64) -> Self {
        Self {
            optimizer,
            milestones: milestones.to_vec(),
            gamma: gamma,
            count: 0,
        }
    }
}

impl Scheduler for MultiStepLR {
    fn step(&mut self) -> () {
        for m in &self.milestones {
            if self.count == *m {
                self.optimizer.as_mut().change_lr(self.gamma);
            }
        }
        self.optimizer.as_mut().step();
        self.count += 1;
    }
}
