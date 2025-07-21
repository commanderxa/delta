use super::Optim;

pub mod multistep_lr;

pub struct Scheduler {
    optimizer: Box<dyn Optim>,
}

impl Scheduler {
    fn step(&mut self) -> ();
}
