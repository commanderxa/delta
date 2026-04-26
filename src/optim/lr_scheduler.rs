pub mod multistep_lr;

pub trait Scheduler {
    fn step(&mut self) -> ();
}
