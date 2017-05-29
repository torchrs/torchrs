
pub struct Optimizer {}

impl Optimizer {
    pub fn new() -> Self {
        Optimizer {}
    }
}

pub trait OptIntf {
    fn zero_grad(&mut self) {}
    /* ignore largely unused closure arg to start */
    fn step(&mut self);
}
