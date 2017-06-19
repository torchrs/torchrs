pub use std::collections::HashMap;
pub use nn::ParamIter;

pub struct Optimizer<'a, T: Copy + 'a> {
    params: ParamIter<'a, T>,
    defaults: HashMap<&'static str, OptimOpts>,
}
#[derive(Clone)]
pub enum OptimOpts {
    Bool(bool),
    Int(i32),
    Float(f32),
}
impl<T: Copy> From<T> for OptimOpts {
    #[allow(unused_variables)]
    default fn from(input: T) -> Self {
        unreachable!()
    }
}
impl From<f32> for OptimOpts {
    fn from(input: f32) -> Self {
        OptimOpts::Float(input)
    }
}
impl From<i32> for OptimOpts {
    fn from(input: i32) -> Self {
        OptimOpts::Int(input)
    }
}
impl From<bool> for OptimOpts {
    fn from(input: bool) -> Self {
        OptimOpts::Bool(input)
    }
}

impl<'a, T: Copy + 'a> Optimizer<'a, T> {
    pub fn new(params: ParamIter<'a, T>, defaults: HashMap<&'static str, OptimOpts>) -> Self {
        Optimizer {
            params: params,
            defaults: defaults,
        }
    }
}

pub trait OptIntf<'a, T: Copy + 'a> {
    fn optimizer(&mut self) -> &mut Optimizer<'a, T>;
    fn zero_grad(&mut self) {
        let opt = self.optimizer();
    }
    /* ignore largely unused closure arg to start */
    fn step(&mut self);
}
