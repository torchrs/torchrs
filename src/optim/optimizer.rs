pub use std::collections::HashMap;
pub use autograd::Variable;
pub use nn::ParamIter;
use tensor::NewSelf;

pub struct Optimizer<'a, T: Copy + 'a> {
    pub params: ParamIter<'a, T>,
    pub defaults: HashMap<&'static str, OptimOpts>,
}
#[derive(Clone)]
pub enum OptimOpts {
    Bool(bool),
    Int(i32),
    Float(f32),
    Required,
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

impl OptimOpts {
    pub fn intof32(&self) -> f32 {
        use self::OptimOpts::Float;
        match *self {
            Float(v) => v,
            _ => unimplemented!(),
        }
    }
    pub fn intoi32(&self) -> i32 {
        use self::OptimOpts::Int;
        match *self {
            Int(v) => v,
            _ => unimplemented!(),
        }
    }
    pub fn intobool(&self) -> bool {
        use self::OptimOpts::Bool;
        match *self {
            Bool(v) => v,
            _ => unimplemented!(),
        }
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
        let params = opt.params.clone();
        // XXX figure out point of parameter groups
        for p in params {
            // XXX when would this ever be None since we allocate on lookup?
            let mut opt_grad = p.v.grad();
            if let Some(ref mut grad) = opt_grad.clone() {
                if grad.is_volatile() {
                    grad.data().zero_();
                } else {
                    let data = grad.data();
                    *opt_grad = Some(Variable::new(data.new(()).zero_().clone()));
                }
            }
        }
    }
    /* ignore largely unused closure arg to start */
    fn step(&mut self);
}
