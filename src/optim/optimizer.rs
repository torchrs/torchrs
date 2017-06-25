pub use std::collections::HashMap;
pub use autograd::{Variable, VarKind, VarId};
pub use nn::ModIntf;
pub use tensor::{Tensor, TensorKind, NewSelf, NumLimits};
use nn::Parameter;
use std::ops::Neg;
use utils::unsafe_lib::MutMap;
use num;


pub struct Optimizer {
    pub defaults: HashMap<&'static str, OptimVal>,
    pub state: MutMap<VarId, ParamState>,
}

#[derive(Clone)]
pub enum OptimVal {
    Bool(bool),
    Int(i32),
    Float(f32),
    Tensor(TensorKind),
    Variable(VarKind),
    Required,
}
impl From<f32> for OptimVal {
    fn from(input: f32) -> Self {
        OptimVal::Float(input)
    }
}
impl From<i32> for OptimVal {
    fn from(input: i32) -> Self {
        OptimVal::Int(input)
    }
}
impl From<bool> for OptimVal {
    fn from(input: bool) -> Self {
        OptimVal::Bool(input)
    }
}
impl From<TensorKind> for OptimVal {
    fn from(input: TensorKind) -> Self {
        OptimVal::Tensor(input)
    }
}
impl<T: NumLimits<T>> From<Tensor<T>> for OptimVal {
    fn from(input: Tensor<T>) -> Self {
        OptimVal::Tensor(input.into())
    }
}
impl From<OptimVal> for bool {
    fn from(input: OptimVal) -> Self {
        match input {
            self::OptimVal::Bool(x) => x.clone(),
            _ => unimplemented!(),
        }
    }
}
impl From<OptimVal> for f32 {
    fn from(input: OptimVal) -> Self {
        match input {
            self::OptimVal::Float(x) => x.clone(),
            _ => unimplemented!(),
        }
    }
}

impl<T: NumLimits<T>> From<OptimVal> for Tensor<T> {
    fn from(input: OptimVal) -> Self {
        match input {
            self::OptimVal::Tensor(x) => x.clone().into(),
            _ => unimplemented!(),
        }
    }
}

pub type ParamState = HashMap<&'static str, OptimVal>;

impl Optimizer {
    pub fn new(defaults: HashMap<&'static str, OptimVal>) -> Self {
        Optimizer {
            defaults: defaults,
            state: MutMap::new(),
        }
    }
}

pub trait OptIntf<T: ::tensor::NumLimits<T> + From<OptimVal>> {
    fn optimizer(&mut self) -> &mut Optimizer;
    fn zero_grad(&mut self, model: &mut ModIntf<T>) {
        // XXX figure out point of parameter groups
        model.apply_parameters(&mut |p| {
            let mut opt_grad = p.grad();
            // XXX where is this first allocated?
            if let Some(ref mut grad) = opt_grad.clone() {
                if grad.is_volatile() {
                    grad.data().zero_();
                } else {
                    let data = grad.data();
                    *opt_grad = Some(Variable::new(data.new(()).zero_().clone()));
                }
            }
        });
    }
    /* ignore largely unused closure arg to start */
    fn step(&mut self, model: &mut ModIntf<T>);
}
