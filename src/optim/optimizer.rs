pub use std::collections::HashMap;
pub use autograd::{Variable, VarKind, VarId};
pub use nn::ModIntf;
pub use tensor::{Tensor, TensorKind, NewSelf};
use std::cell::RefCell;
use std::hash::Hash;
use std::ops::{Index, IndexMut};
use std::fmt::Debug;
use nn::Parameter;
use std::marker::PhantomData;
use std::ops::Neg;
use num;

pub struct MutMap<K, V: Default> {
    map: HashMap<K, RefCell<V>>,
}

impl<K: Eq + Hash, V: Default> MutMap<K, V> {
    fn new() -> Self {
        MutMap { map: HashMap::new() }
    }
}

impl<K: Hash + Eq + Clone + Debug, V: Default> Index<K> for MutMap<K, V> {
    type Output = V;
    fn index(&self, idx: K) -> &Self::Output {
        let map = &self.map;
        if !map.contains_key(&idx) {
            panic!("{:?} not found", idx)
        }
        let cntp = map[&idx].as_ptr();
        unsafe { &*cntp }
    }
}
impl<K: Hash + Eq + Clone + Debug, V: Default> IndexMut<K> for MutMap<K, V> {
    fn index_mut(&mut self, idx: K) -> &mut Self::Output {
        let map = &mut self.map;
        if !map.contains_key(&idx) {
            map.insert(idx.clone(), RefCell::new(V::default()));
        }
        let cntp = map[&idx].as_ptr();
        unsafe { &mut *cntp }
    }
}

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
impl<T> From<Tensor<T>> for OptimVal {
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

impl<T> From<OptimVal> for Tensor<T> {
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

pub trait OptIntf<T: Copy + Default + From<OptimVal> + num::Num + num::Float + Neg<Output = T>>
     {
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
