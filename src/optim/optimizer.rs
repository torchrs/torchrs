pub use std::collections::HashMap;
pub use autograd::{Variable, VarKind, VarId};
pub use nn::ParamIter;
pub use tensor::{Tensor, TensorKind, NewSelf};
use std::cell::RefCell;
use std::hash::Hash;
use std::ops::{Index, IndexMut};
use std::fmt::Debug;

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

pub struct Optimizer<'a, T: Copy + 'a> {
    pub params: ParamIter<'a, T>,
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

impl<T> From<OptimVal> for Tensor<T> {
    fn from(input: OptimVal) -> Self {
        match input {
            self::OptimVal::Tensor(x) => x.clone().into(),
            _ => unimplemented!(),
        }
    }
}

pub type ParamState = HashMap<&'static str, OptimVal>;

impl<'a, T: Copy + 'a> Optimizer<'a, T> {
    pub fn new(params: ParamIter<'a, T>, defaults: HashMap<&'static str, OptimVal>) -> Self {
        Optimizer {
            params: params,
            defaults: defaults,
            state: MutMap::new(),
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
            let mut opt_grad = p.v.grad();
            // XXX where is this first allocated?
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
