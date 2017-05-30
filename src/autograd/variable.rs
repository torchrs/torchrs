use autograd::Function;
use tensor::Tensor;
use std::ops::{AddAssign, Index};
use ::*;

pub type VarList<T> = Vec<Variable<T>>;

pub enum SavedVarKind {
    FloatVariable(SavedVariable<f32>),
    LongVariable(SavedVariable<i64>),
}
pub enum VarKind {
    FloatVariable(Variable<f32>),
    LongVariable(Variable<i64>),
}

impl Clone for VarKind {
    fn clone(&self) -> Self {
        panic!("match and clone!")
    }
}

pub struct VariableImpl<T> {
    data: Tensor<T>,
    // AKA Creator
    grad_fn: OptRcMut<Function>,
    grad: OptRcMut<Variable<T>>,
    // version_counter etc ...
    volatile: bool,
    requires_grad: bool,
}

impl<T> VariableImpl<T> {
    fn new(data: Tensor<T>) -> Self {
        VariableImpl {
            data: data,
            grad_fn: None,
            grad: None,
            // XXX
            volatile: false,
            requires_grad: false,
        }
    }
}

pub struct Variable<T> {
    value: RcMut<VariableImpl<T>>,
}

impl<T> Clone for Variable<T> {
    fn clone(&self) -> Self {
        Variable { value: self.value.clone() }
    }
}

pub struct SavedVariable<T> {
    data: Box<Tensor<T>>,
    grad_fn: OptRcMut<Function>,
    grad: OptRcMut<Variable<T>>,
	// version_counter etc ...
}

#[derive(Default, Clone)]
pub struct BackwardArgs {}

pub struct VariableArgs {}

impl<T> Variable<T> {
    pub fn new(data: Tensor<T>) -> Self {
        Variable { value: RcMutNew(VariableImpl::new(data)) }
    }
    pub fn new_volatile(data: Tensor<T>) -> Self {
        Variable { value: RcMutNew(VariableImpl::new(data)) }
    }
    pub fn apply(&mut self, callback: fn(&mut Tensor<T>)) {
        let mut v = self.value.borrow_mut();
        callback(&mut v.data);
    }
    // XXX FIXME
    pub fn data(&mut self) -> Tensor<T> {
        self.value.borrow_mut().data.clone()
    }
    pub fn view(&self, dims: &[i32]) -> Self {
        self.clone()
    }
    // Computes the gradient of current variable w.r.t. graph leaves
    pub fn backward(&mut self, args: &BackwardArgs) {}
    // Detach from graph
    pub fn detach_(&mut self) {}
    // return a new variable detached from graph
    pub fn detach(&self) -> Variable<T> {
        self.clone()
    }
}

impl<T> Default for Variable<T> {
    fn default() -> Self {
        let v = VariableImpl {
            data: Tensor::new(),
            grad_fn: None,
            grad: None,
            // XXX
            volatile: false,
            requires_grad: false,
        };
        Variable { value: RcMutNew(v) }
    }
}

impl<T: Copy> Index<isize> for Variable<T> {
    type Output = T;

    fn index(&self, idx: isize) -> &Self::Output {
        panic!("implement")
    }
}

impl AddAssign<Variable<f32>> for f32 {
    fn add_assign(&mut self, rhs: Variable<f32>) {
        *self = *self + rhs[0]
    }
}
