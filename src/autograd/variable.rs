use autograd::{Function, ExecutionEngine};
use tensor::Tensor;
use std::ops::{AddAssign, Index};
use tensor::*;
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
        unimplemented!()
    }
}

pub struct VariableImpl<T> {
    // AKA Creator
    grad_fn: OptRcMut<Function>,
    grad: OptRcMut<Variable<T>>,
    // version_counter etc ...
    dirty: bool,
    volatile: bool,
    requires_grad: bool,
}

impl<T> VariableImpl<T> {
    fn new(args: VariableArgs) -> Self {

        let grad_fn = match args.creator {
            Some(creator) => Some(RcMutNew(creator)),
            _ => None,
        };

        VariableImpl {
            grad_fn: grad_fn,
            grad: None,
            dirty: false,
            volatile: args.volatile,
            requires_grad: args.requires_grad,
        }
    }
}

pub struct Variable<T> {
    pub data: Tensor<T>,
    value: RcMut<VariableImpl<T>>,
}

impl<T> Clone for Variable<T> {
    fn clone(&self) -> Self {
        Variable {
            data: self.data.clone(),
            value: self.value.clone(),
        }
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


pub struct VariableArgs {
    pub creator: Option<Function>,
    pub volatile: bool,
    pub requires_grad: bool,
}

impl Default for VariableArgs {
    fn default() -> Self {
        VariableArgs {
            creator: None,
            volatile: false,
            requires_grad: true,
        }
    }
}

impl<T> Variable<T> {
    pub fn new(data: Tensor<T>) -> Self {
        Variable::new_args(data, VariableArgs::default())
    }
    fn inner(&mut self) -> RefMut<VariableImpl<T>> {
        self.value.borrow_mut()
    }
    pub fn is_volatile(&self) -> bool {
        self.inner().volatile
    }
    pub fn new_args(data: Tensor<T>, args: VariableArgs) -> Self {
        Variable {
            data: data,
            value: RcMutNew(VariableImpl::new(args)),
        }
    }
    pub fn apply(&mut self, callback: fn(&mut Tensor<T>)) {
        callback(&mut self.data);
    }
    pub fn mark_dirty(&mut self) {
        self.value.borrow_mut().dirty = true;
    }
    pub fn view(&self, dims: &[i32]) -> Self {
        unimplemented!()
    }
    // Computes the gradient of current variable w.r.t. graph leaves
    pub fn backward_args(&mut self, gradient_: Option<&mut Tensor<T>>, retain_variables: bool) {
        let mut store;
        if self.inner().volatile {
            panic!("calling backward on a volatile variable")
        }
        if !self.inner().requires_grad {
            panic!("calling backward on a variable that doesn't require a gradient")
        }
        let mut gradient = match gradient_ {
            Some(gradient) => gradient,
            None => {
                store = self.data.new_(1);
                &mut store
            }
        };
        ExecutionEngine::run_backward(self, &mut gradient, retain_variables)
    }
    pub fn backward(&mut self) {
        self.backward_args(None, false)
    }
    // Detach from graph
    pub fn detach_(&mut self) {
        unimplemented!()
    }
    // return a new variable detached from graph
    pub fn detach(&self) -> Variable<T> {
        unimplemented!()
    }
}

impl<T> Default for Variable<T> {
    fn default() -> Self {
        unimplemented!()
    }
}

impl<T: Copy> Index<isize> for Variable<T> {
    type Output = T;

    fn index(&self, idx: isize) -> &Self::Output {
        unimplemented!()
    }
}

impl AddAssign<Variable<f32>> for f32 {
    fn add_assign(&mut self, rhs: Variable<f32>) {
        *self = *self + rhs[0]
    }
}
