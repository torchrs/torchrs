use autograd::{Function, ExecutionEngine, FuncId};
use tensor::Tensor;
use std::ops::{AddAssign, Index};
use std::collections::VecDeque;
use std::marker::PhantomData;
use tensor::*;
use ::*;

thread_local! {
    pub static VAR_TABLE_F32: RefCell<VecDeque<VariableImpl<f32>>> = RefCell::new(VecDeque::new());
    pub static VAR_TABLE_I64: RefCell<VecDeque<VariableImpl<i64>>> = RefCell::new(VecDeque::new());
}

pub type VarList<T> = Vec<Variable<T>>;
pub type VarId = i32;


pub trait VarAccess<T> {
    fn access(&self) -> &mut VariableImpl<T>;
    fn new_args(data: Tensor<T>, args: VariableArgs) -> Self;
}

impl<T> VarAccess<T> for Variable<T> {
    default fn access(&self) -> &mut VariableImpl<T> {
        panic!("unsupported Tensor type")
    }
    default fn new_args(data: Tensor<T>, args: VariableArgs) -> Self {
        panic!("unsupported Tensor type")
    }
}

impl VarAccess<f32> for Variable<f32> {
    fn access(&self) -> &mut VariableImpl<f32> {
        let vecp = VAR_TABLE_F32.with(|f| f.as_ptr());
        let vec = unsafe { &mut *vecp };
        &mut vec[self.id as usize]
    }
    fn new_args(data: Tensor<f32>, args: VariableArgs) -> Self {
        let mut id = ::std::usize::MAX;
        let value = VariableImpl::new(data, args);

        VAR_TABLE_F32.with(|f| {
                               let mut table = f.borrow_mut();
                               id = table.len();
                               table.push_back(value);
                           });
        Variable {
            id: id as i32,
            phantom: PhantomData,
        }
    }
}

impl VarAccess<i64> for Variable<i64> {
    fn access(&self) -> &mut VariableImpl<i64> {
        let vecp = VAR_TABLE_I64.with(|f| f.as_ptr());
        let vec = unsafe { &mut *vecp };
        &mut vec[self.id as usize]
    }
    fn new_args(data: Tensor<i64>, args: VariableArgs) -> Self {
        let mut id = ::std::usize::MAX;
        let value = VariableImpl::new(data, args);

        VAR_TABLE_I64.with(|f| {
                               let mut table = f.borrow_mut();
                               id = table.len();
                               table.push_back(value);
                           });
        Variable {
            id: id as i32,
            phantom: PhantomData,
        }
    }
}

pub struct VariableImpl<T> {
    pub data: Tensor<T>,
    // AKA Creator Id
    grad_fn: Option<Function>,
    grad: Option<VarId>,
    // version_counter etc ...
    dirty: bool,
    volatile: bool,
    requires_grad: bool,
}


impl<T> VariableImpl<T> {
    fn new(data_: Tensor<T>, args: VariableArgs) -> Self {
        VariableImpl {
            data: data_,
            grad_fn: args.creator,
            grad: None,
            dirty: false,
            volatile: args.volatile,
            requires_grad: args.requires_grad,
        }
    }
}

pub struct Variable<T> {
    pub id: VarId,
    phantom: PhantomData<T>,
}

impl<T> Default for Variable<T> {
    fn default() -> Self {
        Variable {
            id: -1,
            phantom: PhantomData,
        }
    }
}

impl<T> Clone for Variable<T> {
    fn clone(&self) -> Self {
        Variable {
            id: self.id,
            phantom: PhantomData,
        }
    }
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
    pub fn is_volatile(&self) -> bool {
        self.access().volatile
    }
    pub fn requires_grad(&self) -> bool {
        self.access().requires_grad
    }
    pub fn data(&mut self) -> &mut Tensor<T> {
        &mut self.access().data
    }
    pub fn apply(&mut self, callback: fn(&mut Tensor<T>)) {
        callback(&mut self.access().data);
    }
    pub fn mark_dirty(&mut self) {
        self.access().dirty = true;
    }
    pub fn view(&self, dims: &[i32]) -> Self {
        unimplemented!()
    }
    // Computes the gradient of current variable w.r.t. graph leaves
    pub fn backward_args(&mut self, gradient_: Option<&mut Tensor<T>>, retain_variables: bool) {
        let mut store;
        if self.access().volatile {
            panic!("calling backward on a volatile variable")
        }
        if !self.access().requires_grad {
            panic!("calling backward on a variable that doesn't require a gradient")
        }
        let mut gradient = match gradient_ {
            Some(gradient) => gradient,
            None => {
                store = self.access().data.new_(1);
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
