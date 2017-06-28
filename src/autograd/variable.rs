use autograd::{Function, FuncId, ExecutionEngine};
use tensor::Tensor;
use std::ops::{AddAssign, Index};
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::hash::{Hash, Hasher};
use num;
use tensor::*;
use ::*;

thread_local! {
    pub static VAR_TABLE: RefCell<VecDeque<VarKindImpl>> = RefCell::new(VecDeque::new());
}
pub type VarList<T> = Vec<Variable<T>>;
pub type VarKindList = Vec<VarKind>;
pub type OptVarKind = Option<VarKind>;
pub type OptVarKindList = Vec<OptVarKind>;
pub type RefVarKindList<'a> = Vec<&'a VarKind>;
pub type VarId = i32;
pub enum VarKind {
    FloatVariable(Variable<f32>),
    LongVariable(Variable<i64>),
}
pub enum VarKindImpl {
    FloatVariable(VariableImpl<f32>),
    LongVariable(VariableImpl<i64>),
}

impl<T: NumLimits> From<VarKindImpl> for VariableImpl<T> {
    default fn from(input: VarKindImpl) -> Self {
        unreachable!()
    }
}
impl From<VarKindImpl> for VariableImpl<f32> {
    fn from(input: VarKindImpl) -> Self {
        if let VarKindImpl::FloatVariable(v) = input {
            v
        } else {
            unreachable!()
        }
    }
}
impl From<VarKindImpl> for VariableImpl<i64> {
    fn from(input: VarKindImpl) -> Self {
        if let VarKindImpl::LongVariable(v) = input {
            v
        } else {
            unreachable!()
        }
    }
}

impl<T: NumLimits> From<VariableImpl<T>> for VarKindImpl {
    default fn from(input: VariableImpl<T>) -> Self {
        unreachable!()
    }
}
impl From<VariableImpl<f32>> for VarKindImpl {
    fn from(input: VariableImpl<f32>) -> Self {
        VarKindImpl::FloatVariable(input)
    }
}
impl From<VariableImpl<i64>> for VarKindImpl {
    fn from(input: VariableImpl<i64>) -> Self {
        VarKindImpl::LongVariable(input)
    }
}

impl<T: NumLimits> From<Variable<T>> for VarKind {
    default fn from(input: Variable<T>) -> Self {
        panic!("bad cast")
    }
}
impl From<Variable<f32>> for VarKind {
    fn from(input: Variable<f32>) -> Self {
        VarKind::FloatVariable(input)
    }
}
impl From<Variable<i64>> for VarKind {
    fn from(input: Variable<i64>) -> Self {
        VarKind::LongVariable(input)
    }
}
impl From<VarKind> for TensorKind {
    fn from(input: VarKind) -> Self {
        input.clone().data().clone()
    }
}
impl<T: NumLimits> From<VarKind> for Tensor<T> {
    fn from(input: VarKind) -> Self {
        let t: TensorKind = input.into();
        t.into()
    }
}
impl<T: NumLimits> From<Tensor<T>> for VarKind {
    fn from(input: Tensor<T>) -> Self {
        let t: TensorKind = input.into();
        t.into()
    }
}

impl From<TensorKind> for VarKind {
    fn from(input: TensorKind) -> Self {
        VarKind::new_args(input, &VariableArgs::default())
    }
}


impl From<VarId> for VarKind {
    fn from(id: VarId) -> VarKind {
        let vecp = VAR_TABLE.with(|f| f.as_ptr());
        let vec = unsafe { &mut *vecp };
        match vec[id as usize] {
            VarKindImpl::FloatVariable(_) => Variable::<f32>::from(id).into(),
            VarKindImpl::LongVariable(_) => Variable::<i64>::from(id).into(),
        }
    }
}

impl<T: NumLimits> From<VarKind> for Variable<T> {
    default fn from(input: VarKind) -> Self {
        panic!("bad cast");
    }
}
impl From<VarKind> for Variable<f32> {
    fn from(input: VarKind) -> Self {
        if let VarKind::FloatVariable(v) = input {
            v
        } else {
            panic!("bad cast")
        }
    }
}
impl From<VarKind> for Variable<i64> {
    fn from(input: VarKind) -> Self {
        if let VarKind::LongVariable(v) = input {
            v
        } else {
            panic!("bad cast")
        }
    }
}
impl<'a, T: 'a + NumLimits> From<&'a VarKind> for &'a Variable<T> {
    default fn from(input: &'a VarKind) -> Self {
        panic!("bad cast");
    }
}
impl<'a> From<&'a VarKind> for &'a Variable<f32> {
    fn from(input: &'a VarKind) -> Self {
        if let &VarKind::FloatVariable(ref v) = input {
            v
        } else {
            panic!("bad cast")
        }
    }
}
impl<'a> From<&'a VarKind> for &'a Variable<i64> {
    fn from(input: &'a VarKind) -> Self {
        if let &VarKind::LongVariable(ref v) = input {
            v
        } else {
            panic!("bad cast")
        }
    }
}

impl PartialEq for VarKind {
    fn eq(&self, other: &Self) -> bool {
        use self::VarKind::{FloatVariable, LongVariable};
        match (self, other) {
            (&FloatVariable(ref t1), &FloatVariable(ref t2)) => t1.id() == t2.id(),
            (&LongVariable(ref t1), &LongVariable(ref t2)) => t1.id() == t2.id(),
            _ => false,
        }
    }
}
impl Eq for VarKind {}
impl Hash for VarKind {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.varid().hash(state)
    }
}


impl Clone for VarKind {
    fn clone(&self) -> Self {
        use self::VarKind::{FloatVariable, LongVariable};
        match *self {
            FloatVariable(ref v) => FloatVariable(v.clone()),
            LongVariable(ref v) => LongVariable(v.clone()),
            _ => unimplemented!(),
        }
    }
}

pub trait VarAccess<T: NumLimits> {
    fn access<'a>(&self) -> &'a mut VariableImpl<T>;
    fn borrow(&self) -> &VariableImpl<T>;
    fn new_args(data: Tensor<T>, args: &VariableArgs) -> Self;
}

impl<T: NumLimits> VarAccess<T> for Variable<T> {
    default fn access<'a>(&self) -> &'a mut VariableImpl<T> {
        panic!("unsupported Tensor type")
    }
    default fn borrow(&self) -> &VariableImpl<T> {
        panic!("unsupported Tensor type")
    }
    default fn new_args(data: Tensor<T>, args: &VariableArgs) -> Self {
        panic!("unsupported Tensor type")
    }
}

impl VarAccess<f32> for Variable<f32> {
    fn access<'a>(&self) -> &'a mut VariableImpl<f32> {
        let vecp = VAR_TABLE.with(|f| f.as_ptr());
        let vec = unsafe { &mut *vecp };
        match vec[self.id as usize] {
            VarKindImpl::FloatVariable(ref mut t) => t,
            _ => unreachable!(),
        }
    }
    fn borrow(&self) -> &VariableImpl<f32> {
        let vecp = VAR_TABLE.with(|f| f.as_ptr());
        let vec = unsafe { &mut *vecp };
        match vec[self.id as usize] {
            VarKindImpl::FloatVariable(ref t) => t,
            _ => unreachable!(),
        }
    }
    fn new_args(data: Tensor<f32>, args: &VariableArgs) -> Self {
        let mut id = ::std::usize::MAX;
        let value = VariableImpl::new(data, args);

        VAR_TABLE.with(|f| {
                           let mut table = f.borrow_mut();
                           id = table.len();
                           table.push_back(value.into());
                       });
        Variable {
            id: id as i32,
            phantom: PhantomData,
        }
    }
}

impl VarAccess<i64> for Variable<i64> {
    fn access<'a>(&self) -> &'a mut VariableImpl<i64> {
        let vecp = VAR_TABLE.with(|f| f.as_ptr());
        let vec = unsafe { &mut *vecp };
        match vec[self.id as usize] {
            VarKindImpl::LongVariable(ref mut t) => t,
            _ => unreachable!(),
        }

    }
    fn borrow(&self) -> &VariableImpl<i64> {
        let vecp = VAR_TABLE.with(|f| f.as_ptr());
        let vec = unsafe { &mut *vecp };
        match vec[self.id as usize] {
            VarKindImpl::LongVariable(ref t) => t,
            _ => unreachable!(),
        }

    }
    fn new_args(data: Tensor<i64>, args: &VariableArgs) -> Self {
        let mut id = ::std::usize::MAX;
        let value = VariableImpl::new(data, args);

        VAR_TABLE.with(|f| {
                           let mut table = f.borrow_mut();
                           id = table.len();
                           table.push_back(value.into());
                       });
        Variable {
            id: id as i32,
            phantom: PhantomData,
        }
    }
}

pub struct VariableImpl<T: NumLimits> {
    pub data: Tensor<T>,
    // AKA Creator Id
    grad_fn: Option<Function>,
    grad: Option<Variable<T>>,
    // version_counter etc ...
    dirty: bool,
    volatile: bool,
    requires_grad: bool,
}

impl<T: NumLimits> VariableImpl<T> {
    fn new(data: Tensor<T>, args: &VariableArgs) -> Self {
        let creator = match args.creator {
            Some(ref f) => Some(f.clone()),
            None => None,
        };
        VariableImpl {
            data: data,
            grad_fn: creator,
            grad: None,
            dirty: false,
            volatile: args.volatile,
            requires_grad: args.requires_grad,
        }
    }
    fn grad(&mut self) -> &mut Option<Variable<T>> {
        &mut self.grad
    }
    fn _call_hooks(&self, grad_output: &Tensor<T>) {
        unimplemented!()
    }
    pub fn copy_refs(&mut self, rhs: &Self) {
        self.grad_fn = rhs.grad_fn.clone();
        self.grad = rhs.grad.clone();
        self.dirty = rhs.dirty;
        self.volatile = rhs.volatile;
        self.requires_grad = rhs.requires_grad;
    }
}


pub struct Variable<T: NumLimits> {
    pub id: VarId,
    phantom: PhantomData<T>,
}

impl<T: NumLimits> Default for Variable<T> {
    fn default() -> Self {
        Variable {
            id: -1,
            phantom: PhantomData,
        }
    }
}
impl<T: NumLimits> Clone for Variable<T> {
    fn clone(&self) -> Self {
        Variable {
            id: self.id,
            phantom: PhantomData,
        }
    }
}
impl<T: NumLimits> From<u32> for Variable<T> {
    fn from(id: u32) -> Self {
        Variable {
            id: id as i32,
            phantom: PhantomData,
        }
    }
}
impl<T: NumLimits> From<i32> for Variable<T> {
    fn from(id: i32) -> Self {
        Variable {
            id: id,
            phantom: PhantomData,
        }
    }
}
impl<'a, T: 'a + NumLimits> From<&'a i32> for Variable<T> {
    fn from(id: &'a i32) -> Self {
        Variable {
            id: *id,
            phantom: PhantomData,
        }
    }
}
impl<T: NumLimits> From<usize> for Variable<T> {
    fn from(id: usize) -> Self {
        Variable {
            id: id as i32,
            phantom: PhantomData,
        }
    }
}

#[derive(Builder)]
#[builder(pattern="owned")]
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
impl VariableArgs {
    pub fn build() -> VariableArgsBuilder {
        VariableArgsBuilder::default()
    }
}
impl VariableArgsBuilder {
    pub fn done(self) -> VariableArgs {
        self.build().unwrap()
    }
}

macro_rules! impl_var_dispatch {
    ($key:ident, $var:ident, $action:expr ) => {(
        match * $key {
            FloatVariable(ref $var) => $action ,
            LongVariable(ref $var) => $action ,
        }
    )}
}

macro_rules! impl_var_mut_dispatch {
    ($key:ident, $var:ident, $action:expr ) => {(
        match * $key {
            FloatVariable(ref mut $var) => $action ,
            LongVariable(ref mut $var) => $action ,
        }
    )}
}

impl VarKind {
    pub fn new_args(data: TensorKind, args: &VariableArgs) -> Self {
        use self::TensorKind::{FloatTensor, LongTensor};
        match data {
            FloatTensor(t) => Variable::<f32>::new_args(t, args).into(),
            LongTensor(t) => Variable::<i64>::new_args(t, args).into(),
        }
    }

    pub fn is_volatile(&self) -> bool {
        use self::VarKind::{FloatVariable, LongVariable};
        impl_var_dispatch!(self, v, v.is_volatile())
    }
    pub fn varid(&self) -> VarId {
        use self::VarKind::{FloatVariable, LongVariable};
        impl_var_dispatch!(self, v, v.id)
    }
    pub fn requires_grad(&self) -> bool {
        use self::VarKind::{FloatVariable, LongVariable};
        impl_var_dispatch!(self, v, v.requires_grad())
    }
    pub fn grad_fn(&self) -> Option<Function> {
        use self::VarKind::{FloatVariable, LongVariable};
        impl_var_dispatch!(self, v, v.grad_fn())
    }
    pub fn data(&mut self) -> TensorKind {
        use self::VarKind::{FloatVariable, LongVariable};
        impl_var_mut_dispatch!(self, v, v.data_into())
    }
    pub fn data_borrow(&self) -> TensorKind {
        use self::VarKind::{FloatVariable, LongVariable};
        let mut self_ = self.clone();
        let mut self__ = &mut self_;
        impl_var_mut_dispatch!(self__, v, v.data_into())
    }
    pub fn tid(&mut self) -> TensorId {
        use self::VarKind::{FloatVariable, LongVariable};
        match *self {
            FloatVariable(ref mut v) => v.data().id(),
            LongVariable(ref mut v) => v.data().id(),
        }
    }
    pub fn requires_nograd(&mut self) {
        use self::VarKind::{FloatVariable, LongVariable};
        impl_var_mut_dispatch!(self, v, v.requires_nograd())
    }
    pub fn typed<T: NumLimits>(self) -> Variable<T> {
        Variable::<T>::from(self)
    }
    pub fn _do_backward(&mut self, grad_output_: &mut Option<VarKind>) {
        use self::VarKind::{FloatVariable, LongVariable};
        if let Some(ref mut grad_output) = *grad_output_ {
            impl_var_mut_dispatch!(self, v, v._do_backward(&mut grad_output.clone().into()))
        }
    }
}

impl<T: NumLimits> Variable<T> {
    pub fn new(data: Tensor<T>) -> Self {
        Variable::new_args(data, &VariableArgs::default())
    }
    fn data_into(&mut self) -> TensorKind {
        self.data().clone().into()
    }
    pub fn is_volatile(&self) -> bool {
        self.access().volatile
    }
    pub fn requires_grad(&self) -> bool {
        self.access().requires_grad
    }
    pub fn grad_fn(&self) -> Option<Function> {
        match self.access().grad_fn {
            Some(ref func) => Some(func.clone()),
            None => None,
        }
    }
    pub fn grad(&mut self) -> &mut Option<Variable<T>> {
        self.access().grad()
    }
    pub fn data(&mut self) -> &mut Tensor<T> {
        &mut self.access().data
    }
    pub fn data_borrow(&self) -> &Tensor<T> {
        &self.borrow().data
    }
    pub fn apply(&mut self, callback: fn(&mut Tensor<T>)) {
        callback(&mut self.access().data);
    }
    pub fn mark_dirty(&mut self) {
        self.access().dirty = true;
    }
    pub fn requires_nograd(&mut self) {
        self.access().requires_grad = false;
    }
    // Computes the gradient of current variable w.r.t. graph leaves
    pub fn backward_args(&mut self, gradient_: Option<&mut Tensor<T>>, retain_variables: bool) {
        let mut store;
        let gradient;
        {
            let parent = self.access();
            if parent.volatile {
                panic!("calling backward on a volatile variable")
            }
            if !parent.requires_grad {
                panic!("calling backward on a variable that doesn't require a gradient")
            }
            gradient = match gradient_ {
                Some(gradient) => gradient,
                None => {
                    store = parent.data.new(T::one());
                    &mut store
                }
            };
        }
        ExecutionEngine::run_backward(self, gradient.clone().into(), retain_variables)
    }
    pub fn _do_backward(&mut self, grad_output: &mut Variable<T>) {
        let inner = self.access();
        assert_eq!(inner.dirty, false);
        inner._call_hooks(grad_output.data_borrow());
        // in place operations consume their input
        // grad returns a borrowed reference
        inner.grad().clone().unwrap().add_(grad_output[0]);
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

impl<T: NumLimits> Index<isize> for Variable<T> {
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
