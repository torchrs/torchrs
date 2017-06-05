use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::vec::Vec;
use autograd::variable::*;
use tensor::*;
use ::*;

thread_local! {
    pub static FUNC_TABLE: RefCell<VecDeque<FuncImpl>> = RefCell::new(VecDeque::new());
}
pub type FuncId = i32;
pub enum RootKind {
    RootVar(VarKind),
    RootFunc(Function),
}
impl RootKind {
    pub fn requires_grad(&self) -> bool {
        match *self {
            RootKind::RootVar(ref v) => v.requires_grad(),
            RootKind::RootFunc(ref f) => f.requires_grad(),
        }
    }
}

impl_func!(FuncStub);
impl FuncIntf for FuncStub {
    fn forward(&mut self, input: &mut TensorKindList) -> TensorKindList {
        unreachable!()
    }
    fn backward(&mut self, input: &mut OptTensorKindList) -> OptTensorKindList {
        unreachable!()
    }
}
pub struct FuncImpl {
    previous_functions: Vec<(RootKind, i32)>,
    saved_variables: Vec<VarId>,
    needs_input_grad: Vec<bool>,
    non_differentiable: Vec<TensorId>,
    output_ids: HashMap<VarId, usize>,
    to_save: Vec<TensorId>,
    requires_grad: bool,
    owner: RcMut<FuncIntf>,
}
impl Default for FuncImpl {
    fn default() -> Self {
        FuncImpl {
            previous_functions: Vec::new(),
            saved_variables: Vec::new(),
            needs_input_grad: Vec::new(),
            non_differentiable: Vec::new(),
            output_ids: HashMap::new(),
            to_save: Vec::new(),
            requires_grad: false,
            owner: FuncStub::new().value,
        }
    }
}

impl FuncImpl {
    fn _call_hooks(&self, input: &TensorKindList, output: &TensorKindList) {
        unimplemented!();
    }
    fn init(&mut self, intf: RcMut<FuncIntf>) {
        self.owner = intf;
    }
}

#[derive(Clone)]
pub struct Function {
    pub id: FuncId,
}

impl Default for Function {
    fn default() -> Self {
        Function { id: -1 }
    }
}

pub struct FIWrap<T> {
    value: RcMut<T>,
}
impl<T: 'static + FuncIntf> FIWrap<T> {
    pub fn new(arg: T) -> Self {
        let t = FIWrap { value: RcMutNew(arg) };
        t.value.borrow_mut().delegate().init(t.value.clone());
        t
    }
    pub fn f(&mut self, mut input_: &mut VarKindList) -> VarKindList {
        self.value.borrow_mut().f(input_)
    }
}


impl Function {
    pub fn new() -> Self {
        use std::usize;
        let mut id = usize::MAX;
        FUNC_TABLE.with(|f| {
                            let mut table = f.borrow_mut();
                            id = table.len();
                            table.push_back(FuncImpl::default());
                        });
        Function { id: id as i32 }
    }
    pub fn init(&self, intf: RcMut<FuncIntf>) {
        //FUNC_INTF_TABLE.with(|m| m.borrow_mut().insert(self.id, intf));
        self.access().init(intf)
    }
    pub fn previous_functions(&self) -> &Vec<(RootKind, i32)> {
        &self.access().previous_functions
    }
    pub fn output_ids(&self) -> &HashMap<VarId, usize> {
        &self.access().output_ids
    }
    pub fn requires_grad(&self) -> bool {
        self.access().requires_grad
    }
    pub fn needs_input_grad(&self) -> &Vec<bool> {
        &self.access().needs_input_grad
    }
    fn access(&self) -> &mut FuncImpl {
        let vecp = FUNC_TABLE.with(|f| f.as_ptr());
        let vec = unsafe { &mut *vecp };
        &mut vec[self.id as usize]
    }
    pub fn from(id: FuncId) -> Self {
        Function { id: id }
    }
    pub fn saved_tensors(&mut self) -> TensorKindList {
        // XXX see if we can't avoid the clone
        self.access()
            .saved_variables
            .iter()
            .map(|v| VarKind::from(*v).data())
            .collect()
    }
    pub fn saved_variables(&mut self) -> VarKindList {
        // XXX see if we can't avoid the clone
        self.access()
            .saved_variables
            .iter()
            .map(|v| VarKind::from(*v))
            .collect()
    }
    pub fn save_for_backward(&mut self, input: &TensorKindList) {
        self.access().to_save = input.iter().map(|t| t.id()).collect();
    }
    pub fn _do_backward(&mut self,
                        grad_output: &mut OptTensorKindList,
                        retain_variables: bool)
                        -> OptTensorKindList {
        let inner = self.access();
        if inner.saved_variables.is_empty() {
            panic!("Trying to backward through the graph second \
                    time, but the buffers have already been freed. Please \
                    specify retain_variables=True when calling backward for \
                    the first time.");
        };
        let grad_input = inner.owner.borrow_mut().backward(grad_output);
        //inner._call_hooks(&grad_input, grad_output);
        if !retain_variables {
            inner.saved_variables.clear();
        };
        grad_input
    }
}

pub trait FuncDelegate {
    fn delegate(&mut self) -> &mut Function;
}

pub trait FuncIntf: FuncDelegate {
    fn forward(&mut self, input: &mut TensorKindList) -> TensorKindList;
    fn backward(&mut self, input: &mut OptTensorKindList) -> OptTensorKindList;
    fn save_for_backward(&mut self, input: &TensorKindList) {
        self.delegate().save_for_backward(input)
    }
    fn saved_variables(&mut self) -> VarKindList {
        self.delegate().saved_variables()
    }
    fn saved_tensors(&mut self) -> TensorKindList {
        self.delegate().saved_tensors()
    }
    fn needs_input_grad(&mut self) -> &Vec<bool> {
        self.delegate().needs_input_grad()
    }
    fn f(&mut self, mut input_: &mut VarKindList) -> VarKindList {
        let is_volatile = input_.iter().any(|v| v.is_volatile());
        {
            // do start graph stuff with f
            let f = self.delegate();
            let mut inner = f.access();
            if !is_volatile {
                inner.previous_functions = input_
                    .iter()
                    .map(|v| if let Some(grad_fn) = v.grad_fn() {
                             (RootKind::RootFunc(grad_fn), v.varid())
                         } else {
                             (RootKind::RootVar(v.clone()), v.varid())
                         })
                    .collect();
                inner.needs_input_grad = input_.iter().map(|v| v.requires_grad()).collect();
                inner.requires_grad = inner.needs_input_grad.iter().any(|v| *v);
            }
        }
        let v;
        {
            let mut input_tensors = input_.iter_mut().map(|v| v.data()).collect();
            v = self.forward(&mut input_tensors);
        }
        let f = self.delegate();
        let mut inner = f.access();
        let output = if is_volatile {
            let args = VariableArgs::build().volatile(true).done();
            v.into_iter().map(|t| VarKind::new_args(t, &args)).collect()
        } else {
            let args = VariableArgs::build()
                .creator(Some(f.clone()))
                .requires_grad(inner.requires_grad)
                .done();
            let mut output: VarKindList =
                v.into_iter().map(|t| VarKind::new_args(t, &args)).collect();
            for (i, v) in output.iter().enumerate() {
                inner.output_ids.insert(v.varid(), i);
            }
            if !inner.to_save.is_empty() {
                /* if a tensor was modified in place replace the old variable with the new one */
                let mut t2v = HashMap::new();
                for ref mut var in input_.iter_mut() {
                    t2v.insert(var.tid(), var.varid());
                }
                for ref mut var in &mut output.iter_mut() {
                    t2v.insert(var.tid(), var.varid());
                }
                for t in inner.to_save.iter() {
                    inner.saved_variables.push(t2v[t]);
                }
                inner.to_save.clear();
            };
            if !inner.non_differentiable.is_empty() {
                for ref mut var in &mut output {
                    if inner.non_differentiable.contains(&var.tid()) {
                        var.requires_nograd()
                    }
                }
                inner.non_differentiable.clear();
            };
            output
        };
        output
    }
}
