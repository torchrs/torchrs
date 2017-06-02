use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::vec::Vec;
use autograd::variable::*;
use tensor::*;
use ::*;

thread_local! {
    pub static FUNC_TABLE: RefCell<VecDeque<FuncImpl>> = RefCell::new(VecDeque::new());
//    pub static FUNC_INTF_TABLE: RefCell<HashMap<FuncId, &'static FuncIntf>> = RefCell::new(HashMap::new());
}
pub type FuncId = i32;


struct FuncStub {
    delegate: Function,
}
impl_func_delegate!(FuncStub);
impl FuncIntf for FuncStub {
    fn forward(&mut self, input: &TensorKindList) -> TensorKindList {
        unreachable!()
    }
    fn backward(&mut self, input: &TensorKindList) -> TensorKindList {
        unreachable!()
    }
}
impl FuncStub {
    fn new() -> Self {
        FuncStub { delegate: Function::default() }
    }
}

pub struct FuncImpl {
    previous_functions: Vec<Function>,
    saved_variables: Vec<VarId>,
    needs_input_grad: Vec<VarId>,
    non_differentiable: Vec<TensorId>,
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
            to_save: Vec::new(),
            requires_grad: false,
            owner: RcMutNew(FuncStub::new()),
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
    id: FuncId,
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
    pub fn save_for_backward<T>(&mut self, input: &RefTensorList<T>) {
        self.access().to_save = input.iter().map(|t| t.id).collect();
    }
    pub fn _do_backward(&mut self,
                        grad_output: &TensorKindList,
                        retain_variables: bool)
                        -> TensorKindList {
        let inner = self.access();
        if inner.saved_variables.is_empty() {
            panic!("Trying to backward through the graph second \
                    time, but the buffers have already been freed. Please \
                    specify retain_variables=True when calling backward for \
                    the first time.");
        };
        let grad_input = inner.owner.borrow_mut().backward(grad_output);
        inner._call_hooks(&grad_input, grad_output);
        if !retain_variables {
            inner.saved_variables.clear();
        };
        grad_input.into()
    }
}

pub trait FuncDelegate {
    fn delegate(&mut self) -> &mut Function;
}

pub trait FuncIntf: FuncDelegate {
    fn forward(&mut self, input: &TensorKindList) -> TensorKindList;
    fn backward(&mut self, input: &TensorKindList) -> TensorKindList;
    fn f(&mut self, mut input_: &mut VarKindList) -> VarKindList {
        let is_volatile = input_.iter().any(|v| v.is_volatile());
        {
            // do start graph stuff with f
            let mut f = self.delegate();
            let mut inner = f.access();
            if !is_volatile {
                inner.previous_functions = input_.iter().filter_map(|v| v.grad_fn()).collect();
                inner.needs_input_grad = input_
                    .iter()
                    .filter_map(|v| if v.requires_grad() {
                                    Some(v.varid())
                                } else {
                                    None
                                })
                    .collect();
                inner.requires_grad = inner.needs_input_grad.len() != 0;
            }
        }
        let v;
        {
            let input_tensors = input_.iter_mut().map(|v| v.data()).collect();
            v = self.forward(&input_tensors);
        }
        let f = self.delegate();
        let mut fi = f.access();
        let output = if is_volatile {
            let args = VariableArgs::build().volatile(true).done();
            v.into_iter().map(|t| VarKind::new_args(t, &args)).collect()
        } else {
            let args = VariableArgs::build()
                .creator(Some(f.clone()))
                .requires_grad(fi.requires_grad)
                .done();
            let mut output: VarKindList =
                v.into_iter().map(|t| VarKind::new_args(t, &args)).collect();
            if !fi.to_save.is_empty() {
                /* if a tensor was modified in place replace the old variable with the new one */
                let mut t2v = HashMap::new();
                for ref mut var in input_.iter_mut() {
                    t2v.insert(var.tid(), var.varid());
                }
                for ref mut var in &mut output.iter_mut() {
                    t2v.insert(var.tid(), var.varid());
                }
                for t in fi.to_save.iter() {
                    fi.saved_variables.push(t2v[t]);
                }
                fi.to_save.clear();
            };
            if !fi.non_differentiable.is_empty() {
                for ref mut var in &mut output {
                    if fi.non_differentiable.contains(&var.tid()) {
                        var.requires_nograd()
                    }
                }
                fi.non_differentiable.clear();
            };
            output
        };
        output
    }
}
