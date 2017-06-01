use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::vec::Vec;
use autograd::variable::*;
use tensor::*;

thread_local! {
    pub static FUNC_TABLE: RefCell<VecDeque<FuncImpl>> = RefCell::new(VecDeque::new());
}

pub type FuncId = i32;

#[derive(Default)]
pub struct FuncImpl {
    previous_functions: Vec<Function>,
    saved_variables: Vec<VarId>,
    needs_input_grad: Vec<VarId>,
    non_differentiable: Vec<TensorId>,
    to_save: Vec<TensorId>,
    requires_grad: bool,
}

impl FuncImpl {
    fn _call_hooks<T>(&self, input: &TensorList<T>, output: &RefTensorList<T>) {
        unimplemented!();
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
    fn access(&self) -> &mut FuncImpl {
        let vecp = FUNC_TABLE.with(|f| f.as_ptr());
        let vec = unsafe { &mut *vecp };
        &mut vec[self.id as usize]
    }

    pub fn from(id: FuncId) -> Self {
        Function { id: id }
    }
}

pub trait FuncDelegate {
    fn delegate(&mut self) -> &mut Function;
}

pub trait FuncIntf: FuncDelegate {
    fn forward<'a, T>(&mut self, input: &RefTensorList<'a, T>) -> TensorList<T>;
    fn backward<'a, T>(&mut self, input: &RefTensorList<'a, T>) -> TensorList<T>;
    fn f<T>(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
        let is_volatile = input.iter().any(|v| v.is_volatile());
        {
            // do start graph stuff with f
            let mut f = self.delegate().access();
            if !is_volatile {
                f.previous_functions = input.iter().filter_map(|v| v.grad_fn()).collect();
                f.needs_input_grad = input
                    .iter()
                    .filter_map(|v| if v.requires_grad() { Some(v.id) } else { None })
                    .collect();
                f.requires_grad = f.needs_input_grad.len() != 0;
            }
        }
        let v;
        {
            let input_tensors = input.iter_mut().map(|v| v.data()).collect();
            v = self.forward(&input_tensors);
        }
        let f = self.delegate();
        let mut fi = f.access();
        let output = if is_volatile {
            let args = VariableArgs::build().volatile(true).done();
            v.into_iter()
                .map(|t| Variable::new_args(t, &args))
                .collect()
        } else {
            let args = VariableArgs::build()
                .creator(Some(f.clone()))
                .requires_grad(fi.requires_grad)
                .done();
            let mut output: VarList<T> = v.into_iter()
                .map(|t| Variable::new_args(t, &args))
                .collect();
            if !fi.to_save.is_empty() {
                /* if a tensor was modified in place replace the old variable with the new one */
                let mut t2v = HashMap::new();
                for ref var in input.iter() {
                    t2v.insert(var.data_borrow().id, var.id);
                }
                for ref mut var in &mut output.iter() {
                    t2v.insert(var.data_borrow().id, var.id);
                }
                for t in fi.to_save.iter() {
                    fi.saved_variables.push(t2v[t]);
                }
                fi.to_save.clear();
            };
            if !fi.non_differentiable.is_empty() {
                for ref mut var in &mut output {
                    if fi.non_differentiable.contains(&var.data().id) {
                        var.requires_nograd()
                    }
                }
                fi.non_differentiable.clear();
            };
            output
        };
        output
    }
    fn saved_tensors<'a, T: 'a>(&mut self) -> TensorList<T> {
        // XXX see if we can't avoid the clone
        self.delegate()
            .access()
            .saved_variables
            .iter()
            .map(|v| Variable::<T>::from(v).data().clone())
            .collect()
    }
    fn save_for_backward<T>(&mut self, input: &RefTensorList<T>) {
        self.delegate().access().to_save = input.iter().map(|t| t.id).collect();
    }
    fn _do_backward<'a, T>(&mut self,
                           grad_output: &RefTensorList<'a, T>,
                           retain_variables: bool)
                           -> TensorList<T> {
        if self.delegate().access().saved_variables.is_empty() {
            panic!("Trying to backward through the graph second \
                    time, but the buffers have already been freed. Please \
                    specify retain_variables=True when calling backward for \
                    the first time.");
        };
        let grad_input = self.backward(grad_output);
        let fi = self.delegate().access();
        fi._call_hooks(&grad_input, grad_output);
        if !retain_variables {
            fi.saved_variables.clear();
        };
        grad_input
    }
}

pub trait FuncIntfKind: FuncDelegate {
    fn forwardx<'a>(&mut self, input: &RefTensorKindList<'a>) -> TensorKindList;
    fn backwardx<'a>(&mut self, input: &RefTensorKindList<'a>) -> TensorKindList;
    fn fx(&mut self, input: &mut VarKindList) -> VarKindList {
        unimplemented!();

        /*
        {
            // do start graph stuff with f
            let f = self.delegate();
        }
        let inputTs = input.iter_mut().map(|v| v.data()).collect();
        let targetTs = target.iter_mut().map(|v| v.data()).collect();

        let v = self.forwardx(&inputTs, &targetTs);
        let f = self.delegate();
        let fi = f.access();
        let args = VariableArgs::build()
            .creator(Some(f.clone()))
            .requires_grad(fi.requires_grad)
            .done();
        let output = v.into_iter()
            .map(|t| Variable::new_args(t, &args))
            .collect();
        output
*/
    }
}
