use std::rc::Rc;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::vec::Vec;
use autograd::variable::*;
use RcMut;

thread_local! {
    pub static FUNC_TABLE: RefCell<VecDeque<FuncImpl>> = RefCell::new(VecDeque::new());
}

pub type FuncId = i32;

#[derive(Default)]
pub struct FuncImpl {
    previous_functions: Vec<Function>,
    saved_variables: Vec<VarId>,
    needs_input_grad: Vec<VarId>,
    requires_grad: bool,
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
    fn forward<T>(&mut self, input: &mut VarList<T>) -> VarList<T>;
    fn backward<T>(&mut self, input: &mut VarList<T>) -> VarList<T>;
    fn f<T>(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
        let is_volatile = input.iter().any(|v| v.is_volatile());
        {
            // do start graph stuff with f
            let mut f = self.delegate().access();
            if !is_volatile {
                f.needs_input_grad = input
                    .iter()
                    .filter_map(|v| if v.requires_grad() { Some(v.id) } else { None })
                    .collect();
                f.requires_grad = f.needs_input_grad.len() != 0;
                //f.previous_functions = input.iter().filter_map(|v| { let grad_fn if Some(v.value.borrow()))
            }
        }
        let v = self.forward(&mut input);
        // do end graph stuff
        v
    }
}

pub trait FuncIntfX: FuncDelegate {
    fn forwardx<T>(&mut self, input: &VarList<T>, target: &VarList<i64>) -> VarList<T>;
    fn backwardx<T>(&mut self, input: &VarList<T>, target: &VarList<i64>) -> VarList<T>;
    fn fx<T>(&mut self, input: &VarList<T>, target: &VarList<i64>) -> VarList<T> {
        {
            // do start graph stuff with f
            let f = self.delegate();
        }
        let v = self.forwardx(input, target);
        // do end graph stuff
        v
    }
}
