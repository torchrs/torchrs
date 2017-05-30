use std::rc::Rc;
use std::cell::RefCell;

use std::vec::Vec;
use autograd::variable::*;
use RcMut;

pub struct Function {
    saved_variables: Vec<SavedVarKind>,
    next_functions: Vec<(RcMut<Function>, usize)>,
}
impl Function {
    pub fn new() -> Self {
        Function {
            saved_variables: Vec::new(),
            next_functions: Vec::new(),
        }
    }
}

pub trait FuncDelegate {
    fn delegate(&mut self) -> &mut Function;
}

pub trait FuncIntf: FuncDelegate {
    fn forward<T>(&mut self, input: &mut VarList<T>) -> VarList<T>;
    fn backward<T>(&mut self, input: &mut VarList<T>) -> VarList<T>;
    fn f<T>(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
        {
            // do start graph stuff with f
            let f = self.delegate();
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
