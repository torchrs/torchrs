use std::rc::Rc;
use std::cell::RefCell;

use std::vec::Vec;
use autograd::variable::*;
use RcMut;

pub struct Function<T> {
    saved_variables: Vec<SavedVariable<T>>,
    next_functions: Vec<(RcMut<Function<T>>, usize)>,
}
impl<T> Function<T> {
    pub fn new() -> Self {
        Function {
            saved_variables: Vec::new(),
            next_functions: Vec::new(),
        }
    }
}

pub trait FuncDelegate<T> {
    fn delegate(&mut self) -> &mut Function<T>;
}

pub trait FuncIntf<T>: FuncDelegate<T> {
    fn forward(&mut self, input: &mut VarList<T>) -> VarList<T>;
    fn backward(&mut self, input: &mut VarList<T>) -> VarList<T>;
    fn f(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
        {
            // do start graph stuff with f
            let f = self.delegate();
        }
        let v = self.forward(&mut input);
        // do end graph stuff
        v
    }
}

pub trait FuncIntfX<T>: FuncDelegate<T> {
    fn forwardx(&mut self, input: &VarList<T>, target: &VarList<i64>) -> VarList<T>;
    fn backwardx(&mut self, input: &VarList<T>, target: &VarList<i64>) -> VarList<T>;
    fn fx(&mut self, input: &VarList<T>, target: &VarList<i64>) -> VarList<T> {
        {
            // do start graph stuff with f
            let f = self.delegate();
        }
        let v = self.forwardx(input, target);
        // do end graph stuff
        v
    }
}
