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


pub trait FuncIntf<T> {
    fn delegate(&mut self) -> &mut Function<T>;
    fn forward(&mut self, input: &mut Vec<Variable<T>>) -> Vec<Variable<T>>;
    fn f(&mut self, mut input: &mut Vec<Variable<T>>) -> Vec<Variable<T>> {
        {
            let f = self.delegate();
        }
        // do start graph stuff with f
        let v = self.forward(&mut input);
        // do end graph stuff
        v
    }
}
