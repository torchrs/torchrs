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
    pub fn do_forward(&mut self,
                      func: &mut FuncIntf<T>,
                      mut args: &mut Vec<Variable<T>>)
                      -> Vec<Variable<T>> {
        // do start graph stuff
        let v = func.forward(&mut args);
        // do end graph stuff
        v
    }
}


pub trait FuncIntf<T> {
    fn delegate(&mut self) -> &mut Function<T>;
    fn forward(&mut self, input: &mut Vec<Variable<T>>) -> Vec<Variable<T>>;
    // Borrowing rules prevent this, but each FuncIntf implementer needs to implement
    // the following function:
    //    fn f(&mut self, input: &mut Vec<Variable<T>>) -> Vec<Variable<T>> {
    //    	  self.delegate().do_forward(&mut self, input)
    //    }
}
