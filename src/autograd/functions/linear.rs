use autograd::{Function, FuncIntf, FuncDelegate, Variable, VarList};
use macros::*;

pub struct LinearF<T> {
    delegate: Function<T>,
}

impl<T> LinearF<T> {
    pub fn new() -> Self {
        LinearF {
            delegate: Function::new(),
        }
    }
}
impl_func_delegate!(LinearF);

impl<T> FuncIntf<T> for LinearF<T> {
    fn forward(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
    	panic!("implement");
        input.clone()
    }
    fn backward(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
    	panic!("implement");
        input.clone()
    }
}
