use autograd::{Function, FuncIntf, FuncDelegate, Variable, VarList};
use macros::*;

pub struct LinearF {
    delegate: Function,
}

impl LinearF {
    pub fn new() -> Self {
        LinearF { delegate: Function::new() }
    }
}
impl_func_delegate!(LinearF);

impl FuncIntf for LinearF {
    fn forward<T>(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
        panic!("implement");
        input.clone()
    }
    fn backward<T>(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
        panic!("implement");
        input.clone()
    }
}
