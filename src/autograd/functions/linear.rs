use autograd::{Function, FuncIntf, FuncDelegate, Variable};
use macros::*;
use tensor::{RefTensorList, TensorList};

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
    fn forward<T>(&mut self, mut input: &RefTensorList<T>) -> TensorList<T> {
        unimplemented!()
    }
    fn backward<T>(&mut self, mut input: &RefTensorList<T>) -> TensorList<T> {
        unimplemented!()
    }
}
