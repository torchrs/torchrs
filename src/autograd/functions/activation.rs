use autograd::{Function, FuncIntf, FuncDelegate, Variable, VarList};

pub struct Threshold<T> {
    delegate: Function<T>,
    threshold: f32,
    value: f32,
    inplace: bool,
}

impl<T> Threshold<T> {
    pub fn new(threshold: f32, value: f32, inplace: bool) -> Self {
        Threshold {
            delegate: Function::new(),
            threshold: threshold,
            value: value,
            inplace: inplace,
        }
    }
}
impl_func_delegate!(Threshold);

impl<T> FuncIntf<T> for Threshold<T> {
    fn forward(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
        input.clone()
    }
    fn backward(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
        input.clone()
    }
}
