use autograd::{Function, FuncIntf, Variable, VarList};

pub struct LogSoftmax<T> {
    delegate: Function<T>,
}

impl<T> LogSoftmax<T> {
    pub fn new() -> Self {
        LogSoftmax { delegate: Function::new() }
    }
}

impl<T> FuncIntf<T> for LogSoftmax<T> {
    fn delegate(&mut self) -> &mut Function<T> {
        &mut self.delegate
    }
    fn forward(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
        input.clone()
    }
    fn backward(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
        input.clone()
    }
}
