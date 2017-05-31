use autograd::{Function, FuncIntf, FuncIntfX, FuncDelegate, Variable, VarId};
use macros::*;
use tensor::{RefTensorList, TensorList};

pub struct LogSoftmax {
    delegate: Function,
}

impl LogSoftmax {
    pub fn new() -> Self {
        LogSoftmax { delegate: Function::new() }
    }
}
impl_func_delegate!(LogSoftmax);

impl FuncIntf for LogSoftmax {
    fn forward<T>(&mut self, mut input: &RefTensorList<T>) -> TensorList<T> {
        unimplemented!()
    }
    fn backward<T>(&mut self, mut input: &RefTensorList<T>) -> TensorList<T> {
        unimplemented!()
    }
}

#[builder(pattern="owned")]
#[derive(Builder, Clone)]
pub struct NLLLossArgs {
    #[builder(default="false")]
    pub size_average: bool,
    #[builder(default="None")]
    pub weight: Option<VarId>,
}

impl Default for NLLLossArgs {
    fn default() -> Self {
        NLLLossArgsBuilder::default().build().unwrap()
    }
}

pub struct NLLLoss {
    delegate: Function,
    args: NLLLossArgs,
}

impl NLLLoss {
    pub fn new(args: &NLLLossArgs) -> Self {
        NLLLoss {
            delegate: Function::new(),
            args: args.clone(),
        }
    }
}

impl FuncDelegate for NLLLoss {
    fn delegate(&mut self) -> &mut Function {
        &mut self.delegate
    }
}

impl FuncIntfX for NLLLoss {
    fn forwardx<T>(&mut self,
                   mut input: &RefTensorList<T>,
                   target: &RefTensorList<i64>)
                   -> TensorList<T> {
        unimplemented!()
    }
    fn backwardx<T>(&mut self,
                    mut input: &RefTensorList<T>,
                    target: &RefTensorList<i64>)
                    -> TensorList<T> {
        unimplemented!()
    }
}
