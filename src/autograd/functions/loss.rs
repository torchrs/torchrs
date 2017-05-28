use autograd::{Function, FuncIntf, FuncIntfX, FuncDelegate, Variable, VarList};

pub struct LogSoftmax<T> {
    delegate: Function<T>,
}

impl<T> LogSoftmax<T> {
    pub fn new() -> Self {
        LogSoftmax { delegate: Function::new() }
    }
}
impl_func_delegate!(LogSoftmax);

impl<T> FuncIntf<T> for LogSoftmax<T> {
    fn forward(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
        input.clone()
    }
    fn backward(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
        input.clone()
    }
}

#[builder(pattern="owned")]
#[derive(Builder, Clone)]
pub struct NLLLossArgs<T: Default + Copy> {
    #[builder(default="None")]
    pub weight: Option<Variable<T>>,
    #[builder(default="false")]
    pub size_average: bool,
}

impl<T: Default + Copy> Default for NLLLossArgs<T> {
    fn default() -> Self {
        NLLLossArgsBuilder::default().build().unwrap()
    }
}

pub struct NLLLoss<T: Default + Copy> {
    delegate: Function<T>,
    args: NLLLossArgs<T>,
}

impl<T: Default + Copy> NLLLoss<T> {
    pub fn new(args: &NLLLossArgs<T>) -> Self {
        NLLLoss {
            delegate: Function::new(),
            args: args.clone(),
        }
    }
}

impl<T: Default + Copy> FuncDelegate<T> for NLLLoss<T> {
    fn delegate(&mut self) -> &mut Function<T> {
        &mut self.delegate
    }
}

impl<T: Default + Copy> FuncIntfX<T> for NLLLoss<T> {
    fn forwardx(&mut self, mut input: &VarList<T>, target: &VarList<i64>) -> VarList<T> {
        input.clone()
    }
    fn backwardx(&mut self, mut input: &VarList<T>, target: &VarList<i64>) -> VarList<T> {
        input.clone()
    }
}
