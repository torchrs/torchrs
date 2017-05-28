use autograd::{Function, FuncIntf, FuncDelegate, Variable, VarList};

pub struct Dropout1d<T> {
    delegate: Function<T>,
    args: DropoutArgs,
}

pub struct Dropout2d<T> {
    delegate: Function<T>,
    args: DropoutArgs,
}

#[builder(pattern="owned")]
#[derive(Builder, Clone)]
pub struct DropoutArgs {
    #[builder(default="0.5")]
    pub p: f32,
    #[builder(default="false")]
    pub training: bool,
    #[builder(default="false")]
    pub inplace: bool,
}

impl Default for DropoutArgs {
    fn default() -> Self {
        DropoutArgsBuilder::default().build().unwrap()
    }
}

impl<T> Dropout1d<T> {
    pub fn new(args: &DropoutArgs) -> Self {
        Dropout1d {
            delegate: Function::new(),
            args: args.clone(),
        }
    }
}
impl_func_delegate!(Dropout1d);

impl<T> Dropout2d<T> {
    pub fn new(args: &DropoutArgs) -> Self {
        Dropout2d {
            delegate: Function::new(),
            args: args.clone(),
        }
    }
}
impl_func_delegate!(Dropout2d);

impl<T> FuncIntf<T> for Dropout1d<T> {
    fn forward(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
        input.clone()
    }
    fn backward(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
        input.clone()
    }
}

impl<T> FuncIntf<T> for Dropout2d<T> {
    fn forward(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
        input.clone()
    }
    fn backward(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
        input.clone()
    }
}
