use autograd::{Function, FuncIntf, FuncDelegate, Variable, VarList};

pub struct Dropout1d {
    delegate: Function,
    args: DropoutArgs,
}

pub struct Dropout2d {
    delegate: Function,
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

impl Dropout1d {
    pub fn new(args: &DropoutArgs) -> Self {
        Dropout1d {
            delegate: Function::new(),
            args: args.clone(),
        }
    }
}
impl_func_delegate!(Dropout1d);

impl Dropout2d {
    pub fn new(args: &DropoutArgs) -> Self {
        Dropout2d {
            delegate: Function::new(),
            args: args.clone(),
        }
    }
}
impl_func_delegate!(Dropout2d);

impl FuncIntf for Dropout1d {
    fn forward<T>(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
        unimplemented!()
    }
    fn backward<T>(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
        unimplemented!()
    }
}

impl FuncIntf for Dropout2d {
    fn forward<T>(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
        unimplemented!()
    }
    fn backward<T>(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
        unimplemented!()
    }
}
