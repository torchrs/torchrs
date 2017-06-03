use autograd::{Function, FuncIntf, FuncDelegate, Variable, FIWrap};
use tensor::{RefTensorList, TensorKindList};
use ::*;

#[derive(Clone)]
pub struct Dropout1d {
    delegate: Function,
    args: DropoutArgs,
}

#[derive(Clone)]
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
    pub fn new(args: &DropoutArgs) -> FIWrap<Self> {
        FIWrap::new(Dropout1d {
                        delegate: Function::new(),
                        args: args.clone(),
                    })
    }
}
impl_func_delegate!(Dropout1d);

impl Dropout2d {
    pub fn new(args: &DropoutArgs) -> FIWrap<Self> {
        FIWrap::new(Dropout2d {
                        delegate: Function::new(),
                        args: args.clone(),
                    })
    }
}
impl_func_delegate!(Dropout2d);

impl FuncIntf for Dropout1d {
    fn forward(&mut self, mut input: &mut TensorKindList) -> TensorKindList {
        unimplemented!()
    }
    fn backward(&mut self, mut input: &mut TensorKindList) -> TensorKindList {
        unimplemented!()
    }
}

impl FuncIntf for Dropout2d {
    fn forward(&mut self, mut input: &mut TensorKindList) -> TensorKindList {
        unimplemented!()
    }
    fn backward(&mut self, mut input: &mut TensorKindList) -> TensorKindList {
        unimplemented!()
    }
}
