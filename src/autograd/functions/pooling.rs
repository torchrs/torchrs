use autograd::{Function, FuncIntf, FuncDelegate, Variable};
use tensor::{RefTensorList, TensorList};

#[builder(pattern="owned")]
#[derive(Builder, Clone)]
pub struct MaxPoolFArgs {
    // just for code re-use
    #[builder(default="vec![1]")]
    pub kernel_size: Vec<u32>,
    pub stride: Vec<u32>,
    pub padding: Vec<u32>,
    pub dilation: Vec<u32>,
    #[builder(default="false")]
    pub ceil_mode: bool,
    #[builder(default="false")]
    pub return_indices: bool,
}

pub struct MaxPool1dArgs {
    pub v: MaxPoolFArgs,
}
pub struct MaxPool2dArgs {
    pub v: MaxPoolFArgs,
}
pub struct MaxPool3dArgs {
    pub v: MaxPoolFArgs,
}

impl Default for MaxPool1dArgs {
    fn default() -> Self {
        let args = MaxPoolFArgsBuilder::default()
            .stride(vec![1])
            .padding(vec![0])
            .dilation(vec![1])
            .build()
            .unwrap();
        MaxPool1dArgs { v: args }
    }
}

impl Default for MaxPool2dArgs {
    fn default() -> Self {
        let args = MaxPoolFArgsBuilder::default()
            .stride(vec![1, 1])
            .padding(vec![0, 0])
            .dilation(vec![1, 1])
            .build()
            .unwrap();
        MaxPool2dArgs { v: args }
    }
}

impl Default for MaxPool3dArgs {
    fn default() -> Self {
        let args = MaxPoolFArgsBuilder::default()
            .stride(vec![1, 1, 1])
            .padding(vec![0, 0, 0])
            .dilation(vec![1, 1, 1])
            .build()
            .unwrap();
        MaxPool3dArgs { v: args }
    }
}


pub struct MaxPool2d {
    delegate: Function,
    args: MaxPoolFArgs,
}

impl MaxPool2d {
    pub fn new(args: &MaxPoolFArgs) -> Self {
        MaxPool2d {
            delegate: Function::new(),
            args: args.clone(),
        }
    }
    fn forward_apply<T>(&mut self, input: &RefTensorList<T>) -> TensorList<T> {
        unimplemented!()
    }
    fn backward_apply<T>(&mut self, input: &RefTensorList<T>) -> TensorList<T> {
        unimplemented!()
    }
}
impl_func_delegate!(MaxPool2d);

impl FuncIntf for MaxPool2d {
    fn forward<T>(&mut self, mut input: &RefTensorList<T>) -> TensorList<T> {
        self.forward_apply(input)
    }
    fn backward<T>(&mut self, mut input: &RefTensorList<T>) -> TensorList<T> {
        self.backward_apply(input)
    }
}
