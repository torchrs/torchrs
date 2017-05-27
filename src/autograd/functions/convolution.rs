use autograd::{Function, FuncIntf, Variable, VarList};

#[builder(pattern="owned")]
#[derive(Builder, Clone)]
pub struct ConvNdArgs {
    stride: Vec<u32>,
    padding: Vec<u32>,
    dilation: Vec<u32>,
    #[builder(default="false")]
    transposed: bool,
    output_padding: Vec<u32>,
    groups: u32,
    #[builder(default="false")]
    benchmark: bool,
    #[builder(default="false")]
    cudnn_enabled: bool,
}

#[builder(pattern="owned")]
#[derive(Builder)]
pub struct Conv1dFArgs<T: Default> {
    #[builder(default="None")]
    bias: Option<Variable<T>>,
    #[builder(default="vec![1]")]
    stride: Vec<u32>,
    #[builder(default="vec![0]")]
    padding: Vec<u32>,
    #[builder(default="vec![1]")]
    dilation: Vec<u32>,
    #[builder(default="1")]
    groups: u32,
}

impl<T: Default> Default for Conv1dFArgs<T> {
    fn default() -> Self {
        Conv1dFArgsBuilder::default().build().unwrap()
    }
}

#[builder(pattern="owned")]
#[derive(Builder)]
pub struct Conv2dFArgs<T: Default> {
    #[builder(default="None")]
    pub bias: Option<Variable<T>>,
    #[builder(default="vec![1, 1]")]
    pub stride: Vec<u32>,
    #[builder(default="vec![0, 0]")]
    pub padding: Vec<u32>,
    #[builder(default="vec![1, 1]")]
    pub dilation: Vec<u32>,
    #[builder(default="1")]
    pub groups: u32,
}

impl<T: Default + Copy> Default for Conv2dFArgs<T> {
    fn default() -> Self {
        Conv2dFArgsBuilder::default().build().unwrap()
    }
}

#[builder(pattern="owned")]
#[derive(Builder)]
pub struct Conv3dFArgs<T: Default> {
    #[builder(default="None")]
    bias: Option<Variable<T>>,
    #[builder(default="vec![1, 1, 1]")]
    stride: Vec<u32>,
    #[builder(default="vec![0, 0, 0]")]
    padding: Vec<u32>,
    #[builder(default="vec![1, 1, 1]")]
    dilation: Vec<u32>,
    #[builder(default="1")]
    groups: u32,
}

impl<T: Default> Default for Conv3dFArgs<T> {
    fn default() -> Self {
        Conv3dFArgsBuilder::default().build().unwrap()
    }
}

impl<'a, T: Default> From<&'a mut Conv2dFArgs<T>> for ConvNdArgs {
    fn from(input: &'a mut Conv2dFArgs<T>) -> Self {
        ConvNdArgsBuilder::default()
            .stride(input.stride.clone())
            .padding(input.padding.clone())
            .dilation(input.dilation.clone())
            .groups(input.groups)
            .output_padding(vec![0, 0])
            .build()
            .unwrap()
    }
}

pub struct ConvNd<T> {
    delegate: Function<T>,
    args: ConvNdArgs,
}

impl<T> ConvNd<T> {
    pub fn new(args: &ConvNdArgs) -> Self {
        ConvNd {
            delegate: Function::new(),
            args: args.clone(),
        }
    }
}

impl<T> FuncIntf<T> for ConvNd<T> {
    fn delegate(&mut self) -> &mut Function<T> {
        &mut self.delegate
    }
    fn forward(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
        // run native code here
        input.clone()
    }
    fn backward(&mut self, mut input: &mut VarList<T>) -> VarList<T> {
        // run native code here
        input.clone()
    }
}
