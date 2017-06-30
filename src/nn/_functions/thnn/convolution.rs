use autograd::{Function, FuncIntf, FuncDelegate, Variable, FIWrap};
use tensor::{TensorKindList, OptTensorKindList};
use ::*;

#[builder(pattern="owned")]
#[derive(Builder, Clone)]
pub struct ConvNdArgs {
    stride: Vec<u32>,
    padding: Vec<u32>,
    dilation: Vec<u32>,
    kernel_size: Vec<usize>,
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
pub struct Conv1dFArgs {
    #[builder(default="vec![1]")]
    pub kernel_size: Vec<usize>,
    #[builder(default="vec![1]")]
    stride: Vec<u32>,
    #[builder(default="vec![0]")]
    padding: Vec<u32>,
    #[builder(default="vec![1]")]
    dilation: Vec<u32>,
    #[builder(default="1")]
    groups: u32,
}

impl Default for Conv1dFArgs {
    fn default() -> Self {
        Conv1dFArgsBuilder::default().build().unwrap()
    }
}

#[builder(pattern="owned")]
#[derive(Builder)]
pub struct Conv2dFArgs {
    #[builder(default="1")]
    pub in_features: usize,
    #[builder(default="1")]
    pub out_features: usize,
    #[builder(default="vec![1, 1]")]
    pub kernel_size: Vec<usize>,
    #[builder(default="vec![1, 1]")]
    pub stride: Vec<u32>,
    #[builder(default="vec![0, 0]")]
    pub padding: Vec<u32>,
    #[builder(default="vec![1, 1]")]
    pub dilation: Vec<u32>,
    #[builder(default="1")]
    pub groups: u32,
}

impl Default for Conv2dFArgs {
    fn default() -> Self {
        Conv2dFArgsBuilder::default().build().unwrap()
    }
}

#[builder(pattern="owned")]
#[derive(Builder)]
pub struct Conv3dFArgs {
    #[builder(default="vec![1, 1, 1]")]
    pub kernel_size: Vec<usize>,
    #[builder(default="vec![1, 1, 1]")]
    stride: Vec<u32>,
    #[builder(default="vec![0, 0, 0]")]
    padding: Vec<u32>,
    #[builder(default="vec![1, 1, 1]")]
    dilation: Vec<u32>,
    #[builder(default="1")]
    groups: u32,
}

impl Default for Conv3dFArgs {
    fn default() -> Self {
        Conv3dFArgsBuilder::default().build().unwrap()
    }
}

impl<'a> From<&'a mut Conv2dFArgs> for ConvNdArgs {
    fn from(input: &'a mut Conv2dFArgs) -> Self {
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

impl_func_args!(ConvNd, ConvNdArgs);

impl FuncIntf for ConvNd {
    fn forward(&mut self, mut input: &mut TensorKindList) -> TensorKindList {
        // run native code here
        unimplemented!()
    }
    fn backward(&mut self, mut input: &mut OptTensorKindList) -> OptTensorKindList {
        // run native code here
        unimplemented!()
    }
}
