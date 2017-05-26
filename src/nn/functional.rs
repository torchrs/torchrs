//use torchrs::nn::functional::{max_pool2d, relu, conv2d, dropout, dropout2d, linear, log_softmax};

use autograd::variable::Variable;

#[builder(pattern="owned")]
#[derive(Builder)]
pub struct MaxPoolFArgs {
    #[builder(default="1")]
    stride: u32,
    #[builder(default="0")]
    padding: u32,
    #[builder(default="1")]
    dilation: u32,
    #[builder(default="false")]
    ceil_mode: bool,
}

impl Default for MaxPoolFArgs {
    fn default() -> Self {
        MaxPoolFArgsBuilder::default().build().unwrap()
    }
}

pub fn max_pool2d<T>(input: &Variable<T>,
                     kernel_size: (u32, u32),
                     args: MaxPoolFArgs)
                     -> Variable<T> {
    input.clone()
}

#[builder(pattern="owned")]
#[derive(Builder)]
pub struct DropoutFArgs {
    #[builder(default="0.5")]
    p: f32,
    #[builder(default="false")]
    training: bool,
    #[builder(default="false")]
    inplace: bool,
}

impl Default for DropoutFArgs {
    fn default() -> Self {
        DropoutFArgsBuilder::default().build().unwrap()
    }
}

pub fn dropout<T>(input: &Variable<T>, args: DropoutFArgs) -> Variable<T> {
    input.clone()
}

pub fn dropout2d<T>(input: &Variable<T>, args: DropoutFArgs) -> Variable<T> {
    input.clone()
}


#[builder(pattern="owned")]
#[derive(Builder)]
pub struct Conv2dFArgs<T:Default> {
    #[builder(default="None")]
    bias: Option<Variable<T>>,
    #[builder(default="vec!(1, 1)")]
    stride: Vec<u32>,
    #[builder(default="vec![0, 0]")]
    padding: Vec<u32>,
    #[builder(default="vec![1, 1]")]
    dilation: Vec<u32>,
    #[builder(default="1")]
    groups: u32,
}

impl<T:Default> Default for Conv2dFArgs<T> {
    fn default() -> Self {
        Conv2dFArgsBuilder::default().build().unwrap()
    }
}

pub fn conv2d<T:Default>(input: &Variable<T>, weight: &Variable<T>, args: Conv2dFArgs<T>) -> Variable<T> {
    //let convf = ConvNd::new(args.stride, args.padding, args.dilation, false, vec![0, 0],  args.groups));
    //convf.f(input, weight, args.bias)
    input.clone()
}

pub fn relu<T>(input: &Variable<T>, inplace: bool) -> Variable<T> {
    input.clone()
}

pub fn log_softmax<T>(input: &Variable<T>) -> Variable<T> {
    input.clone()
}
