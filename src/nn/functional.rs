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

pub fn relu<T>(input: &Variable<T>, inplace: bool) -> Variable<T> {
    input.clone()
}

pub fn log_softmax<T>(input: &Variable<T>) -> Variable<T> {
    input.clone()
}
