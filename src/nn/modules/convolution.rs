use nn::modules::module::*;
use nn::parameter::Parameter;
use autograd::variable::Variable;
use std::marker::PhantomData;

#[derive(ModParse)]
pub struct Conv2d<T: Default> {
    delegate: Module<T>,
    weight: Parameter<T>,
}

impl<T: Default> Conv2d<T> {
    pub fn build(in_channels: u32, out_channels: u32, kernel_size: u32) -> Conv2dArgsBuilder<T> {
        Conv2dArgsBuilder::default()
            .in_channels(in_channels)
            .out_channels(out_channels)
            .kernel_size(kernel_size)
    }
    pub fn new(args: Conv2dArgs<T>) -> Conv2d<T> {
        Conv2d {
            delegate: Module::new(),
            weight: Parameter::default(),
        }
    }
}

#[builder(pattern="owned")]
#[derive(Builder)]
pub struct Conv2dArgs<T: Default> {
    in_channels: u32,
    out_channels: u32,
    kernel_size: u32,
    #[builder(default="1")]
    stride: u32,
    #[builder(default="0")]
    padding: u32,
    #[builder(default="1")]
    dilation: u32,
    #[builder(default="1")]
    groups: u32,
    #[builder(default="true")]
    bias: bool,
    #[builder(default="PhantomData")]
    phantom: PhantomData<T>,
}
impl<T: Default> Conv2dArgsBuilder<T> {
    pub fn done(self) -> Conv2d<T> {
        let args = self.build().unwrap();
        Conv2d::new(args)
    }
}

impl<T: Default> ModIntf<T> for Conv2d<T> {
    fn delegate(&mut self) -> &mut Module<T> {
        &mut self.delegate
    }
    fn forward(&mut self, input: &mut Variable<T>) -> Variable<T> {
        input.clone()
    }
    fn forwardv(&mut self, input: &mut Vec<Variable<T>>) -> Vec<Variable<T>> {
        panic!("not valid");
    }
}
