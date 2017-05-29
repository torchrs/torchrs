use nn::{Module, ModuleStruct, ModDelegate, ModIntf, Parameter};
use autograd::Variable;
use std::marker::PhantomData;

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
impl_mod_delegate!(Conv2d);

impl<T: Default> ModIntf<T> for Conv2d<T> {
    fn forward(&mut self, input: &mut Variable<T>) -> Variable<T> {
        input.clone()
    }
}
