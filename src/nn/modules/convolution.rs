use nn::modules::module::*;
use nn::parameter::Parameter;
use num::Num;

#[derive(ModParse)]
pub struct Conv2d<'a, T:'a + Num> {
    delegate: Module<'a, T>,
    weight: Parameter<'a, T>,
}

impl<'a, T: 'a + Num> Conv2d<'a, T> {
    pub fn build(in_channels: u32, out_channels: u32, kernel_size: u32) -> Conv2dArgsBuilder {
        Conv2dArgsBuilder::default()
            .in_channels(in_channels)
            .out_channels(out_channels)
            .kernel_size(kernel_size)
    }
    pub fn new(args: Conv2dArgs) -> Conv2d<'a, T> {
        Conv2d {
            delegate: Module::new(),
            weight: Parameter::default(),
        }
    }
}

#[builder(pattern="owned")]
#[derive(Builder)]
pub struct Conv2dArgs {
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
}
impl Conv2dArgsBuilder {
    pub fn done<'a, T: 'a + Num>(self) -> Conv2d<'a, T> {
        let args = self.build().unwrap();
        Conv2d::new(args)
    }
}
