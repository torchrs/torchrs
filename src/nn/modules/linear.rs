use nn::modules::module::*;
use nn::parameter::Parameter;
use autograd::variable::Variable;


#[derive(ModParse)]
pub struct Linear<'a, T: 'a> {
    delegate: Module<'a, T>,
    #[ignore]
    in_features: u32,
    #[ignore]
    out_features: u32,
    weight: Parameter<'a, T>,
    bias: Option<Parameter<'a, T>>,
}

impl<'a, T> Linear<'a, T> {
    pub fn build(in_features: u32, out_features: u32) -> LinearArgsBuilder {
        LinearArgsBuilder::default()
            .in_features(in_features)
            .out_features(out_features)
    }
    pub fn new(args: LinearArgs) -> Linear<'a, T> {
        let mut t = Linear {
            delegate: Module::new(),
            in_features: args.in_features,
            out_features: args.out_features,
            weight: Parameter::default(),
            bias: None,
        };
        t.init_module();
        t
    }
}
#[builder(pattern="owned")]
#[derive(Builder)]
pub struct LinearArgs {
    in_features: u32,
    out_features: u32,
    #[builder(default="true")]
    bias: bool,
}
impl LinearArgsBuilder {
    pub fn done<'a, T: 'a>(self) -> Linear<'a, T> {
        let args = self.build().unwrap();
        Linear::new(args)
    }
}
impl<'a, T> ModIntf<'a, T> for Linear<'a, T> {
    fn delegate(&mut self) -> &mut Module<'a, T> {
        &mut self.delegate
    }
    fn forward<'b>(&'a mut self, input: &'b Vec<Variable<'a, T>>) -> Vec<Variable<'a, T>> {
        input.clone()
    }
}
