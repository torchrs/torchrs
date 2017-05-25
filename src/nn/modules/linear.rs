use nn::modules::module::*;
use nn::parameter::Parameter;
use autograd::variable::Variable;


#[derive(ModParse)]
pub struct Linear<T> {
    delegate: Module<T>,
    #[ignore]
    in_features: u32,
    #[ignore]
    out_features: u32,
    weight: Parameter<T>,
    bias: Option<Parameter<T>>,
}

impl<T> Linear<T> {
    pub fn build(in_features: u32, out_features: u32) -> LinearArgsBuilder {
        LinearArgsBuilder::default()
            .in_features(in_features)
            .out_features(out_features)
    }
    pub fn new(args: LinearArgs) -> Linear<T> {
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
    pub fn done<T>(self) -> Linear<T> {
        let args = self.build().unwrap();
        Linear::new(args)
    }
}
impl<T> ModIntf<T> for Linear<T> {
    fn delegate(&mut self) -> &mut Module<T> {
        &mut self.delegate
    }
    fn forward(&mut self, input: &mut Vec<Variable<T>>) -> Vec<Variable<T>> {
        input.clone()
    }
}
