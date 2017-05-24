use nn::modules::module::*;
use nn::parameter::Parameter;

#[derive(ModParse)]
pub struct Linear<'a> {
    delegate: Module<'a>,
    #[ignore]
    in_features: u32,
    #[ignore]
    out_features: u32,
    weight: Parameter<'a>,
    bias: Option<Parameter<'a>>
}

impl<'a> Linear<'a> {
	pub fn build(in_features: u32, out_features: u32) -> LinearArgsBuilder {
		LinearArgsBuilder::default()
			.in_features(in_features)
			.out_features(out_features)
	}
    pub fn new(args: LinearArgs) -> Linear<'a> {
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
    #[builder(default="false")]
    bias: bool
}
impl LinearArgsBuilder {
	pub fn done<'a>(self) -> Linear<'a> {
		let args = self.build().unwrap();
		Linear::new(args)
	}
}


impl<'a> ModIntf<'a> for Linear<'a> {
    fn delegate(&mut self) -> &mut Module<'a> {
        &mut self.delegate
    }
    fn forward(&mut self) {}
}
