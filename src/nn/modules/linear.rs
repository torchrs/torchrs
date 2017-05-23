use nn::modules::module::*;

#[derive(ModParse)]
pub struct Linear<'a> {
    delegate: Module<'a>,
    in_features: u32,
}

impl<'a> Linear<'a> {
    pub fn new() -> Linear<'a> {
        let mut t = Linear {
            delegate: Module::new(),
            in_features: 0,
        };
        t.init_module();
        t
    }
}

impl<'a> ModIntf<'a> for Linear<'a> {
    fn delegate(&mut self) -> &mut Module<'a> {
        &mut self.delegate
    }
    fn forward(&mut self) {}
}
