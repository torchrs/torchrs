pub use std::collections::HashMap;
pub use autograd::{Variable, VarId};
pub use nn::ModIntf;
use utils::unsafe_lib::MutMap;
use utils::TRVal;

pub struct Optimizer {
    pub defaults: HashMap<&'static str, TRVal>,
    pub state: MutMap<VarId, ParamState>,
}

pub type ParamState = HashMap<&'static str, TRVal>;

impl Optimizer {
    pub fn new(defaults: HashMap<&'static str, TRVal>) -> Self {
        Optimizer {
            defaults: defaults,
            state: MutMap::new(),
        }
    }
}

pub trait OptIntf<T: ::tensor::NumLimits + From<TRVal>> {
    fn optimizer(&mut self) -> &mut Optimizer;
    fn zero_grad(&mut self, model: &mut ModIntf<T>) {
        // XXX figure out point of parameter groups
        model.apply_parameters(&mut |p| if p.requires_grad() {
                                        // :-/
                                        let data = p.data().clone();
                                        let new_data =
                                            data.new(()).resize_as_(&data).zero_().clone();
                                        *p.grad() = Some(Variable::new(new_data))
                                    });
    }
    /* ignore largely unused closure arg to start */
    fn step(&mut self, model: &mut ModIntf<T>);
}
