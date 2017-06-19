use optim::*;


pub struct SGD<'a, T: Copy + 'a> {
    optimizer: Optimizer<'a, T>,
}

impl<'a, T: Copy + 'a> SGD<'a, T> {
    pub fn new(params: ParamIter<'a, T>, defaults: HashMap<&'static str, OptimOpts>) -> Self {
        let mut sgd_defaults = map_opt!{"momentum" => 0.0, "dampening" => 0.0,
                 "weight_decay"=> 0.0, "nesterov"=>false};

        for (ref key, ref value) in defaults {
            let cloned = value.clone();
            sgd_defaults.insert(key, cloned);
        }
        SGD { optimizer: Optimizer::new(params, sgd_defaults) }
    }
}

impl<'a, T: Copy + 'a> OptIntf<'a, T> for SGD<'a, T> {
    fn step(&mut self) {}
    fn optimizer(&mut self) -> &mut Optimizer<'a, T> {
        &mut self.optimizer
    }
}
