use std::collections::HashMap;
use tensor::{Tensor, NumLimits};
use autograd::{Variable, VarId, VarAccess};
use std::slice;

pub trait InitModuleStruct {
    fn init_module(self) -> Self;
}
pub trait GetFieldStruct<T: NumLimits> {
    fn get_param(&mut self, name: &str) -> Option<VarId>;
    fn get_module(&mut self, name: &str) -> &mut ModIntf<T>;
}

pub struct Module<T: NumLimits> {
    pub _name: String,
    pub training: bool,
    //_backend: TorchBackend,
    //	_backward_hooks:
    //	_forward_hooks:
    _buffers: HashMap<String, Tensor<T>>,
    pub _params: Vec<&'static str>,
    pub _modules: Vec<&'static str>,
}

impl<T: NumLimits> Module<T> {
    pub fn new() -> Module<T> {
        Module {
            _name: String::from(""),
            _buffers: HashMap::new(),
            _params: Vec::new(),
            _modules: Vec::new(),
            training: true,
        }
    }
    pub fn add_module(&mut self, module: &'static str) {
        self._modules.push(module);
    }
    pub fn add_param(&mut self, name: &'static str) {
        self._params.push(name);
    }
    pub fn register_buffer(&mut self, name: &str, tensor: &mut Tensor<T>) {
        self._buffers.insert(String::from(name), tensor.clone());
    }
    /*
    pub fn repr(&mut self) -> String {
        let mut tmpstr = format!("{} (\n", self._name);
        for (key, mut module) in self.modules_iter_mut() {
            let modstr = module.repr();
            tmpstr = format!("{} ({}): {}\n", tmpstr, key, modstr);
        }
        tmpstr = format!("{})", tmpstr);
        tmpstr
    }
    */
    pub fn eval(&mut self) {
        self.train(false)
    }
    fn train(&mut self, mode: bool) {
        self.training = mode;
    }
}

pub trait ModDelegate<T: NumLimits> {
    fn delegate(&mut self) -> &mut Module<T>;
    fn params_iter_mut(&mut self) -> ::std::vec::IntoIter<Variable<T>>;
    fn _apply(&mut self, callback: fn(&mut ::tensor::Tensor<T>));
    fn apply(&mut self, callback: fn(&mut ModIntf<T>));
}

pub trait ModIntf<T: NumLimits>: ModDelegate<T> + GetFieldStruct<T> {
    fn forward(&mut self, input: &mut Variable<T>) -> Variable<T>;
    fn f(&mut self, mut input: Variable<T>) -> Variable<T> {
        {
            let mut m = self.delegate();
            // do pre-forward hooks
        }
        let output = self.forward(&mut input);
        {
            let mut m = self.delegate();
            // do post-forward hooks
        }
        output
    }
    fn train(&mut self) {
        self.delegate().training = true;
        let mod_names = self.delegate()._modules.clone();
        for name in mod_names {
            let module = self.get_module(name);
            module.train()
        }
    }
    fn max_id(&mut self) -> i32 {
        let mut max = 0;
        self.apply_parameters(&mut |v| if v.id > max {
                                       max = v.id
                                   });
        max
    }
    fn free_grad(&mut self) {
        self.apply_parameters(&mut |v| *v.access().grad() = None);
    }
    fn free_graph(&mut self) {
        self.free_grad();
        ::autograd::var_table_reset(self.max_id());
    }
    fn apply_parameters(&mut self, func: &mut FnMut(&mut Variable<T>)) {
        let mod_names = self.delegate()._modules.clone();
        for name in mod_names {
            let module = self.get_module(name);
            module.apply_parameters(func)
        }
        for mut param in self.params_iter_mut() {
            func(&mut param)
        }
    }
    fn eval(&mut self) {
        self.delegate().training = false;
        let mod_names = self.delegate()._modules.clone();
        for name in mod_names {
            let module = self.get_module(name);
            module.eval()
        }
    }
}
pub trait ModIntfV<T: NumLimits>: ModDelegate<T> {
    fn forwardv(&mut self, input: &mut Vec<Variable<T>>) -> Vec<Variable<T>>;
    fn fv(&mut self, input: &mut Vec<Variable<T>>) -> Vec<Variable<T>> {
        {
            let mut m = self.delegate();
            // do pre-forward hooks
        }
        let output = self.forwardv(input);
        {
            let mut m = self.delegate();
            // do post-forward hooks
        }
        output
    }
}
