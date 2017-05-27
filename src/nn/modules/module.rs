use std::collections::HashMap;
use std::collections::hash_map;
use tensor::*;
use autograd::variable::Variable;
use nn::Parameter;


// placeholder
struct TorchBackend {}

pub trait ModuleStruct {
    fn init_module(&mut self);
}

pub struct Module<T> {
    pub _name: String,
    pub training: bool,
    _backend: TorchBackend,

    _buffers: HashMap<String, Tensor<T>>,
    //	_backward_hooks:
    //	_forward_hooks:
    _params: HashMap<String, *mut Parameter<T>>,
    _modules: HashMap<String, *mut Module<T>>,
}
pub struct PtrIterMut<'a, T: 'a> {
    //mod_iter: linked_hash_map::IterMut<'a, &'a str, *mut T>,
    mod_iter: hash_map::IterMut<'a, String, *mut T>,
}
pub struct PtrIter<'a, T: 'a> {
    //mod_iter: linked_hash_map::IterMut<'a, &'a str, *mut T>,
    mod_iter: hash_map::Iter<'a, String, *mut T>,
}

impl<'a, T> Iterator for PtrIterMut<'a, T> {
    type Item = (&'a String, &'a mut T);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some((name, t)) = self.mod_iter.next() {
            Some((name, unsafe { &mut **t as &mut T }))
        } else {
            None
        }
    }
}

impl<'a, T> Iterator for PtrIter<'a, T> {
    type Item = (&'a str, &'a T);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some((name, t)) = self.mod_iter.next() {
            Some((name, unsafe { &**t as &T }))
        } else {
            None
        }
    }
}

impl<T> Module<T> {
    pub fn new() -> Module<T> {
        Module {
            _name: String::from(""),
            _backend: TorchBackend {},
            _buffers: HashMap::new(),
            _params: HashMap::new(),
            _modules: HashMap::new(),
            training: true,
        }
    }
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut Module<T> {
        self as *mut Module<T>
    }
    pub fn add_module(&mut self, module: &mut ModIntf<T>) {
        let m = module.delegate();
        self._modules.insert(m._name.clone(), m.as_mut_ptr());

    }
    pub fn add_param(&mut self, name: &str, param: &mut Parameter<T>) {
        let s = String::from(name);
        self._params.insert(s, param.as_mut_ptr());

    }
    pub fn modules_iter_mut(&mut self) -> PtrIterMut<Module<T>> {
        PtrIterMut { mod_iter: self._modules.iter_mut() }
    }
    pub fn modules_iter(&mut self) -> PtrIter<Module<T>> {
        PtrIter { mod_iter: self._modules.iter() }
    }
    pub fn params_iter_mut(&mut self) -> PtrIterMut<Parameter<T>> {
        PtrIterMut { mod_iter: self._params.iter_mut() }
    }
    pub fn register_buffer(&mut self, name: &str, tensor: &mut Tensor<T>) {
        self._buffers.insert(String::from(name), tensor.clone());
    }
    fn _apply(&mut self, callback: fn(&mut Tensor<T>)) {
        for (_, module) in self.modules_iter_mut() {
            module._apply(callback)
        }
        for (_, param) in self.params_iter_mut() {
            param.apply(callback)
            /*
			if let Some(g) = p._grad {
					callback(g.data)
			}
			*/
            /* see also _buffers */
        }
    }
    fn apply(&mut self, callback: fn(&mut Self)) {
        for (_, module) in self.modules_iter_mut() {
            module.apply(callback)
        }
        callback(self)
    }
    pub fn repr(&mut self) -> String {
        let mut tmpstr = format!("{} (\n", self._name);
        for (key, module) in self.modules_iter_mut() {
            let modstr = module.repr();
            let modstr = format!("  {}", modstr);
            tmpstr = format!("{} ({}): {}\n", tmpstr, key, modstr);
        }
        tmpstr = format!("{})", tmpstr);
        tmpstr
    }
    pub fn eval(&mut self) {
        self.train(false)
    }
    fn train(&mut self, mode: bool) {
        self.training = mode;
        for (_, module) in self.modules_iter_mut() {
            module.train(mode)
        }

    }
}

pub trait ModIntf<T> {
    fn delegate(&mut self) -> &mut Module<T>;
    fn forward(&mut self, input: &mut Variable<T>) -> Variable<T>;
    fn forwardv(&mut self, input: &mut Vec<Variable<T>>) -> Vec<Variable<T>>;
    fn f(&mut self, input: &mut Variable<T>) -> Variable<T> {
        {
            let mut m = self.delegate();
            // do pre-forward hooks
        }
        let output = self.forward(input);
        {
            let mut m = self.delegate();
            // do post-forward hooks
        }
        output
    }
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
