use std::collections::{HashMap, hash_map};
use tensor::Tensor;
use autograd::Variable;
use std::vec::IntoIter;
use nn;

// placeholder
struct TorchBackend {}

pub trait ModuleStruct {
    fn init_module(self) -> Self;
}

pub struct Module<T: Copy> {
    pub _name: String,
    pub training: bool,
    _backend: TorchBackend,

    _buffers: HashMap<String, Tensor<T>>,
    //	_backward_hooks:
    //	_forward_hooks:
    _params: HashMap<String, *mut nn::Parameter<T>>,
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

pub struct ModIter<'a, T: 'a + Copy> {
    pub modules: IntoIter<*mut Module<T>>,
    pub iter: PtrIterMut<'a, nn::Parameter<T>>,
}
fn mod_accum<TMod: Copy>(module: &mut Module<TMod>, arg: &mut Vec<*mut Module<TMod>>) {
    arg.push((module));
}
impl<'a, TMod: 'a + Copy> ModIter<'a, TMod> {
    pub fn new(root: &'a mut Module<TMod>) -> Self {
        let mut mods = Vec::new();
        root.apply_arg(&mut mods, mod_accum);
        // XXX assert is root
        mods.pop();
        ModIter {
            modules: mods.into_iter(),
            iter: root.params_iter_mut(),
        }
    }
}

impl<'a, T: Copy> Iterator for ModIter<'a, T> {
    type Item = &'a mut nn::Parameter<T>;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some((_, t)) = self.iter.next() {
            Some(t)
        } else if let Some(modulep) = self.modules.next() {
            let mut module = unsafe { &mut *modulep as &mut Module<T> };
            self.iter = module.params_iter_mut();
            self.next()
        } else {
            None
        }
    }
}

impl<T: Copy> Module<T> {
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
    pub fn add_param(&mut self, name: &str, param: &mut nn::Parameter<T>) {
        let s = String::from(name);
        self._params.insert(s, param.as_mut_ptr());

    }
    pub fn modules_iter_mut(&mut self) -> PtrIterMut<Module<T>> {
        PtrIterMut { mod_iter: self._modules.iter_mut() }
    }
    pub fn modules_iter(&mut self) -> PtrIter<Module<T>> {
        PtrIter { mod_iter: self._modules.iter() }
    }
    pub fn params_iter_mut(&mut self) -> PtrIterMut<nn::Parameter<T>> {
        PtrIterMut { mod_iter: self._params.iter_mut() }
    }
    pub fn register_buffer(&mut self, name: &str, tensor: &mut Tensor<T>) {
        self._buffers.insert(String::from(name), tensor.clone());
    }
    pub fn parameters(&mut self) -> ModIter<T> {
        ModIter::new(self)
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
    pub fn apply(&mut self, callback: fn(&mut Self)) {
        for (_, module) in self.modules_iter_mut() {
            module.apply(callback)
        }
        callback(self)
    }
    fn apply_arg<Ta>(&mut self, arg: &mut Ta, callback: fn(&mut Self, &mut Ta)) {
        for (_, module) in self.modules_iter_mut() {
            module.apply_arg(arg, callback)
        }
        callback(self, arg)
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

pub trait ModDelegate<T: Copy> {
    fn delegate(&mut self) -> &mut Module<T>;
}

pub trait ModIntf<T: Copy>: ModDelegate<T> {
    fn forward(&mut self, input: &mut Variable<T>) -> Variable<T>;
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
    fn train(&mut self) {
        self.delegate().train(true)
    }
    fn eval(&mut self) {
        self.delegate().train(false)
    }
}

pub trait ModIntfV<T: Copy>: ModDelegate<T> {
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
