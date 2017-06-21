use std::collections::{HashMap, hash_map};
use tensor::Tensor;
use autograd::Variable;
use std::vec::IntoIter;
use std::ops::{Deref, DerefMut};
use std::marker::PhantomData;
use nn;

pub type Parameters<T> = Box<Iterator<Item = ModRefMut<'static, nn::Parameter<T>>>>;

pub trait ModuleStruct {
    fn init_module(self) -> Self;
}
pub struct ModRefMut<'a, T: 'a + ?Sized> {
    value: *mut T,
    phantom: PhantomData<&'a T>,
}

impl<'a, T> ModRefMut<'a, T> {
    pub fn new(value: *mut T) -> Self {
        ModRefMut {
            value: value,
            phantom: PhantomData,
        }
    }
}

impl<'a, T> From<&'a mut T> for ModRefMut<'a, T> {
    fn from(value: &mut T) -> Self {
        ModRefMut::new(value as *mut T)
    }
}
impl<'a, T> From<*mut T> for ModRefMut<'a, T> {
    fn from(value: *mut T) -> Self {
        ModRefMut::new(value)
    }
}

impl<'a, T> Deref for ModRefMut<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.value as &T }
    }
}
impl<'a, T> DerefMut for ModRefMut<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.value as &mut T }
    }
}


pub struct Module<T: Copy> {
    pub _name: String,
    pub training: bool,
    //_backend: TorchBackend,
    //	_backward_hooks:
    //	_forward_hooks:
    _buffers: HashMap<String, Tensor<T>>,
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

pub struct ParamIter<'a, T: Copy + 'a> {
    pub modules: Vec<*mut Module<T>>,
    pub mod_iter: IntoIter<*mut Module<T>>,
    pub iter: Box<Iterator<Item = *mut nn::Parameter<T>>>,
    phantom: PhantomData<&'a T>,
}
fn mod_accum<T: Copy>(module: &mut Module<T>, arg: &mut Vec<*mut Module<T>>) {
    arg.push((module));
}
impl<'a, T: 'static + Copy> ParamIter<'a, T> {
    pub fn new(root: &mut Module<T>) -> Self {
        let mut mods = Vec::new();
        root.apply_arg(&mut mods, mod_accum);
        let mut module = unsafe { &mut *mods[0] as &mut Module<T> };
        ParamIter {
            modules: mods.clone(),
            mod_iter: mods.into_iter(),
            iter: Box::new(module._params.iter_mut().map(|(_, d)| *d)),
            phantom: PhantomData,
        }
    }
}


impl<'a, T: Copy + 'static> Iterator for ParamIter<'a, T> {
    type Item = ModRefMut<'a, nn::Parameter<T>>;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(t) = self.iter.next() {
            Some(ModRefMut::new(t))
        } else if let Some(modulep) = self.mod_iter.next() {
            let mut module = unsafe { &mut *modulep as &mut Module<T> };
            self.iter = Box::new(module._params.iter_mut().map(|(_, d)| *d));
            self.next()
        } else {
            None
        }
    }
}

impl<'a, T: Copy + 'static> Clone for Box<ParamIter<'a, T>> {
    fn clone(&self) -> Self {
        let mut new_mod_list = self.modules.clone();
        let param_iter = Box::new(unsafe { &mut *new_mod_list[0] }
                                      ._params
                                      .iter_mut()
                                      .map(|(_, d)| *d));
        Box::new(ParamIter {
                     modules: new_mod_list,
                     mod_iter: self.modules.clone().into_iter(),
                     iter: param_iter,
                     phantom: PhantomData,
                 })
    }
}

impl<T: Copy> Module<T> {
    pub fn new() -> Module<T> {
        Module {
            _name: String::from(""),
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
        self._params.insert(name.into(), param.as_mut_ptr());

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
    pub fn parameters(&mut self) -> Parameters<T> {
        Box::new(ParamIter::new(self))
    }
    fn _apply(&mut self, callback: fn(&mut Tensor<T>)) {
        for (_, module) in self.modules_iter_mut() {
            module._apply(callback)
        }
        for (_, param) in self.params_iter_mut() {
            param.apply(callback);
            if let &mut Some(ref mut g) = param.v.grad() {
                g.apply(callback)
            }
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
    fn parameters(&mut self) -> Box<Iterator<Item = ModRefMut<'static, nn::Parameter<T>>>> {
        self.delegate().parameters()
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
