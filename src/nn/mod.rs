use super::*;
use std::collections::HashMap;
use std::collections::hash_map;
use std::iter;
//use linked_hash_map;
//use linked_hash_map::LinkedHashMap;

pub trait ModuleStruct<'a> {
    fn init_module(&mut self);
}

struct TorchBackend {}
pub struct Parameter<'a> {
    data: &'a mut Tensor,
}
impl<'a> Parameter<'a> {
    fn valid(&self) -> bool {
        true
    }
}

pub struct Module<'a> {
    pub _name: &'a str,
    _backend: TorchBackend,
    //	_buffers: HashTable<&str, Tensor>
    //	_backward_hooks:
    //	_forward_hooks:
    //_modulesp: LinkedHashMap<&'a str, *mut Module<'a>>,
    _params: HashMap<&'a str, *mut Parameter<'a>>,
    _modulesp: HashMap<&'a str, *mut Module<'a>>,
    training: bool,
}
pub struct PtrIterMut<'a, T: 'a> {
    //mod_iter: linked_hash_map::IterMut<'a, &'a str, *mut T>,
    mod_iter: hash_map::IterMut<'a, &'a str, *mut T>,
}

impl<'a, T> Iterator for PtrIterMut<'a, T> {
    type Item = (&'a str, &'a mut T);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some((name, t)) = self.mod_iter.next() {
            Some((name, unsafe { &mut **t as &mut T }))
        } else {
            None
        }
    }
}

impl<'a> Module<'a> {
    pub fn new() -> Module<'a> {
        Module {
            _name: "",
            _backend: TorchBackend {},
            _params: HashMap::new(),
            //_modulesp: LinkedHashMap::new(),
            _modulesp: HashMap::new(),
            training: true,
        }
    }
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut Module<'a> {
        self as *mut Module<'a>
    }
    pub fn add_module(&mut self, module: &mut ModIntf<'a>) {
        let m = module.delegate();
        self._modulesp.insert(m._name, m.as_mut_ptr());

    }
    pub fn modules_iter_mut(&mut self) -> PtrIterMut<Module<'a>> {
        PtrIterMut { mod_iter: self._modulesp.iter_mut() }
    }
    pub fn params_iter_mut(&mut self) -> PtrIterMut<Parameter<'a>> {
        PtrIterMut { mod_iter: self._params.iter_mut() }
    }
    fn _apply(&mut self, callback: fn(&mut Tensor)) {
        for (_, module) in self.modules_iter_mut() {
            module._apply(callback)
        }
        for param in self.params_iter_mut()
                .filter_map(|(_, p)| if p.valid() { Some(p) } else { None }) {
            callback(param.data);
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

pub trait ModIntf<'a> {
    fn delegate(&mut self) -> &mut Module<'a>;
    fn forward(&mut self);
}

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
