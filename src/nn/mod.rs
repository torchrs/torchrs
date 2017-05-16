use std::slice;

use std::marker;

pub trait ModuleStruct<'a> {
	fn init_module(&mut self);
}

struct TorchBackend {}
struct Parameter {}

pub struct Module<'a> {
	pub _name: &'a str,
	_backend : TorchBackend,
//	_buffers: HashTable<&str, Tensor>
//	_backward_hooks: 
//	_forward_hooks: 
	_params: Vec<&'a str>,
    _modulesp: Vec<*mut Module<'a>>,
	training: bool,
}
pub struct ModuleIter<'module, 'iter> {
	mod_iter: slice::IterMut<'iter, *mut Module<'module> >,
//	 _marker: marker::PhantomData<&'module Module<'module>>,
}

impl <'a>Module<'a> {
	pub fn new() -> Module<'a> {
		Module {_name: "", _backend: TorchBackend {}, _params: Vec::new(), 
		_modulesp: Vec::new(), training: true }
	}
	#[inline]
    fn as_mut_ptr(&mut self) -> *mut Module<'a> {
        self as *mut Module<'a>
    }
    pub fn add_module(&mut self, module: &mut ModIntf<'a>) {
    	self._modulesp.push(module.delegate().as_mut_ptr())

    }
	pub fn modules_iter<'b>(&mut self) -> ModuleIter<'a, 'b> {
		ModuleIter {mod_iter: self._modulesp.iter_mut()}  //_marker: marker::PhantomData } }
	}
}

pub trait ModIntf<'a> {
	fn delegate(&mut self) -> &mut Module<'a>;
	fn forward(&mut self /* Tensor */ ) /* -> Tensor */;
}

#[derive(ModParse)]
pub struct Linear<'a> {
	delegate: Module<'a> ,
	in_features: u32,
}

impl <'a>Linear<'a> {
	pub fn new(/* args: LinearArgs */) -> Linear<'a> {
		let mut t = Linear {delegate: Module::new(), in_features: 0 };
		t.init_module();
		t
	}
}

impl <'a> ModIntf<'a> for Linear<'a>  {
	fn delegate(&mut self) -> &mut Module<'a> {
		&mut self.delegate
	}
	fn forward(&mut self/* Tensor */) /* -> Tensor */ {

	}
}





