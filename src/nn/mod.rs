#![feature(trace_macros)]
#![feature(log_syntax)]

use linked_hash_map::LinkedHashMap;
use std::rc::Rc;

pub trait ModuleStruct {
	fn init_module(&self);
}

struct TorchBackend {}
struct Parameter {}

struct Module<'a> {
	_name: &'a str,
	_backend : TorchBackend,
//	_buffers: HashTable<&str, Tensor>
//	_backward_hooks: 
//	_forward_hooks: 
	_params: LinkedHashMap<&'a str, &'a Parameter >,
    _modules: LinkedHashMap<&'a str, Rc< ModIntf<'a> > >,
	training: bool,
}

impl <'a>Module<'a> {
	pub fn new() -> Module<'a> {
		Module {_name: "", _backend: TorchBackend {}, _params: LinkedHashMap::new(), _modules: LinkedHashMap::new(), training: true }
	}
}

trait ModIntf<'a> {
	fn delegate(&'a self) -> &'a Module<'a>;
	fn forward(&mut self /* Tensor */ ) /* -> Tensor */;
}

#[derive(ModParse)]
struct Linear<'a> {
	delegate: Module<'a> ,
	in_features: u32,
}

impl <'a>Linear<'a> {
	pub fn new(/* args: LinearArgs */) -> Rc<Linear<'a>> {
		let t = Rc::new(Linear {delegate: Module::new(), in_features: 0 });
		t.init_module();
		t
	}
}

impl <'a> ModIntf<'a> for Linear<'a>  {
	fn delegate(&'a self) -> &'a Module<'a> {
		&self.delegate
	}
	fn forward(&mut self/* Tensor */) /* -> Tensor */ {

	}
}

#[derive(ModParse)]
struct MyMod<'a> {
	delegate: Module<'a>,
	#[module]
	a: Rc< Linear<'a> >,
}

impl <'a> MyMod<'a> {
	pub fn new() -> Rc< MyMod<'a> > {
		let t = Rc::new( MyMod {delegate: Module::new(), a: Linear::new()} );
		t.init_module();
		t
	}
}




