use autograd::function::*;
use tensor::Tensor;
use ::*;


pub struct Variable<'a, T: 'a> {
	data: Option<&'a mut Tensor<'a, T>>,
	grad_fn: OptRcMut<Function<'a, T>>,
	grad: OptRcMut<Variable<'a, T>>,
	// version_counter etc ...
}

pub struct SavedVariable<'a, T: 'a> {
	data: &'a mut Tensor<'a, T>,
	grad_fn: OptRcMut<Function<'a, T>>,
	grad: OptRcMut<Variable<'a, T>>,
	// version_counter etc ...
}

impl <'a, T: 'a>Variable<'a, T> {
	pub fn apply(&mut self,  callback: fn(&mut Tensor<'a, T>)) {
		if let Some(ref mut t) = self.data {
			callback(*t)
		}
	}
}

impl <'a, T: 'a>Default for Variable<'a, T> {
	fn default() -> Self {
		Variable {data: None, grad_fn: None, grad: None}
	}
}