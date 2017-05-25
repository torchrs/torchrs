use autograd::function::*;
use tensor::Tensor;
use ::*;

pub struct VariableImpl<'a, T: 'a> {
	data: Option<&'a mut Tensor<'a, T>>,
	grad_fn: OptRcMut<Function<'a, T>>,
	grad: OptRcMut<Variable<'a, T>>,
	// version_counter etc ...
}

pub struct Variable<'a, T: 'a> {
	value: RcMut<VariableImpl<'a, T>>,
}

impl<'a, T> Clone for Variable<'a, T> {
    fn clone(&self) -> Self {
        Variable { value: self.value.clone() }
    }
}

pub struct SavedVariable<'a, T: 'a> {
	data: &'a mut Tensor<'a, T>,
	grad_fn: OptRcMut<Function<'a, T>>,
	grad: OptRcMut<Variable<'a, T>>,
	// version_counter etc ...
}

impl <'a, T>Variable<'a, T> {
	pub fn apply(&mut self,  callback: fn(&mut Tensor<'a, T>)) {
		let mut v = self.value.borrow_mut();
		if let Some(ref mut t) = v.data {
			callback(*t)
		}
	}
}

impl <'a, T>Default for Variable<'a, T> {
	fn default() -> Self {
		let v = VariableImpl {data: None, grad_fn: None, grad: None};
		Variable { value: RcMutNew(v) }
	}
}