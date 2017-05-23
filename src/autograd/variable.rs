use std::rc::Rc;
use std::cell::RefCell;
use autograd::function::*;
use tensor::Tensor;


type RcMut<T> = Rc<RefCell<T>>;
type OptRcMut<T> = Option<RcMut<T>>;

pub struct Variable<'a> {
	data: Option<&'a mut Tensor>,
	grad_fn: OptRcMut<Function<'a>>,
	grad: OptRcMut<Variable<'a>>,
	// version_counter etc ...
}

pub struct SavedVariable<'a> {
	data: &'a mut Tensor,
	grad_fn: OptRcMut<Function<'a>>,
	grad: OptRcMut<Variable<'a>>,
	// version_counter etc ...
}

impl <'a>Variable<'a> {
	pub fn apply(&mut self,  callback: fn(&mut Tensor)) {
		if let Some(ref mut t) = self.data {
			callback(*t)
		}
	}
}

impl <'a>Default for Variable<'a> {
	fn default() -> Self {
		Variable {data: None, grad_fn: None, grad: None}
	}
}