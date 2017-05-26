use autograd::{Function, FuncIntf, Variable, VarList};

pub struct ConvNd<T> {
	delegate: Function<T>,
}

impl<T> ConvNd<T> {
	pub fn new() -> Self {
		ConvNd {}
	}
	fn forward_apply(&mut self, input: &mut VarList<T>) -> VarList<T> {
		input
	}
	fn backward_apply(&mut self, input: &mut VarList<T>) -> VarList<T> {
		input
	}
}

impl<T> FuncIntf<T> for ConvNd<T> {
	fn delegate(&mut self) -> &mut Function<T> {
		&mut self.delegate
	}
	fn forward(&mut self, input: &mut VarList<T>) -> VarList<T> {
		self.forward_apply(&mut input)
	}
	fn backward(&mut self, input: &mut VarList<T>) -> VarList<T> {
		self.backward_apply(&mut input)
	}
}