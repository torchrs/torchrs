use autograd::variable::Variable;
use tensor::*;

pub struct Parameter<'a> {
	pub v: Variable<'a>,

}
impl <'a>Parameter<'a> {
	pub fn apply(&mut self,  callback: fn(&mut Tensor)) {
		self.v.apply(callback)
	}
}
