use autograd::variable::Variable;
use tensor::*;

pub struct Parameter<'a, T: 'a> {
	pub v: Variable<'a, T>,

}
impl <'a, T: 'a>Parameter<'a, T> {
	pub fn apply(&mut self,  callback: fn(&mut Tensor<'a, T>)) {
		self.v.apply(callback)
	}
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut Parameter<'a, T> {
        self as *mut Parameter<'a, T>
    }
}
impl <'a, T: 'a>Default for Parameter<'a, T> {
	fn default() -> Self {
		Parameter {v: Variable::default() }
	}
}
