use autograd::variable::Variable;
use tensor::*;

pub struct Parameter<'a> {
	pub v: Variable<'a>,

}
impl <'a>Parameter<'a> {
	pub fn apply(&mut self,  callback: fn(&mut Tensor)) {
		self.v.apply(callback)
	}
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut Parameter<'a> {
        self as *mut Parameter<'a>
    }
}
impl <'a>Default for Parameter<'a> {
	fn default() -> Self {
		Parameter {v: Variable::default() }
	}
}
