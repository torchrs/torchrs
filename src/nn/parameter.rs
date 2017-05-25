use autograd::variable::Variable;
use tensor::*;

pub struct Parameter<T> {
    pub v: Variable<T>,
}
impl<T> Parameter<T> {
    pub fn apply(&mut self, callback: fn(&mut Tensor<T>)) {
        self.v.apply(callback)
    }
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut Parameter<T> {
        self as *mut Parameter<T>
    }
}
impl<T> Default for Parameter<T> {
    fn default() -> Self {
        Parameter { v: Variable::default() }
    }
}
