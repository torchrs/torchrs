use autograd::variable::Variable;
use num;
use tensor::*;

pub struct Parameter<T: NumLimits<T>> {
    pub v: Variable<T>,
}
impl<T: NumLimits<T>> Parameter<T> {
    pub fn new(dims: Vec<usize>) -> Self {
        panic!("implement")
    }
    pub fn apply(&mut self, callback: fn(&mut Tensor<T>)) {
        self.v.apply(callback)
    }
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut Parameter<T> {
        self as *mut Parameter<T>
    }
}
impl<T: NumLimits<T>> Default for Parameter<T> {
    fn default() -> Self {
        Parameter { v: Variable::default() }
    }
}
