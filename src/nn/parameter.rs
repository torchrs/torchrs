use autograd::variable::Variable;
use num;
use torch;
use tensor::*;

pub struct Parameter<T: NumLimits> {
    pub v: Variable<T>,
}
impl<T: NumLimits> Parameter<T> {
    pub fn new<S>(data: S) -> Self
        where S: Into<THVec<T>>
    {
        let t = torch::tensor(data);
        let v = Variable::new(t);
        Parameter { v: v }
    }
    pub fn apply(&mut self, callback: fn(&mut Tensor<T>)) {
        self.v.apply(callback)
    }
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut Parameter<T> {
        self as *mut Parameter<T>
    }
}
impl<T: NumLimits> Default for Parameter<T> {
    fn default() -> Self {
        Parameter { v: Variable::default() }
    }
}
