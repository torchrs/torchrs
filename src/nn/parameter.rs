use autograd::variable::Variable;
use num;
use tensor::*;

pub struct Parameter<T: Copy> {
    pub v: Variable<T>,
}
impl<T: Copy + Default + num::Num> Parameter<T> {
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
impl<T: Copy> Default for Parameter<T> {
    fn default() -> Self {
        Parameter { v: Variable::default() }
    }
}
