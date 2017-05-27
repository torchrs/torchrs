use autograd::function::*;
use tensor::Tensor;
use ::*;

pub type VarList<T> = Vec<Variable<T>>;

pub struct VariableImpl<T> {
    data: Option<Tensor<T>>,
    grad_fn: OptRcMut<Function<T>>,
    grad: OptRcMut<Variable<T>>,
	// version_counter etc ...
}

pub struct Variable<T> {
    value: RcMut<VariableImpl<T>>,
}

impl<T> Clone for Variable<T> {
    fn clone(&self) -> Self {
        Variable { value: self.value.clone() }
    }
}

pub struct SavedVariable<T> {
    data: Box<Tensor<T>>,
    grad_fn: OptRcMut<Function<T>>,
    grad: OptRcMut<Variable<T>>,
	// version_counter etc ...
}

pub struct BackwardArgs {}

impl<T> Variable<T> {
    pub fn apply(&mut self, callback: fn(&mut Tensor<T>)) {
        let mut v = self.value.borrow_mut();
        if let Some(ref mut t) = v.data {
            callback(&mut *t);
        }
    }
    pub fn view(&self, dims: &[i32]) -> Self {
        self.clone()
    }
    // Computes the gradient of current variable w.r.t. graph leaves
    pub fn backward(&mut self, args: &BackwardArgs) {}
    // Detach from graph
    pub fn detach_(&mut self) {}
    // return a new variable detached from graph
    pub fn detach(&self) -> Variable<T> {
        self.clone()
    }
}

impl<T> Default for Variable<T> {
    fn default() -> Self {
        let v = VariableImpl {
            data: None,
            grad_fn: None,
            grad: None,
        };
        Variable { value: RcMutNew(v) }
    }
}
