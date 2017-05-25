use std::rc::Rc;
use std::cell::RefCell;

use std::vec::Vec;
use autograd::variable::*;

type RcMut<T> = Rc<RefCell<T>>;
type OptRcMut<T> = Option<RcMut<T>>;


pub struct Function<T> {
    saved_variables: Vec<SavedVariable<T>>,
    next_functions: Vec<(RcMut<Function<T>>, usize)>,
}
