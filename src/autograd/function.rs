use std::rc::Rc;
use std::cell::RefCell;

use std::vec::Vec;
use autograd::variable::*;

type RcMut<T> = Rc<RefCell<T>>;
type OptRcMut<T> = Option<RcMut<T>>;


pub struct Function<'a> {
	saved_variables: Vec<SavedVariable<'a>>, 
	next_functions: Vec<(RcMut<Function<'a>>, usize)>,
}