#![feature(trace_macros)]
#![feature(log_syntax)]

extern crate num;
extern crate rand;
extern crate rutorch;
#[macro_use]
extern crate modparse_derive;
#[macro_use]
extern crate derive_builder;

pub mod nn;
pub mod autograd;
pub mod tensor;
pub mod storage;
use std::rc::Rc;
use std::cell::RefCell;


pub type RcMut<T> = Rc<RefCell<T>>;
pub type OptRcMut<T> = Option<RcMut<T>>;

#[allow(non_snake_case)] 
pub fn RcMutNew<T>(arg: T) -> RcMut<T> {
	Rc::new(RefCell::new(arg))
}

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
    }
}
