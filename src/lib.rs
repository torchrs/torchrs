#![feature(trace_macros)]
#![feature(log_syntax)]

extern crate num;
extern crate rand;
extern crate rutorch;
#[macro_use]
extern crate modparse_derive;
#[macro_use]
extern crate derive_builder;

#[macro_use]
pub mod macros;


pub mod nn;
pub mod autograd;
pub mod optim;
pub mod utils;
pub mod tensor;
pub mod storage;
use std::rc::Rc;
use std::cell::RefCell;


// Mutable reference counted T
pub type RcMut<T> = Rc<RefCell<T>>;
// Nullable mutable reference counted T
pub type OptRcMut<T> = Option<RcMut<T>>;
/// Array index type
pub type Ix = usize;
/// Array index type (signed)
pub type Ixs = isize;


#[allow(non_snake_case)]
pub fn RcMutNew<T>(arg: T) -> RcMut<T> {
    Rc::new(RefCell::new(arg))
}

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {}
}
