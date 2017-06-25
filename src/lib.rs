#![feature(specialization)]
 #![feature(type_ascription)]
 #![feature(concat_idents)]

extern crate rutorch;
#[macro_use]
extern crate modparse_derive;
#[macro_use]
extern crate derive_builder;

extern crate num;
extern crate rand;
extern crate itertools;

// only needed for torchvision
extern crate curl;
extern crate flate2;
extern crate memmap;

// serialization support
#[macro_use]
extern crate serde_derive;
extern crate rmp;
extern crate rmp_serde as rmps;
extern crate serde;


#[macro_use]
pub mod macros;


pub mod nn;
pub mod autograd;
pub mod optim;
pub mod utils;
pub mod tensor;
pub mod storage;
pub mod torch;
use std::rc::Rc;
use std::cell::RefCell;


// Mutable reference counted T
pub type RcMut<T> = Rc<RefCell<T>>;
// Nullable mutable reference counted T
pub type OptRcMut<T> = Option<RcMut<T>>;
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
