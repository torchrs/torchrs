#![feature(trace_macros)]
#![feature(log_syntax)]

extern crate num;
extern crate rand;
extern crate rutorch;
#[macro_use]
extern crate modparse_derive;


pub mod nn;
pub mod autograd;
pub mod tensor;
pub mod storage;


#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
    }
}
