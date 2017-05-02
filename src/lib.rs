extern crate num;
extern crate rutorch;

use rutorch::*;
use std::ops::Index;
use num::Num;
use std::cmp::max;


pub trait ArgsArray<T> {
	fn is_leaf(&self) -> bool;
	fn child(&self) -> &T;
	fn dim(&self) -> usize;
}

impl <T> ArgsArray<T> for [T]  {
	fn is_leaf(&self) -> bool { false }
	fn child(&self) ->  &T { &self[0]}
	fn dim(&self) -> usize { self.len()}
}

impl <T:Num + Sized> ArgsArray<T> for T  {
	fn is_leaf(&self) -> bool { true }
	fn child(&self) ->  &T { &self}
	fn dim(&self) -> usize { 1 }
}


//fn from(args: &ArgsArray<T>) -> Self;

trait Tensor<'a> : Index<&'a [isize]> {
	fn new() -> Self;
}

struct FloatTensor {
	t: *mut THFloatTensor,
	dims: Vec<isize>,
}


impl <'a>Tensor<'a> for FloatTensor {
	fn new() -> Self {
		unsafe { FloatTensor { t : THFloatTensor_new(), dims: Vec::new()  }}
	}
}

impl <'a> Index<&'a [isize]> for FloatTensor {
	type Output = f32;

	fn index(&self, idx: &'a [isize]) -> &f32 {
		let mut index : isize = 0;
		let lastidx = max(0, idx.len() as isize - 1) as usize;
		if idx.len() != self.dims.len() { panic!("bad dimlen")}
		for i in 0..lastidx {
			if idx[i] >= self.dims[i] { panic!("bad dimlen")}
			index += idx[i] * self.dims[i]
		}
		if idx[lastidx] >= self.dims[lastidx] { panic!("bad dimlen")}
		index += idx[lastidx];
		unsafe {&*(*(*self.t).storage).data.offset(index)}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
    #[test]
    fn args_array_works() {
    	let v = [[5, 2], [3, 4]];
    	let x = v.child();
    	let z = x.child();
    	println!("is_leaf = {} {} {:?}, {} {} {:?}, {} {} {:?}", 
    		v.is_leaf(), v.dim(), v, 
            x.is_leaf(), x.dim(), x,
            z.is_leaf(), z.dim(), z
            );
    }
    fn tensor_ops_work() {
		let v = [[5, 2], [3, 4]];

    }
}
