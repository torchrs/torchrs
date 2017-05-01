extern crate num;
extern crate rutorch;

use rutorch::*;

use num::Num;
//use std::ops::Index;


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


trait Tensor<T> {
	fn new() -> Self;
	fn from(args: &ArgsArray<T>) -> Self;
}

struct FloatTensor3d {
	t: *mut THFloatTensor,
}

/*
struct FloatTensorCursor {
	t: *const THFloatTensor,
	off: usize
}
*/

impl Tensor<f32> for FloatTensor3d {
	fn new() -> Self {
		unsafe { FloatTensor3d { t : THFloatTensor_new() }}
	}
	fn from(args : &ArgsArray<f32>) -> Self {
		unsafe { FloatTensor3d { t : THFloatTensor_new() }}
	}
}

struct FloatTensor2d {
	t: *mut THFloatTensor,
}

impl Tensor<f32> for FloatTensor2d {
	fn new() -> Self {
		unsafe { FloatTensor2d { t : THFloatTensor_new() }}
	}
	fn from(args : &ArgsArray<f32>) -> Self {
		unsafe { FloatTensor2d { t : THFloatTensor_new() }}
	}
}


struct FloatTensor1d {
	t: *mut THFloatTensor,
}

impl Tensor<f32> for FloatTensor1d {
	fn new() -> Self {
		unsafe { FloatTensor1d { t : THFloatTensor_new() }}
	}
	fn from(args : &ArgsArray<f32>) -> Self {
		unsafe { FloatTensor1d { t : THFloatTensor_new() }}
	}
}
/*
impl <'a> Index<isize> for FloatTensor3d {
	type Output = FloatTensor2d;

	fn index(&self, idx: isize) -> &FloatTensor2d {
		let t = Box::new(FloatTensor2d::new());
//		let u: BoxRef<FloatTensor2d> = BoxRef::new(t);
		t as &FloatTensor2d
	}
}

impl Index<isize> for FloatTensor2d {
	type Output = BoxRef<FloatTensor1d>;

	fn index(&self, idx: isize) -> &BoxRef<FloatTensor1d> {
		let t = Box::new(FloatTensor1d::new());
		let u: BoxRef<FloatTensor1d> = BoxRef::new(t);
		&u
	}
}

impl Index<isize> for FloatTensor1d {
	type Output = f32;

	fn index(&self, idx: isize) -> &f32 {
		&self[idx]
	}
}
*/
/*
struct Tensor <T> {
	t: *const ::std::os::raw::c_void;
}

pub trait TensorFuncs {
	fn new() -> Self;
}


impl TensorFuncs for Tensor<Float> {
	fn new(init: ) -> Self {
		unsafe { Tensor { t : THFloatTensor_new() }}
	}
}
*/


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
