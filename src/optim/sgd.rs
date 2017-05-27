
use optim::{Optimizer, OptIntf};

pub struct SGD {
	optimizer: Optimizer,
}

impl SGD {
	pub fn new() -> Self{
		SGD {optimizer: Optimizer::new()}
	}
}

impl OptIntf for SGD {
	fn step(&mut self) {

	}
}