

trait NNTestCase<T : NumLimits> : TestCase {

	pub fn _analytical_jacobian_ext(&mut self, module: &mut nn::ModIntf<T>, input: &mut Variable<T>, jacobian_input: bool, jacobian_parameters: bool) {
		let output = self._forward(module, input);
		let output_t = output.data_borrow();
		let mut d_out =  output_t.new(()).resize_as_(&output_t);
		let flat_d_out = d_out.view([-1]);

		let jacobian_param : Tensor<T>;
		let flat_jacobian_input : Vec<Tensor<T>>;
		if jacobian_input {

		}
		if jacobian_parameters {
			let (param, d_param) = self._get_parameters(madule);
			let num_param = d_param.iter().map(|p| p.numel()).sum();
			jacobian_param = torch::zeros([num_param, d_out.numel()]);
		}
		for i in 0..flat_d_out.numel() {
			d_out.zero_();
			flat_d_out[i] = 1;
			if jacobian_parameters {
				self._zero_grad_parameters(module);
			}
			if jacobian_input {
				self._zero_grad_input(input)
			}
			let d_input = self._backward(module, input, output, d_out);

			if jacobian_input {
				for jacobian_x, d_x in zip(flat_jacobian_input, iter_tensors(d_input)) {
					jacobian_x
				}
			}

		} 
	}

	pub fn _analytical_jacobian(&mut self, module: &mut nn::ModIntf<T>, input: &mut Variable<T>) {
		self._analytical_jacobian_ext(module, input, true, true)
	}

}