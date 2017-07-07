use autograd::{Function, FuncIntf, FuncDelegate, FIWrap};
use tensor::{OptTensorKindList, TensorKindList};

impl_func!(LinearF);

impl FuncIntf for LinearF {
    fn forward(&mut self, input_list: &mut TensorKindList) -> TensorKindList {
        self.save_for_backward(input_list);
        let (input, weight) = (input_list.remove(0), input_list.remove(0));
        let mut output = input.new(()).resize_([input.size()[0], weight.size()[0]]);
        output.zero_().addmm_(0, 1, &input, &weight.t());
        if input_list.len() != 0 {
            let bias = input_list.remove(0).expand_as(&output);
            output.addt_(1, &bias);
        }
        vec![output]
    }
    fn backward(&mut self, grad_output_list: &mut OptTensorKindList) -> OptTensorKindList {
        let tensorlist = self.saved_tensors();
        let grad_output_ = grad_output_list.remove(0);
        let needs_input_grad = self.needs_input_grad();
        let grad_output = match grad_output_ {
            Some(f) => f,
            None => unreachable!(),
        };
        let (input, weight) = (&tensorlist[0], &tensorlist[1]);
        let mut output: OptTensorKindList = Vec::new();
        let grad_input = if needs_input_grad[0] {
            Some(grad_output.mm(weight))
        } else {
            None
        };
        output.push(grad_input);
        let grad_weight = if needs_input_grad[1] {
            Some(grad_output.t().mm(input))
        } else {
            None
        };
        output.push(grad_weight);
        if tensorlist.len() > 0 && needs_input_grad[2] {
            let grad_bias = grad_output.sum_reduce(0, false);
            output.push(Some(grad_bias));
        }
        output
    }
}
