use autograd::{Function, FuncIntf, FuncDelegate, FIWrap};
use tensor::*;

impl_func!(LinearF);

impl FuncIntf for LinearF {
    fn forward(&mut self, input_list: &mut TensorKindList) -> TensorKindList {
        self.save_for_backward(input_list);
        let (input, weight) = (input_list.remove(0), input_list.remove(0));
        let mut output = input.new_([input.size()[0], weight.size()[0]]);
        output = output.addmm_(&0.into(), &1.into(), &input, &weight.t());
        if input_list.len() != 0 {
            let bias = input_list.remove(0);
            output = output.clone().addt_(&1.into(), &bias.expand_as(&output));
        }

        vec![output]
    }
    fn backward(&mut self, input: &mut TensorKindList) -> TensorKindList {
        unimplemented!()
    }
}
