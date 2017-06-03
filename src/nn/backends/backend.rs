// Maintain the Torch names
#![allow(non_snake_case)]

use tensor::TensorKind;

pub trait BackendIntf {
    fn SpatialDilatedMaxPooling_updateOutput(&self,
                                             input: &TensorKind,
                                             output: &mut TensorKind,
                                             indices: &mut TensorKind,
                                             kernel_size: (i32, i32),
                                             stride: (i32, i32),
                                             padding: (i32, i32),
                                             dilation: (i32, i32),
                                             ceil_mode: bool);

    fn SpatialDilatedMaxPooling_updateGradInput(&self,
                                                input: &TensorKind,
                                                grad_output: &TensorKind,
                                                grad_input: &mut TensorKind,
                                                indices: &TensorKind,
                                                kernel_size: (i32, i32),
                                                stride: (i32, i32),
                                                padding: (i32, i32),
                                                dilation: (i32, i32),
                                                ceil_mode: bool);
}
