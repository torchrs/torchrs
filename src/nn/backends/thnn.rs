// Maintain the Torch names
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(unused_variables)]

use tensor::TensorKind;
use nn::backends::backend::BackendIntf;
use rutorch::*;
static FLOAT: i32 = 1;
static DOUBLE: i32 = 2;


pub struct THNN_Backend {
    state: *mut ::std::os::raw::c_void,
}

impl BackendIntf for THNN_Backend {
    fn SpatialDilatedMaxPooling_updateOutput(&self,
                                             input: &TensorKind,
                                             output: &mut TensorKind,
                                             indices: &mut TensorKind,
                                             kernel_size: (i32, i32),
                                             stride: (i32, i32),
                                             padding: (i32, i32),
                                             dilation: (i32, i32),
                                             ceil_mode: bool) {
        let kind = match *input {
            TensorKind::FloatTensor(_) => FLOAT,
            _ => panic!("unknown tensor type"),
        };
        if kind == FLOAT {
            unsafe {
                THNN_FloatSpatialDilatedMaxPooling_updateOutput(self.state,
                                                                input.in_thft(),
                                                                output.in_thft(),
                                                                indices.in_thlt(),
                                                                kernel_size.0,
                                                                kernel_size.1,
                                                                stride.0,
                                                                stride.1,
                                                                padding.0,
                                                                padding.1,
                                                                dilation.0,
                                                                dilation.1,
                                                                ceil_mode);
            }
        }

    }

    fn SpatialDilatedMaxPooling_updateGradInput(&self,
                                                input: &TensorKind,
                                                grad_output: &TensorKind,
                                                grad_input: &mut TensorKind,
                                                indices: &TensorKind,
                                                kernel_size: (i32, i32),
                                                stride: (i32, i32),
                                                padding: (i32, i32),
                                                dilation: (i32, i32),
                                                ceil_mode: bool) {

    }
}
