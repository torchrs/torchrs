#![allow(unused_variables)]
use tensor::*;
use std::cell::RefMut;

macro_rules! self_op {
    ($key:ident, $action:ident ) => { {
        let t = $key.new(()).resize_as_($key);
        let inner = $key.inner();
        t.value.borrow_mut().$action (inner);
        t
     }}
}

macro_rules! self_inplace_op {
    ($key:ident, $action:ident ) => {{
        let inner = $key.inner();
        $key.value.borrow_mut().$action (inner);
        $key
     }}
}
macro_rules! binary_op {
    ($key:ident, $other:ident, $action:ident ) => {{
        let t = $key.new(()).resize_as_($key);
        let inner = $key.inner();
        let rhs = $other.inner();
        t.value.borrow_mut().$action (inner, rhs);
        t
     }}
}
macro_rules! binary_scalar_op {
    ($key:ident, $other:ident, $action:ident ) => {{
        let t = $key.new(()).resize_as_($key);
        let inner = $key.inner();
        t.value.borrow_mut().$action (inner, $other);
        t
     }}
}

macro_rules! binary_inplace_op {
    ($key:ident, $rhs:ident, $action:ident ) => {{
        let inner = $key.inner();
        let rhs = $rhs.inner();
        $key.value.borrow_mut().$action (inner, rhs);
        $key
     }}
}

macro_rules! binary_scalar_inplace_op {
    ($key:ident, $rhs:ident, $action:ident ) => {{
        let inner = $key.inner();
        $key.value.borrow_mut().$action (inner, $rhs);
        $key
     }}
}

macro_rules! gemm_op {
    ($key:ident, $beta:ident, $alpha:ident, $mat1:ident, $mat2:ident, $action:ident ) => {{
        let t = $key.new(()).resize_as_($key);
        let inner = $key.inner();
        let mat1p = $mat1.inner();
        let mat2p = $mat2.inner();
        t.value.borrow_mut().$action ($beta, inner, $alpha, mat1p, mat2p);
        t
     }}
}

macro_rules! gemm_inplace_op {
    ($key:ident, $beta:ident, $alpha:ident, $mat1:ident, $mat2:ident, $action:ident ) => {{
        let inner = $key.inner();
        let mat1p = $mat1.inner();
        let mat2p = $mat2.inner();
        $key.value.borrow_mut().$action ($beta, inner, $alpha, mat1p, mat2p);
        $key
     }}
}

macro_rules! addc_op {
    ($key:ident, $alpha:ident, $mat1:ident, $mat2:ident, $action:ident ) => {{
        let t = $key.new(()).resize_as_($key);
        let inner = $key.inner();
        let mat1p = $mat1.inner();
        let mat2p = $mat2.inner();
        t.value.borrow_mut().$action (inner, $alpha, mat1p, mat2p);
        t
     }}
}

macro_rules! addc_inplace_op {
    ($key:ident, $alpha:ident, $mat1:ident, $mat2:ident, $action:ident ) => {{
        let inner = $key.inner();
        let mat1p = $mat1.inner();
        let mat2p = $mat2.inner();
        $key.value.borrow_mut().$action (inner, $alpha, mat1p, mat2p);
        $key
     }}
}


impl<T: NumLimits> Tensor<T> {
    pub fn abs(&self) -> Self {
        self_op!(self, abs)
    }
    pub fn abs_(&mut self) -> &mut Self {
        self_inplace_op!(self, abs)
    }
    pub fn acos(&self) -> Self {
        self_op!(self, acos)
    }
    pub fn acos_(&mut self) -> &mut Self {
        self_inplace_op!(self, acos)
    }
    pub fn add(&self, rhs: T) -> Self {
        binary_scalar_op!(self, rhs, add)
    }
    pub fn add_(&mut self, rhs: T) -> &mut Self {
        binary_scalar_inplace_op!(self, rhs, add)
    }
    pub fn addt(&self, val: T, rhs: &Self) -> Self {
        let t = self.new(()).resize_as_(self);
        t.value.borrow_mut().addt(self.inner(), val, rhs.inner());
        t
    }
    pub fn addt_(&mut self, val: T, rhs: &Self) -> &mut Self {
        {
            let mut selfcell = self.value.borrow_mut();
            let srcp = selfcell.inner();
            selfcell.addt(srcp, val, rhs.inner());
        }
        self
    }
    pub fn addbmm(&self, beta: T, alpha: T, batch1: &Self, batch2: &Self) -> Self {
        gemm_op!(self, beta, alpha, batch1, batch2, addbmm)
    }
    pub fn addbmm_(&mut self, beta: T, alpha: T, batch1: &Self, batch2: &Self) -> &mut Self {
        gemm_inplace_op!(self, beta, alpha, batch1, batch2, addbmm)
    }
    pub fn addcdiv(&self, value: T, tensor1: &Self, tensor2: &Self) -> Self {
        addc_op!(self, value, tensor1, tensor2, addcdiv)
    }
    pub fn addcdiv_(&mut self, value: T, tensor1: &Self, tensor2: &Self) -> &mut Self {
        addc_inplace_op!(self, value, tensor1, tensor2, addcdiv)
    }
    pub fn addcmul(&self, value: T, tensor1: &Self, tensor2: &Self) -> Self {
        addc_op!(self, value, tensor1, tensor2, addcmul)
    }
    pub fn addcmul_(&mut self, value: T, tensor1: &Self, tensor2: &Self) -> &mut Self {
        addc_inplace_op!(self, value, tensor1, tensor2, addcmul)
    }
    pub fn addmm(&self, beta: T, alpha: T, mat1: &Self, mat2: &Self) -> Self {
        gemm_op!(self, beta, alpha, mat1, mat2, addmm)
    }
    pub fn addmm_(&mut self, beta: T, alpha: T, mat1: &Self, mat2: &Self) -> &mut Self {
        gemm_inplace_op!(self, beta, alpha, mat1, mat2, addmm)
    }
    pub fn addmv(&self, beta: T, alpha: T, tensor1: &Self, vec: &Self) -> Self {
        gemm_op!(self, beta, alpha, tensor1, vec, addmv)
    }
    pub fn addmv_(&mut self, beta: T, alpha: T, tensor1: &Self, vec: &Self) -> &mut Self {
        gemm_inplace_op!(self, beta, alpha, tensor1, vec, addmv)
    }
    pub fn addr(&self, beta: T, alpha: T, vec1: &Self, vec2: &Self) -> Self {
        gemm_op!(self, beta, alpha, vec1, vec2, addr)
    }
    pub fn addr_(&mut self, beta: T, alpha: T, vec1: &Self, vec2: &Self) -> &mut Self {
        gemm_inplace_op!(self, beta, alpha, vec1, vec2, addr)
    }
    pub fn asin(&self) -> Self {
        self_op!(self, asin)
    }
    pub fn asin_(&mut self) -> &mut Self {
        self_inplace_op!(self, asin)
    }
    pub fn atan(&self) -> Self {
        self_op!(self, atan)
    }
    pub fn atan2(&self) -> Self {
        self_op!(self, atan2)
    }
    pub fn atan2_(&mut self) -> &mut Self {
        self_inplace_op!(self, atan2)
    }
    pub fn baddbmm(&self, beta: T, alpha: T, tensor1: &Self, tensor2: &Self) -> Self {
        gemm_op!(self, beta, alpha, tensor1, tensor2, baddbmm)
    }
    pub fn baddbmm_(&mut self, beta: T, alpha: T, tensor1: &Self, tensor2: &Self) -> &mut Self {
        gemm_inplace_op!(self, beta, alpha, tensor1, tensor2, baddbmm)
    }
    pub fn bernoulli(&self, p: f64) -> Self {
        let mut t = self.new(()).resize_as_(self);
        t.bernoulli_(p);
        t
    }
    pub fn bernoulli_(&mut self, p: f64) -> &mut Self {
        self.value.borrow_mut().bernoulli(p);
        self
    }
    pub fn bmm(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn byte(&mut self) -> Tensor<u8> {
        self.cast()
    }
    //
    // cauchy_
    //
    pub fn ceil(&self) -> Self {
        self_op!(self, ceil)
    }
    pub fn ceil_(&mut self) -> &mut Self {
        self_inplace_op!(self, ceil)
    }
    pub fn char(&mut self) -> Tensor<i8> {
        self.cast()
    }
    pub fn chunk(&self, n_chunks: usize, dim: usize) -> Vec<Self> {
        unimplemented!()
    }
    pub fn clamp(&self, min: T, max: T) -> Self {
        let t = self.new(()).resize_as_(self);
        t.value.borrow_mut().clamp(self.inner(), min, max);
        t
    }
    pub fn clamp_(&mut self, min: T, max: T) -> &mut Self {
        self.value.borrow_mut().clamp(self.inner(), min, max);
        self
    }
    pub fn contiguous(&self) -> Self {
        unimplemented!()
    }
    // perform deep copy
    pub fn copy(&self) -> Self {
        let mut t = self.new(()).resize_as_(self);
        t.copy_(self);
        t
    }
    pub fn copy_(&mut self, src: &Self) -> &mut Self {
        self.value.borrow_mut().copy(&src.value);
        self
    }
    pub fn copy_async_(&mut self, src: &Self) -> &mut Self {
        unimplemented!()
    }
    pub fn cos(&self) -> Self {
        self_op!(self, cos)
    }
    pub fn cos_(&mut self) -> &mut Self {
        self_inplace_op!(self, cos)
    }
    pub fn cosh(&self) -> Self {
        self_op!(self, cosh)
    }
    pub fn cosh_(&mut self) -> &mut Self {
        self_inplace_op!(self, cosh)
    }
    pub fn cpu(&self) -> Self {
        self.clone()
    }
    pub fn cross(&self, dim: Option<i32>) -> Self {
        let t = self.new(());
        t.value.borrow_mut().cross(self.inner(), dim);
        t
    }
    pub fn cuda(&self, device: Option<i32>) -> Self {
        unimplemented!()
    }
    pub fn cuda_async(&self, device: Option<i32>) -> Self {
        unimplemented!()
    }
    pub fn diag(&self, diag: u32) -> Self {
        let t = self.new(());
        t.value.borrow_mut().diag(self.inner(), diag);
        t
    }
    pub fn dim(&self) -> i32 {
        self.value.borrow().dim()
    }
    pub fn dist(&self, other: &Self, p: u32) -> f64 {
        self.value.borrow().dist(other.inner(), p)
    }
    pub fn div(&self, value: T) -> Self {
        binary_scalar_op!(self, value, div)
    }
    pub fn div_(&mut self, value: T) -> &mut Self {
        binary_scalar_inplace_op!(self, value, div)
    }
    pub fn divt(&self, value: &Self) -> Self {
        binary_op!(self, value, divt)
    }
    pub fn divt_(&mut self, value: &Self) -> &mut Self {
        binary_inplace_op!(self, value, divt)
    }
    pub fn dot(&self, other: &Self) -> Self {
        binary_op!(self, other, dot)
    }
    pub fn double(&self) -> Tensor<f64> {
        self.cast()
    }
    pub fn eig(&self, eigenvectors: bool) -> (Self, Self) {
        unimplemented!()
    }
    pub fn element_size(&self) -> i32 {
        ::std::mem::size_of::<T>() as i32
    }
    pub fn eq_tensor(&self, other: &Self) -> Tensor<u8> {
        let out: Tensor<u8> = ::torch::tensor(());
        self.value.borrow().eq_tensor(other.inner(), out.inner());
        out
    }
    pub fn eq_tensor_(&mut self, other: &Self) -> &mut Self {
        binary_inplace_op!(self, other, eq_tensor)
    }
    pub fn exp(&self) -> Self {
        self_op!(self, exp)
    }
    pub fn exp_(&mut self) -> &mut Self {
        self_inplace_op!(self, exp)
    }
    pub fn expand<D>(&self, dims: D) -> Self
        where D: AsRef<[usize]>
    {
        self.value.borrow().expand(dims.as_ref())
    }
    pub fn expand_as(&self, tensor: &Self) -> Self {
        self.expand(tensor.size())
    }
    pub fn fill_(&mut self, value: T) -> &mut Self {
        self.value.borrow_mut().fill(value);
        self
    }
    pub fn float(&mut self) -> Tensor<f32> {
        self.cast()
    }
    pub fn floor(&self) -> Self {
        self_op!(self, floor)
    }
    pub fn floor_(&mut self) -> &mut Self {
        self_inplace_op!(self, floor)
    }
    pub fn fmod(&self, divisor: T) -> Self {
        binary_scalar_op!(self, divisor, fmod)
    }
    pub fn fmod_(&mut self, divisor: T) -> &mut Self {
        binary_scalar_inplace_op!(self, divisor, fmod)
    }
    pub fn frac(&self) -> Self {
        self_op!(self, frac)
    }
    pub fn frac_(&mut self) -> &mut Self {
        self_inplace_op!(self, frac)
    }
    pub fn gather(&self, dim: i32, index: Tensor<i64>) {
        unimplemented!()
    }
    pub fn ge_tensor(&self, other: &Self) -> Tensor<u8> {
        let out: Tensor<u8> = ::torch::tensor(());
        self.value.borrow().ge_tensor(other.inner(), out.inner());
        out
    }
    pub fn ge_tensor_(&mut self, other: &Self) -> &mut Self {
        binary_inplace_op!(self, other, ge_tensor)
    }
    pub fn gels(&self, other: &Self) -> Self {
        binary_op!(self, other, gels)
    }
    pub fn gt_tensor(&self, other: &Self) -> Tensor<u8> {
        let out: Tensor<u8> = ::torch::tensor(());
        self.value.borrow().gt_tensor(other.inner(), out.inner());
        out
    }
    pub fn gt_tensor_(&mut self, other: &Self) -> &mut Self {
        binary_inplace_op!(self, other, gt_tensor)
    }
    pub fn half(&mut self) -> Tensor<i16> {
        self.cast()
    }
    pub fn id(&self) -> usize {
        self.value.borrow().inner() as usize
    }
    pub fn index_masked(&self, m: &Tensor<u8>) -> &mut Self {
        unimplemented!()
    }
    pub fn index_add_(&mut self, dim: i32, index: Tensor<i64>, tensor: &Self) -> &mut Self {
        unimplemented!()
    }
    pub fn index_copy_(&mut self, dim: i32, index: Tensor<i64>, tensor: &Self) -> &mut Self {
        unimplemented!()
    }
    pub fn index_fill_(&mut self, dim: i32, index: Tensor<i64>, val: f32) -> &mut Self {
        unimplemented!()
    }
    pub fn index_select(&self, dim: i32, index: Tensor<i64>) -> Self {
        unimplemented!()
    }
    pub fn inner_impl(&self) -> RefMut<TIArg<T>> {
        self.value.borrow_mut()
    }
    pub fn inner(&self) -> *mut ::std::os::raw::c_void {
        self.value.borrow().inner()
    }
    pub fn int(&mut self) -> Tensor<i32> {
        self.cast()
    }
    pub fn is_cuda(&self) -> bool {
        self.value.borrow().is_cuda()
    }
    pub fn is_pinned(&self) -> bool {
        unimplemented!()
    }
    pub fn is_set_to(&self, tensor: &Self) -> bool {
        self.id() == tensor.id()
    }
    pub fn is_signed(&self) -> bool {
        unimplemented!()
    }
    pub fn kthvalue(&self, k: i32, dim: Option<i32>) -> (Self, Tensor<i64>) {
        unimplemented!()
    }
    pub fn le_tensor(&self, other: &Self) -> Tensor<u8> {
        let out: Tensor<u8> = ::torch::tensor(());
        self.value.borrow().le_tensor(other.inner(), out.inner());
        out
    }
    pub fn le_tensor_(&mut self, other: &Self) -> &mut Self {
        binary_inplace_op!(self, other, le_tensor)
    }
    pub fn le_value(&self, value: T) -> Tensor<u8> {
        let out: Tensor<u8> = ::torch::tensor(());
        self.value.borrow().le_value(value, out.inner());
        out
    }
    pub fn lerp(&self, start: &Self, end: &Self, weight: f32) -> Self {
        let t = self.new(());
        t.value
            .borrow_mut()
            .lerp(self.inner(), start.inner(), end.inner(), weight);
        t
    }
    pub fn lerp_(&mut self, start: &Self, end: &Self, weight: f32) -> &mut Self {
        self.value
            .borrow_mut()
            .lerp(self.inner(), start.inner(), end.inner(), weight);
        self
    }
    pub fn log(&self) -> Self {
        self_op!(self, log)
    }
    pub fn log_(&mut self) -> &mut Self {
        self_inplace_op!(self, log)
    }
    pub fn log1p(&self) -> Self {
        self_op!(self, log1p)
    }
    pub fn log1p_(&mut self) -> &mut Self {
        self_inplace_op!(self, log1p)
    }
    //
    // log_normal(...)
    //
    pub fn long(&mut self) -> Tensor<i64> {
        self.cast()
    }
    pub fn lt_tensor(&self, other: &Self) -> Tensor<u8> {
        let out: Tensor<u8> = ::torch::tensor(());
        self.value.borrow().lt_tensor(other.inner(), out.inner());
        out
    }
    pub fn lt_tensor_(&mut self, other: &Self) -> &mut Self {
        binary_inplace_op!(self, other, lt_tensor)
    }
    pub fn lt_value(&self, value: T) -> Tensor<u8> {
        let out: Tensor<u8> = ::torch::tensor(());
        self.value.borrow().lt_value(value, out.inner());
        out
    }
    //
    // map_
    //
    pub fn masked_fill_(&mut self, mask: Tensor<u8>, value: T) -> &mut Self {
        self.value
            .borrow_mut()
            .masked_fill(self.inner(), mask.inner(), value);
        self
    }
    pub fn masked_fill(&self, mask: Tensor<u8>, value: T) -> Self {
        let t = self.new(());
        t.value
            .borrow_mut()
            .masked_fill(self.inner(), mask.inner(), value);
        t
    }
    pub fn masked_scatter_(&mut self, mask: Tensor<u8>, source: &Self) -> &mut Self {
        self.value
            .borrow_mut()
            .masked_scatter(self.inner(), mask.inner(), source.inner());
        self
    }
    pub fn masked_select(&self, mask: Tensor<u8>) -> Self {
        let t = self.new(());
        t.value
            .borrow_mut()
            .masked_select(self.inner(), mask.inner());
        t
    }
    pub fn max(&self) -> T {
        self.value.borrow().max()
    }
    pub fn max_reduce(&self, dim: usize, keepdim: bool) -> (Self, Tensor<i64>) {
        let mut dims = self.size();
        dims[dim] = 1;
        let values = self.new(()).resize_(&dims);
        let indices: Tensor<i64> = ::torch::tensor(()).resize_(&dims);
        self.value
            .borrow()
            .max_reduce(values.inner(), indices.inner(), dim, keepdim);
        (values, indices)
    }
    pub fn mean(&self) -> f64 {
        self.value.borrow().mean()
    }
    pub fn mean_reduce(&self, dim: i32) -> (Self, Tensor<i64>) {
        unimplemented!()
    }
    //
    // median
    //
    pub fn min(&self) -> T {
        self.value.borrow().min()
    }
    pub fn min_reduce(&self, dim: usize, keepdim: bool) -> (Self, Tensor<i64>) {
        let mut dims = self.size();
        dims[dim] = 1;
        let values = self.new(()).resize_(&dims);
        let indices: Tensor<i64> = ::torch::tensor(()).resize_(&dims);
        self.value
            .borrow()
            .min_reduce(values.inner(), indices.inner(), dim, keepdim);
        (values, indices)
    }
    pub fn mm(&self, rhs: &Self) -> Self {
        let out = self.new([self.size()[0], rhs.size()[1]]);
        out.value.borrow_mut().mm(self.inner(), rhs.inner());
        out
    }
    //
    // mode
    //
    pub fn mul(&self, rhs: T) -> Self {
        binary_scalar_op!(self, rhs, mul)
    }
    pub fn mul_(&mut self, rhs: T) -> &mut Self {
        binary_scalar_inplace_op!(self, rhs, mul)
    }
    pub fn mult(&self, rhs: &Self) -> Self {
        binary_op!(self, rhs, mult)
    }
    pub fn mult_(&mut self, rhs: &Self) -> &mut Self {
        binary_inplace_op!(self, rhs, mult)
    }
    //
    // multinomial
    //
    pub fn mv(&self, vec: &Self) -> Self {
        binary_op!(self, vec, mv)
    }
    pub fn narrow(&self, dim: i32, start: i32, length: i32) -> Self {
        let t = self.new(());
        t.value
            .borrow_mut()
            .narrow(self.inner(), dim, start, length);
        t
    }
    pub fn ndimension(&self) -> i32 {
        self.dim()
    }
    pub fn ne_tensor(&self, other: &Self) -> Tensor<u8> {
        let out: Tensor<u8> = ::torch::tensor(());
        self.value.borrow().ne_tensor(other.inner(), out.inner());
        out
    }
    pub fn ne_tensor_(&mut self, other: &Self) -> &mut Self {
        binary_inplace_op!(self, other, ne_tensor)
    }
    pub fn neg(&self) -> Self {
        self_op!(self, neg)
    }
    pub fn neg_(&mut self) -> &mut Self {
        self_inplace_op!(self, neg)
    }
    pub fn nonzero(&self) -> Tensor<i64> {
        unimplemented!()
    }
    pub fn norm(&self, p: i32) -> f64 {
        self.value.borrow().norm(p)
    }
    //
    // normal_
    //
    pub fn numel(&self) -> usize {
        self.size().iter().product()
    }
    //
    // numpy() (need native tensor equivalent - rust-ndarray?)
    //
    //
    // orgqr
    //
    // ormqr
    //
    pub fn permute(&self, dims: &[u32]) -> Self {
        unimplemented!()
    }
    pub fn pin_memory(&mut self) -> &mut Self {
        self_inplace_op!(self, pin_memory)
    }
    //
    // potrf
    //
    //
    // potri
    //
    //
    // potrs
    //
    pub fn pow(&self) -> Self {
        self_op!(self, pow)
    }
    pub fn pow_(&mut self) -> &mut Self {
        self_inplace_op!(self, pow)
    }
    pub fn prod<R: NumLimits>(&self) -> R {
        let mut result = 0.;
        self.value.borrow().prod(&mut result);
        <R as ::num::NumCast>::from(result).unwrap()
    }
    //
    // pstrf
    //
    //
    // qr
    //
    //
    // random_
    //
    pub fn reciprocal(&self) -> Self {
        self_op!(self, reciprocal)
    }
    pub fn reciprocal_(&mut self) -> &mut Self {
        self_inplace_op!(self, reciprocal)
    }
    pub fn remainder(&self, divisor: T) -> Self {
        binary_scalar_op!(self, divisor, remainder)
    }
    pub fn remainder_(&mut self, divisor: T) -> &mut Self {
        binary_scalar_inplace_op!(self, divisor, remainder)
    }
    //
    // renorm
    //
    //
    // renorm_
    //
    pub fn repeat(&self, sizes: &[i32]) -> Self {
        // NB: copies data
        unimplemented!()
    }
    pub fn resize_<D>(&mut self, sizes: D) -> Self
        where D: AsRef<[usize]>
    {
        self.value.borrow_mut().resize(sizes.as_ref());
        self.clone()
    }
    pub fn resize_as_(&mut self, tensor: &Self) -> Self {
        self.resize_(tensor.size())
    }
    pub fn round(&self) -> Self {
        self_op!(self, round)
    }
    pub fn round_(&mut self) -> &mut Self {
        self_inplace_op!(self, round)
    }
    pub fn rsqrt(&self) -> Self {
        self_op!(self, rsqrt)
    }
    pub fn rsqrt_(&mut self) -> &mut Self {
        self_inplace_op!(self, rsqrt)
    }
    pub fn scatter_(&mut self, dim: i32, index: Tensor<i64>, src: &Self) -> &mut Self {
        unimplemented!()
    }
    pub fn select(&self, dim: i32, index: i32) -> Self {
        let t = self.new(());
        t.value.borrow().select(self.inner(), dim, index);
        t
    }
    //
    // set_
    //
    //
    // share_memory_
    //
    pub fn short(&mut self) -> Tensor<u16> {
        self.cast()
    }
    pub fn sigmoid(&self) -> Self {
        self_op!(self, sigmoid)
    }
    pub fn sigmoid_(&mut self) -> &mut Self {
        self_inplace_op!(self, sigmoid)
    }
    pub fn sign(&self) -> Self {
        self_op!(self, sign)
    }
    pub fn sign_(&mut self) -> &mut Self {
        self_inplace_op!(self, sign)
    }
    pub fn sin(&self) -> Self {
        self_op!(self, sin)
    }
    pub fn sin_(&mut self) -> &mut Self {
        self_inplace_op!(self, sin)
    }
    pub fn sinh(&self) -> Self {
        self_op!(self, sinh)
    }
    pub fn sinh_(&mut self) -> &mut Self {
        self_inplace_op!(self, sinh)
    }
    pub fn size(&self) -> Vec<usize> {
        self.value.borrow().size()
    }
    pub fn sort(&self, dim: Option<i32>, descending: bool) -> (Self, Tensor<i64>) {
        unimplemented!()
    }
    pub fn sqrt(&self) -> Self {
        self_op!(self, sqrt)
    }
    pub fn sqrt_(&mut self) -> &mut Self {
        self_inplace_op!(self, sqrt)
    }
    pub fn squeeze(&self, dim: Option<usize>) -> Self {
        let mut t = self.copy();
        t.squeeze_(dim);
        t
    }
    pub fn squeeze_(&mut self, dim: Option<usize>) -> &mut Self {
        self.value.borrow_mut().squeeze(dim);
        self
    }
    pub fn std(&self) -> f64 {
        self.value.borrow().std()
    }
    //
    // storage
    //
    //
    // storage_offset
    //
    pub fn stride(&self) -> Vec<i32> {
        self.value.borrow().stride()
    }
    pub fn sub(&self, rhs: &Self) -> Self {
        binary_op!(self, rhs, sub)
    }
    pub fn sub_(&mut self, rhs: &Self) -> &mut Self {
        binary_inplace_op!(self, rhs, sub)
    }
    pub fn sum<R: NumLimits>(&self) -> R {
        let mut result = 0.;
        self.value.borrow().sum_float(&mut result);
        <R as ::num::NumCast>::from(result).unwrap()
    }
    pub fn sum_reduce(&self, dim: usize, keepdim: bool) -> Self {
        let mut dims = self.size();
        dims[dim] = 1;
        let out = self.new(dims);
        out.value
            .borrow_mut()
            .sum_reduce(self.inner(), dim, keepdim);
        out
    }
    pub fn svd(&self, some: bool) -> (Self, Self, Self) {
        unimplemented!()
    }
    //
    // symeig
    //
    pub fn t(&self) -> Self {
        self.transpose(0, 1)
    }
    pub fn t_(&mut self) -> &mut Self {
        self.transpose_(0, 1)
    }
    pub fn tan(&self) -> Self {
        self_op!(self, tan)
    }
    pub fn tan_(&mut self) -> &mut Self {
        self_inplace_op!(self, tan)
    }
    pub fn tanh(&self) -> Self {
        self_op!(self, tanh)
    }
    pub fn tanh_(&mut self) -> &mut Self {
        self_inplace_op!(self, tanh)
    }
    //
    // tolist
    //
    pub fn topk(k: i32, dim: Option<i32>, largest: bool, sorted: bool) -> (Self, Tensor<i64>) {
        unimplemented!()
    }
    pub fn trace(&self) -> Self {
        self_op!(self, trace)
    }
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        let t = self.new(()).resize_as_(self);
        t.value.borrow_mut().transpose(self.inner(), dim0, dim1);
        t
    }
    pub fn transpose_(&mut self, dim0: usize, dim1: usize) -> &mut Self {
        let p = self.inner();
        self.value.borrow_mut().transpose(p, dim0, dim1);
        self
    }
    //
    // tril
    //
    //
    // tril_
    //
    //
    // triu
    //
    //
    // tril_
    //
    //
    // trtrs
    //
    pub fn trunc(&self) -> Self {
        self_op!(self, trunc)
    }
    pub fn trunc_(&mut self) -> &mut Self {
        self_inplace_op!(self, trunc)
    }
    pub fn type_as(&self, tensor: &Self) -> Self {
        unimplemented!()
    }
    pub fn typecast(&self, new_type: TensorType, async: bool) -> Self {
        unimplemented!()
    }
    pub fn unfold(&self, dim: i32, size: i32, step: i32) -> Self {
        let t = self.new(());
        t.value.borrow_mut().unfold(self.inner(), dim, size, step);
        t
    }
    pub fn uniform_(&mut self, range: (f64, f64)) -> &mut Self {
        self.value.borrow_mut().uniform(range);
        self
    }
    pub fn unsqueeze(&self, dim: usize) -> Self {
        let mut t = self.copy();
        t.unsqueeze_(dim);
        t
    }
    pub fn unsqueeze_(&mut self, dim: usize) -> &mut Self {
        self.value.borrow_mut().unsqueeze(dim);
        self
    }
    pub fn validate(&self, arg: &str) {
        println!("validate - {}: {}", arg, self.sum() : f64);
        assert_eq!(self.is_valid(), true);
    }
    pub fn var(&self) -> f64 {
        self.value.borrow().var()
    }
    pub fn view<D>(&self, dims: D) -> Self
        where D: AsRef<[isize]>
    {
        self.value.borrow_mut().view(dims.as_ref())
    }
    pub fn view_as(&self, tensor: &Self) -> Self {
        let dims: Vec<isize> = tensor.size().iter().map(|t| *t as isize).collect();
        self.view(dims)
    }
    pub fn zero_(&mut self) -> Self {
        self.value.borrow_mut().zero();
        self.clone()
    }
}
