#![allow(unused_variables)]
use tensor::*;
use std::cell::RefMut;


impl<T: NumLimits> Tensor<T> {
    pub fn abs(&self) -> Self {
        unimplemented!()
    }
    pub fn abs_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn acos(&self) -> Self {
        unimplemented!()
    }
    pub fn acos_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn add(&self, rhs: T) -> Self {
        let mut t = self.copy();
        t.value
            .borrow_mut()
            .add(self.value.borrow_mut().inner(), rhs);
        t
    }
    pub fn add_(&mut self, rhs: T) -> &mut Self {
        {
            let mut selfcell = self.value.borrow_mut();
            let srcp = selfcell.inner();
            selfcell.add(srcp, rhs);
        }
        self
    }
    pub fn addt(&self, val: T, rhs: &Self) -> Self {
        let mut t = self.copy();
        t.value.borrow_mut().addt(self.inner(), rhs.inner());
        t
    }
    pub fn addt_(&mut self, val: T, rhs: &Self) -> &mut Self {
        {
            let mut selfcell = self.value.borrow_mut();
            let srcp = selfcell.inner();
            selfcell.addt(srcp, rhs.inner());
        }
        self
    }
    pub fn addbmm(&self, beta: T, alpha: T, batch1: &Self, batch2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addbmm_(&mut self, beta: T, alpha: T, batch1: &Self, batch2: &Self) -> &mut Self {
        unimplemented!()
    }
    pub fn addcdiv(&self, value: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addcdiv_(&mut self, value: T, tensor1: &Self, tensor2: &Self) -> &mut Self {
        unimplemented!()
    }
    pub fn addcmul(&self, value: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addcmul_(&mut self, value: T, tensor1: &Self, tensor2: &Self) -> &mut Self {
        unimplemented!()
    }
    pub fn addmm(&self, beta: T, alpha: T, mat1: &Self, mat2: &Self) -> Self {
        let mut t = self.new(());
        t.value
            .borrow_mut()
            .addmm(self.inner(), beta, alpha, mat1.inner(), mat2.inner());
        t
    }
    pub fn addmm_(&mut self, beta: T, alpha: T, mat1: &Self, mat2: &Self) -> &mut Self {
        let p = self.inner();
        self.value
            .borrow_mut()
            .addmm(p, beta, alpha, mat1.inner(), mat2.inner());
        self
    }
    pub fn addmv(&self, beta: T, alpha: T, tensor1: &Self, vec: &Self) -> Self {
        unimplemented!()
    }
    pub fn addmv_(&mut self, beta: T, alpha: T, tensor1: &Self, vec: &Self) -> &mut Self {
        unimplemented!()
    }
    pub fn addr(&self, beta: T, alpha: T, vec1: &Self, vec2: &Self) -> Self {
        unimplemented!()
    }
    pub fn addr_(&mut self, beta: T, alpha: T, vec1: &Self, vec2: &Self) -> &mut Self {
        unimplemented!()
    }
    pub fn asin(&self) -> Self {
        unimplemented!()
    }
    pub fn asin_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn atan(&self) -> Self {
        unimplemented!()
    }
    pub fn atan2(&self) -> Self {
        unimplemented!()
    }
    pub fn atan2_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn baddbmm(&self, beta: T, alpha: T, tensor1: &Self, tensor2: &Self) -> Self {
        unimplemented!()
    }
    pub fn baddbmm_(&mut self, beta: T, alpha: T, tensor1: &Self, tensor2: &Self) -> &mut Self {
        unimplemented!()
    }
    pub fn bernoulli(&self, p: f64) -> Self {
        let mut t = self.new(());
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
    pub fn byte(&mut self) -> &mut Self {
        unimplemented!()
    }
    //
    // cauchy_
    //
    pub fn ceil(&self) -> Self {
        unimplemented!()
    }
    pub fn ceil_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn char(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn chunk(&self, n_chunks: usize, dim: usize) -> Vec<Self> {
        unimplemented!()
    }
    pub fn clamp(&self, min: T, max: T) -> Self {
        unimplemented!()
    }
    pub fn clamp_(&mut self, min: T, max: T) -> &mut Self {
        unimplemented!()
    }
    pub fn contiguous(&self) -> Self {
        unimplemented!()
    }
    // perform deep copy
    pub fn copy(&self) -> Self {
        let mut t = self.new(());
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
        unimplemented!()
    }
    pub fn cos_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn cosh(&self) -> Self {
        unimplemented!()
    }
    pub fn cosh_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn cpu(&self) -> Self {
        unimplemented!()
    }
    pub fn cross(&self, dim: Option<i32>) -> Self {
        unimplemented!()
    }
    pub fn cuda(&self, device: Option<i32>) -> Self {
        unimplemented!()
    }
    pub fn cuda_async(&self, device: Option<i32>) -> Self {
        unimplemented!()
    }
    pub fn diag(&self, diag: u32) -> Self {
        unimplemented!()
    }
    pub fn dim(&self) -> i32 {
        self.value.borrow().dim()
    }
    pub fn dist(&self, other: &Self, p: u32) -> f32 {
        unimplemented!()
    }
    pub fn div(&self, value: T) -> Self {
        let mut t = self.copy();
        t.value
            .borrow_mut()
            .div(self.value.borrow_mut().inner(), value);
        t
    }
    pub fn div_(&mut self, value: T) -> &mut Self {
        {
            let mut selfcell = self.value.borrow_mut();
            let srcp = selfcell.inner();
            selfcell.div(srcp, value);
        }
        self
    }
    pub fn divt(&self, value: &Self) -> Self {

        unimplemented!()
    }
    pub fn divt_(&mut self, value: &Self) -> &mut Self {
        unimplemented!()
    }
    pub fn dot(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn double(&self) -> Self {
        unimplemented!()
    }
    pub fn eig(&self, eigenvectors: bool) -> (Self, Self) {
        unimplemented!()
    }
    pub fn element_size(&self) -> i32 {
        unimplemented!()
    }
    pub fn eq_tensor(&self, other: &Self) -> Tensor<u8> {
        unimplemented!()
    }
    pub fn eq_tensor_(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn exp(&self) -> Self {
        unimplemented!()
    }
    pub fn exp_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn expand<D>(&self, dims: D) -> Self
        where D: AsRef<[usize]>
    {
        self.value.borrow().expand(dims.as_ref())
    }
    pub fn expand_as(&self, tensor: &Self) -> Self {
        self.expand(tensor.size())
    }
    pub fn fill_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn float(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn floor(&self) -> Self {
        unimplemented!()
    }
    pub fn floor_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn fmod(&self, divisor: T) -> Self {
        unimplemented!()
    }
    pub fn fmod_(&mut self, divisor: T) -> &mut Self {
        unimplemented!()
    }
    pub fn frac(&self) -> Self {
        unimplemented!()
    }
    pub fn frac_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn gather(&self, dim: i32, index: Tensor<i64>) {
        unimplemented!()
    }
    pub fn ge_tensor(&self, other: &Self) -> Tensor<u8> {
        unimplemented!()
    }
    pub fn ge_tensor_(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn gels(&self, other: &Self) -> Self {
        unimplemented!();
    }
    pub fn gt_tensor(&self, other: &Self) -> Tensor<u8> {
        unimplemented!()
    }
    pub fn gt_tensor_(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn half(&mut self) -> &mut Self {
        unimplemented!()
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
    pub fn int(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn is_cuda(&self) -> bool {
        unimplemented!();
        false
    }
    pub fn is_pinned(&self) -> bool {
        unimplemented!()
    }
    pub fn is_set_to(&self, tensor: &Self) -> bool {
        unimplemented!()
    }
    pub fn is_signed(&self) -> bool {
        unimplemented!()
    }
    pub fn kthvalue(&self, k: i32, dim: Option<i32>) -> (Self, Tensor<i64>) {
        unimplemented!()
    }
    pub fn le_tensor(&self, other: &Self) -> Tensor<u8> {
        unimplemented!()
    }
    pub fn le_tensor_(&self, other: &Self) -> Self {
        unimplemented!()
    }
    pub fn lerp(&self, start: &Self, end: &Self, weight: f32) -> Self {
        unimplemented!()
    }
    pub fn lerp_(&self, start: &Self, end: &Self, weight: f32) -> Self {
        unimplemented!()
    }
    pub fn log(&self) -> Self {
        unimplemented!()
    }
    pub fn log_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn log1p(&self) -> Self {
        unimplemented!()
    }
    pub fn log1p_(&mut self) -> &mut Self {
        unimplemented!()
    }
    //
    // log_normal(...)
    //
    pub fn long(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn lt_tensor(&self, other: &Self) -> Tensor<u8> {
        unimplemented!()
    }
    pub fn lt_tensor_(&mut self, other: &Self) -> &mut Self {
        unimplemented!()
    }
    //
    // map_
    //
    pub fn masked_copy_(&mut self, mask: Tensor<u8>, source: &Self) -> &mut Self {
        unimplemented!()
    }
    pub fn masked_select(&self, mask: Tensor<u8>) -> Self {
        unimplemented!()
    }
    pub fn max(&self) -> T {
        unimplemented!()
    }
    pub fn max_reduce(&self, dim: i32) -> (Self, Tensor<i64>) {
        unimplemented!()
    }
    pub fn mean(&self) -> T {
        unimplemented!()
    }
    pub fn mean_reduce(&self, dim: i32) -> (Self, Tensor<i64>) {
        unimplemented!()
    }
    //
    // median
    //
    pub fn min(&self) -> T {
        unimplemented!()
    }
    pub fn min_reduce(&self, dim: i32) -> (Self, Tensor<i64>) {
        unimplemented!()
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
        let mut t = self.copy();
        t.value.borrow_mut().mul(self.value.borrow().inner(), rhs);
        t
    }
    pub fn mul_(&mut self, rhs: T) -> &mut Self {
        {
            let mut selfcell = self.value.borrow_mut();
            let srcp = selfcell.inner();
            selfcell.mul(srcp, rhs);
        }
        self
    }
    pub fn mult(&self, rhs: &Self) -> Self {
        let mut t = self.copy();
        t.value.borrow_mut().mult(self.inner(), rhs.inner());
        t
    }
    pub fn mult_(&mut self, rhs: &Self) -> &mut Self {
        {
            let mut selfcell = self.value.borrow_mut();
            let srcp = selfcell.inner();
            selfcell.mult(srcp, rhs.inner());
        }
        self
    }
    //
    // multinomial
    //
    pub fn mv(&self, vec: &Self) -> Self {
        unimplemented!()
    }
    pub fn narrow(&self, dim: i32, start: i32, length: i32) -> Self {
        unimplemented!()
    }
    pub fn ndimension(&self) -> i32 {
        unimplemented!()
    }
    pub fn ne_tensor(&self, other: &Self) -> Tensor<u8> {
        unimplemented!()
    }
    pub fn ne_tensor_(&mut self, other: &Self) -> &mut Self {
        unimplemented!()
    }
    pub fn neg(&self) -> Self {
        unimplemented!()
    }
    pub fn neg_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn nonzero(&self) -> Tensor<i64> {
        unimplemented!()
    }
    pub fn norm(&self, p: i32) -> f32 {
        unimplemented!()
    }
    //
    // normal_
    //
    pub fn numel(&self) -> i32 {
        unimplemented!()
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
        unimplemented!()
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
        unimplemented!()
    }
    pub fn pow_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn prod<R: NumLimits>(&self) -> R {
        unimplemented!()
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
        unimplemented!()
    }
    pub fn reciprocal_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn remainder(&self, divisor: T) -> Self {
        unimplemented!()
    }
    pub fn remainder_(&mut self, divisor: T) -> &mut Self {
        unimplemented!()
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
    pub fn resize_(&mut self, sizes: &[i32]) -> &mut Self {
        unimplemented!()
    }
    pub fn resize_as_(&mut self, tensor: &Self) -> &mut Self {
        unimplemented!()
    }
    pub fn round(&self) -> Self {
        unimplemented!()
    }
    pub fn round_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn rsqrt(&self) -> Self {
        unimplemented!()
    }
    pub fn rsqrt_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn scatter_(&mut self, dim: i32, index: Tensor<i64>, src: &Self) -> &mut Self {
        unimplemented!()
    }
    pub fn select(&self, dim: i32, index: i32) -> Self {
        unimplemented!()
    }
    //
    // set_
    //
    //
    // share_memory_
    //
    pub fn short(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn sigmoid(&self) -> Self {
        unimplemented!()
    }
    pub fn sigmoid_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn sign(&self) -> Self {
        unimplemented!()
    }
    pub fn sign_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn sin(&self) -> Self {
        unimplemented!()
    }
    pub fn sin_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn sinh(&self) -> Self {
        unimplemented!()
    }
    pub fn sinh_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn size(&self) -> Vec<usize> {
        self.value.borrow().size()
    }
    pub fn sort(&self, dim: Option<i32>, descending: bool) -> (Self, Tensor<i64>) {
        unimplemented!()
    }
    pub fn sqrt(&self) -> Self {
        unimplemented!()
    }
    pub fn sqrt_(&mut self) -> &mut Self {
        unimplemented!()
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
    pub fn std(&self) -> f32 {
        unimplemented!()
    }
    //
    // storage
    //
    //
    // storage_offset
    //
    pub fn stride(&self) -> Vec<i32> {
        unimplemented!()
    }
    pub fn sub(&self, rhs: &Self) -> Self {
        unimplemented!()
    }
    pub fn sub_(&mut self, rhs: &Self) -> &mut Self {
        unimplemented!()
    }
    pub fn sum<R: NumLimits>(&self) -> R {
        unimplemented!()
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
        unimplemented!()
    }
    pub fn tan_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn tanh(&self) -> Self {
        unimplemented!()
    }
    pub fn tanh_(&mut self) -> &mut Self {
        unimplemented!()
    }
    //
    // tolist
    //
    pub fn topk(k: i32, dim: Option<i32>, largest: bool, sorted: bool) -> (Self, Tensor<i64>) {
        unimplemented!()
    }
    pub fn trace(&self) -> Self {
        unimplemented!()
    }
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        let t = self.new(());
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
        unimplemented!()
    }
    pub fn trunc_(&mut self) -> &mut Self {
        unimplemented!()
    }
    pub fn type_as(&self, tensor: &Self) -> Self {
        unimplemented!()
    }
    pub fn typecast(&self, new_type: TensorType, async: bool) -> Self {
        unimplemented!()
    }
    pub fn unfold(&self, dim: i32, size: i32, step: i32) -> Self {
        unimplemented!()
    }
    pub fn uniform_(&mut self, range: (f64, f64)) -> &mut Self {
        self.value.borrow_mut().uniform_(range);
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
    pub fn var(&self) -> f32 {
        unimplemented!()
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
    pub fn zero_(&mut self) -> &mut Self {
        self.value.borrow_mut().zero();
        self
    }
}
