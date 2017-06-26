
#[macro_export]
macro_rules! impl_func_delegate {
	($name:ident) => (
		impl FuncDelegate for $name {
		    fn delegate(&mut self) -> &mut Function {
    		    &mut self.delegate
    		}
		}

	)
}

#[macro_export]
macro_rules! impl_mod_delegate {
	($name:ident) => (
        impl<T: ::tensor::NumLimits<T>> ModDelegate<T> for $name<T> {
		    fn delegate(&mut self) -> &mut Module<T> {
    		    &mut self.delegate
    		}
            fn params_iter_mut(&mut self) -> ::std::vec::IntoIter<Variable<T>> {
                let mut v : Vec<Variable<T>> = Vec::new();
                for name in self.delegate()._params.clone().iter() {
                    if let Some(id) = self.get_param(name) {
                        v.push(id.into())
                    }
                }
                v.into_iter()
            }
    fn _apply(&mut self, callback: fn(&mut ::tensor::Tensor<T>)) {
        let mod_names = self.delegate()._modules.clone();
        for name in mod_names {
            let module = self.get_module(name);
            module._apply(callback)
        }
        for mut param in self.params_iter_mut() {
            param.apply(callback);
            if let &mut Some(ref mut g) = param.grad() {
                g.apply(callback)
            }
            /* see also _buffers */
        }
    }
    fn apply(&mut self, callback: fn(&mut ModIntf<T>)) {
        let mod_names = self.delegate()._modules.clone();
        for name in mod_names {
            let module = self.get_module(name);
            module.apply(callback)
        }
        callback(self)
    }
    }

	)
}

#[macro_export]
macro_rules! typecast {
    ($value:expr, $type:ident) => {$value as *mut $type}
}

#[macro_export]
macro_rules! impl_tensor_impl {
    ($name:ident, $type:ident, $thname:ident) => {
    impl TensorImpl<$type> for $name {
    fn new(&self) -> RefTI<$type> {
        RcMutNew($name ::new())
    }
    fn add(&self, value: $type, output: &TIArg<$type>) {
        let out = typecast!(output.inner(), $thname);
        unsafe {
            concat_idents!(TH, $name, _add)(out, self.t, value);
        };
    }
    fn inner(&self) -> *mut ::std::os::raw::c_void {
        self.t as *mut ::std::os::raw::c_void
    }
    fn addt(&self, value: $type, rhs: &TIArg<$type>, output: &TIArg<$type>) {
        let out = typecast!(output.inner(), $thname);
        let rhsp = typecast!(rhs.inner(), $thname);
        unsafe {
            concat_idents!($thname, _add)(out, rhsp, value);
        };
    }
    }
    impl Default for $name {
        fn default() -> Self {
            $name ::new()
        }
    }

    impl<'a> Index<&'a [isize]> for $name {
    type Output = $type;

    fn index(&self, idx: &'a [isize]) -> &Self::Output {
        let mut index: isize = 0;
        let lastidx = max(0, idx.len() as isize - 1) as usize;
        if idx.len() != self.dims.len() {
            panic!("bad dimlen")
        }
        for i in 0..lastidx {
            if idx[i] >= self.dims[i] {
                panic!("bad dimlen")
            }
            index += idx[i] * self.dims[i]
        }
        if idx[lastidx] >= self.dims[lastidx] {
            panic!("bad dimlen")
        }
        index += idx[lastidx];
        &self.storage[index]
    }
    }

    impl<'a> IndexMut<&'a [isize]> for $name {
    fn index_mut(&mut self, idx: &'a [isize]) -> &mut Self::Output {
        let mut index: isize = 0;
        let lastidx = max(0, idx.len() as isize - 1) as usize;
        if idx.len() != self.dims.len() {
            panic!("bad dimlen")
        }
        for i in 0..lastidx {
            if idx[i] >= self.dims[i] {
                panic!("bad dimlen")
            }
            index += idx[i] * self.dims[i]
        }
        if idx[lastidx] >= self.dims[lastidx] {
            panic!("bad dimlen")
        }
        index += idx[lastidx];
        &mut self.storage[index]
    }
    }
    impl Index<isize> for $name {
    type Output = $type;
    fn index(&self, idx: isize) -> &Self::Output {
        unimplemented!()
    }
    }
    impl Drop for $name {
        fn drop(&mut self) {
            unsafe { concat_idents!($thname, _free)(self.t) }
        }
    }
    impl Serialize for $name {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        unimplemented!()
    }
    }
    impl<'de> Deserialize<'de> for $name {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>
    {
        unimplemented!()
    }
    }
    }
}

#[macro_export]
macro_rules! impl_func {
	($name:ident) => (
		#[derive(Clone)]
		pub struct $name {
    		delegate: Function,
		}
		impl $name {
    		pub fn new() -> FIWrap<Self> {
        		FIWrap::new($name { delegate: Function::new() })
    		}
		}
		impl FuncDelegate for $name {
		    fn delegate(&mut self) -> &mut Function {
    		    &mut self.delegate
    		}
		}

	)
}


#[macro_export]
macro_rules! impl_func_args {
	($name:ident, $args:ident) => (

#[derive(Clone)]
pub struct $name {
    delegate: Function,
    args: $args,
}

impl $name {
    pub fn new(args: & $args) -> FIWrap<Self> {
        FIWrap::new($name {
                        delegate: Function::new(),
                        args: args.clone(),
                    })
    }
}

impl FuncDelegate for $name {
    fn delegate(&mut self) -> &mut Function {
        &mut self.delegate
    }
}
)}



#[macro_export]
macro_rules! map(
    { $($key:expr => $value:expr),+ } => {
        {
            let mut m = ::std::collections::HashMap::new();
            $(
                m.insert($key, $value);
            )+
            m
        }
     };
);

#[macro_export]
macro_rules! map_opt(
    { $($key:expr => $value:expr),+ } => {
        {
            let mut m = ::std::collections::HashMap::<&'static str, OptimVal>::new();
            $(
                m.insert($key, $value .into());
            )+
            m
        }
     };
);
