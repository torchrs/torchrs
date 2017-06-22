
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
        impl<T:Default+Copy> ModDelegate<T> for $name<T> {
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
