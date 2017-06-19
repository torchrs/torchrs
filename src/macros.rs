
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
