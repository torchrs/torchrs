
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
