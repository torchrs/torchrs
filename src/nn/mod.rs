

pub trait ModuleStruct<'a> {
    fn init_module(&mut self);
    fn get_module(&mut self, name: &str) -> Option<&mut ModIntf<'a>>;
}

struct TorchBackend {}
struct Parameter {}

pub struct Module<'a> {
    _name: &'a str,
    _backend: TorchBackend,
    //	_buffers: HashTable<&str, Tensor>
    //	_backward_hooks:
    //	_forward_hooks:
    _params: Vec<&'a str>,
    _modules: Vec<&'a str>, //LinkedHashMap<String, &'a Module<'a>  >,
    training: bool,
}

impl<'a> Module<'a> {
    pub fn new() -> Module<'a> {
        Module {
            _name: "",
            _backend: TorchBackend {},
            _params: Vec::new(),
            _modules: Vec::new(),
            training: true,
        }
    }
}

pub trait ModIntf<'a> {
    fn delegate(&mut self) -> &mut Module<'a>;
    fn forward(&mut self);
}

#[derive(ModParse)]
struct Linear<'a> {
    delegate: Module<'a>,
    in_features: u32,
}

impl<'a> Linear<'a> {
    pub fn new() -> Linear<'a> {
        let mut t = Linear {
            delegate: Module::new(),
            in_features: 0,
        };
        t.init_module();
        t
    }
}

impl<'a> ModIntf<'a> for Linear<'a> {
    fn delegate(&mut self) -> &mut Module<'a> {
        &mut self.delegate
    }
    fn forward(&mut self) {}
}

#[derive(ModParse)]
pub struct MyMod<'a> {
    delegate: Module<'a>,
    #[module]
    a: Linear<'a>,
}

impl<'a> MyMod<'a> {
    pub fn new() -> MyMod<'a> {
        let mut t = MyMod {
            delegate: Module::new(),
            a: Linear::new(),
        };
        t.init_module();
        t
    }
}
