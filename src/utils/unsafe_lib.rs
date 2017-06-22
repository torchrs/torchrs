use std::collections::HashMap;
use std::cell::{Cell, RefCell};
use std::hash::Hash;
use std::ops::{Index, IndexMut};
use std::fmt::Debug;

pub struct MutMap<K, V: Default> {
    map: HashMap<K, RefCell<V>>,
}

impl<K: Eq + Hash, V: Default> MutMap<K, V> {
    pub fn new() -> Self {
        MutMap { map: HashMap::new() }
    }
}

impl<K: Hash + Eq + Clone + Debug, V: Default> Index<K> for MutMap<K, V> {
    type Output = V;
    fn index(&self, idx: K) -> &Self::Output {
        let map = &self.map;
        if !map.contains_key(&idx) {
            panic!("{:?} not found", idx)
        }
        let cntp = map[&idx].as_ptr();
        unsafe { &*cntp }
    }
}
impl<K: Hash + Eq + Clone + Debug, V: Default> IndexMut<K> for MutMap<K, V> {
    fn index_mut(&mut self, idx: K) -> &mut Self::Output {
        let map = &mut self.map;
        if !map.contains_key(&idx) {
            map.insert(idx.clone(), RefCell::new(V::default()));
        }
        let cntp = map[&idx].as_ptr();
        unsafe { &mut *cntp }
    }
}


// Pythonesque Counter implementation
// XXX Move to a separate module
static ZERO: usize = 0;
pub struct Counter<T: Hash + Eq + Clone> {
    pub map: HashMap<T, Cell<usize>>,
}
impl<T: Hash + Eq + Clone> Counter<T> {
    pub fn new() -> Self {
        Counter { map: HashMap::new() }
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn remove(&mut self, idx: &T) {
        self.map.remove(idx);
    }
}
impl<T: Hash + Eq + Clone> Index<T> for Counter<T> {
    type Output = usize;
    fn index(&self, idx: T) -> &Self::Output {
        if self.map.contains_key(&idx) {
            let cntp = self.map[&idx].as_ptr();
            unsafe { &*cntp }
        } else {
            //map.insert(idx, Cell::new(0));
            //let mut cntp = map[&idx].as_ptr();
            //unsafe {& *cntp}
            &ZERO
        }
    }
}
impl<T: Hash + Eq + Clone> IndexMut<T> for Counter<T> {
    fn index_mut(&mut self, idx: T) -> &mut Self::Output {
        if self.map.contains_key(&idx) {
            let cntp = self.map[&idx].as_ptr();
            unsafe { &mut *cntp }
        } else {
            self.map.insert(idx.clone(), Cell::new(0));
            let cntp = self.map[&idx].as_ptr();
            unsafe { &mut *cntp }
        }
    }
}
