use std::fs::File;

use rmps::{Deserializer, Serializer};
use serde::{Deserialize, Serialize};

use std::path::Path;
use std::io;
use std::io::{Write, Read, Error, ErrorKind};

use tensor::TensorKind;

pub fn save<P, T: Serialize>(path: P, arg: &Vec<T>) -> io::Result<usize>
    where P: AsRef<Path>
{
    let mut buffer = File::create(path)?;
    let mut encoded = Vec::new();
    arg.serialize(&mut Serializer::new(&mut encoded)).unwrap();
    buffer.write(encoded.as_slice())
}

pub fn load<'a, P, T: 'a + Deserialize<'a>>(path: P) -> io::Result<Vec<T>>
    where P: AsRef<Path>
{
    let mut buffer = File::open(path)?;
    let mut encoded = Vec::new();
    let size = buffer.read_to_end(&mut encoded)?;
    if size == 0 {
        return Err(Error::new(ErrorKind::UnexpectedEof, "Empty File"));
    };
    let mut de = Deserializer::new(&encoded[..]);
    let decoded = Deserialize::deserialize(&mut de).expect("decode failed");
    Ok(decoded)
}
