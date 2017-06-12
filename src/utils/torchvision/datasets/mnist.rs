#![allow(deprecated)]
use utils::data as D;
use std::ops::Index;
use std::path::PathBuf;
use std::{io, fs};
use curl::easy::Easy;
use flate2::{Flush, Decompress};
use std::io::Write;

type Sample = (Vec<u8>, u32);

static URLS: [&str; 4] = ["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                          "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                          "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                          "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"];




fn download(root: &String) -> io::Result<()> {
    let raw_path = PathBuf::from(root).join("raw");
    let processed_path = PathBuf::from(root).join("processed");
    // XXX how to ignore EEXIST?
    fs::create_dir(raw_path.clone())?;
    fs::create_dir(processed_path)?;

    for url in URLS.iter() {
        println!("downloading {}", url);
        let paths: Vec<_> = url.split("/").collect();
        let fname: Vec<_> = paths[5].split("/").collect();
        let file_path = raw_path.join(fname[0]);
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .open(file_path)?;
        let mut data = Vec::new();
        let mut handle = Easy::new();
        handle.url(url).unwrap();
        {
            let mut transfer = handle.transfer();
            transfer
                .write_function(|new_data| {
                                    data.extend_from_slice(new_data);
                                    Ok(new_data.len())
                                })
                .unwrap();
            transfer.perform().unwrap();
        }
        let mut output = Vec::with_capacity(data.len());
        // do we have a header?
        Decompress::new(false)
            .decompress_vec(data.as_slice(), &mut output, Flush::Finish)?;
        file.write_all(output.as_slice())?;
    }
    println!("Proceeding");
    Ok(())
}

pub struct MNIST {
    pub root: String,
    pub train: bool,
//    pub transform: Option<fn(Vec<u8>)>,
}

#[builder(pattern="owned")]
#[derive(Builder)]
pub struct MNISTArgs {
    root: String,
    #[builder(default="true")]
    train: bool,
    #[builder(default="false")]
    download: bool,
//    #[builder(default="None")]
//    transform: Option<Box<fn(&Vec<u8>)>>,
}

impl MNISTArgsBuilder {
    pub fn done(self) -> MNIST {
        let args = self.build().unwrap();
        MNIST::new(&args)
    }
}

impl MNIST {
    pub fn build(root: &str) -> MNISTArgsBuilder {
        MNISTArgsBuilder::default().root(root.into())
    }
    pub fn new(args: &MNISTArgs) -> Self {
        if args.download {
            download(&args.root);
        }
        MNIST {
            root: args.root.clone(),
            train: args.train,
//            transform: args.transform,
        }
    }
}

impl Index<usize> for MNIST {
    type Output = Sample;
    fn index(&self, idx: usize) -> &Self::Output {
        unimplemented!()
    }
}

impl D::DatasetIntf<Sample> for MNIST {
    fn len(&self) -> usize {
        if self.train { 60000 } else { 10000 }
    }
    fn iter(&self) -> Box<Iterator<Item = Sample>> {
        unimplemented!();
    }
}
