#![allow(deprecated)]
use utils::data as D;
use std::ops::Index;
use std::path::PathBuf;
use std::{io, fs};
use curl::easy::Easy;
use flate2::{Flush, Decompress};
use std::io::Write;
use utils::data::{Dataset, DatasetIntf};
use std::rc::Rc;
use tensor::{Tensor, TensorKind};
use torch;

type Sample = (Tensor<f32>, Tensor<i64>);

static URLS: [&str; 4] = ["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                          "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                          "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                          "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"];
static RAW_FOLDER: &str = "raw";
static PROCESSED_FOLDER: &str = "processed";
static TRAINING_FILE: &str = "training.pt";
static TEST_FILE: &str = "test.pt";

fn download(root: &String) -> io::Result<()> {
    let raw_path = PathBuf::from(root).join(RAW_FOLDER);
    let processed_path = PathBuf::from(root).join(PROCESSED_FOLDER);
    // XXX how to ignore EEXIST?
    fs::create_dir(raw_path.clone())?;
    fs::create_dir(processed_path.clone())?;

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

    let training_set = vec![read_image_file(raw_path.join("train-images-idx3-ubyte")),
                            read_label_file(raw_path.join("train-labels-idx1-ubyte"))];
    let test_set = vec![read_image_file(raw_path.join("t10k-images-idx3-ubyte")),
                        read_label_file(raw_path.join("t10k-labels-idx1-ubyte"))];

    torch::save(processed_path.join(TRAINING_FILE), &training_set)?;
    torch::save(processed_path.join(TEST_FILE), &test_set)?;

    Ok(())
}

fn read_label_file(path: PathBuf) -> TensorKind {
    unimplemented!()
}
fn read_image_file(path: PathBuf) -> TensorKind {
    unimplemented!()
}


pub struct MNIST {
    pub root: String,
    pub train: bool,
    pub data: Tensor<u8>,
    pub labels: Tensor<i64>,
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
    pub fn done(self) -> Dataset<Sample> {
        let args = self.build().unwrap();
        Dataset::new(Rc::new(MNIST::new(&args)))
    }
}

impl MNIST {
    pub fn build(root: &str) -> MNISTArgsBuilder {
        MNISTArgsBuilder::default().root(root.into())
    }
    pub fn new(args: &MNISTArgs) -> Self {
        if args.download {
            download(&args.root).expect("download failed");
        }
        let processed_path = PathBuf::from(&args.root).join(PROCESSED_FOLDER);
        let mut v = if args.train {
            torch::load(processed_path.join(TRAINING_FILE)).expect("torch load failed")
        } else {
            torch::load(processed_path.join(TEST_FILE)).expect("torch load failed")
        };
        let (data, labels) = (v.remove(0), v.remove(0));

        MNIST {
            root: args.root.clone(),
            train: args.train,
            data: data.into(),
            labels: labels.into(),
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
