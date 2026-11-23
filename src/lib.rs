//! htdemucs: Meta's HTDemucs (Demucs v4) music source separation, reimplemented in Rust on the
//! official pretrained weights. See README.md.

pub mod apply;
pub mod audio;
pub mod config;
pub mod ensemble;
pub mod model;
pub mod ops;
pub mod pyrandom;
pub mod roformer;
pub mod stft;
pub mod tensor;
pub mod transformer;
pub mod weights;

pub use model::HTDemucs;
pub use tensor::Tensor;

/// Optional sink for named intermediate activations (the parity test compares them with the
/// PyTorch dumps). Names and layouts follow the PyTorch module names / tensor shapes, minus the
/// batch dimension.
pub struct Tracer<'a>(pub Option<&'a mut dyn FnMut(&str, &Tensor)>);

impl<'a> Tracer<'a> {
    pub fn none() -> Self {
        Tracer(None)
    }

    pub fn on(&self) -> bool {
        self.0.is_some()
    }

    pub fn emit(&mut self, name: &str, t: &Tensor) {
        if let Some(f) = &mut self.0 {
            f(name, t);
        }
    }
}
