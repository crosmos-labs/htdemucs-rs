//! A minimal dense f32 tensor: row-major, contiguous. Everything in this crate is batch size 1,
//! so shapes omit PyTorch's leading batch dimension.

#[derive(Clone, Debug, Default)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: &[usize], data: Vec<f32>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len(), "shape {shape:?}");
        Tensor { shape: shape.to_vec(), data }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Tensor { shape: shape.to_vec(), data: vec![0.0; shape.iter().product()] }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn dim(&self, i: usize) -> usize {
        self.shape[i]
    }

    /// Same data, new shape (like `view`).
    pub fn reshape(mut self, shape: &[usize]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), self.data.len(), "reshape to {shape:?}");
        self.shape = shape.to_vec();
        self
    }

    /// (A, B, C) -> (B, A, C).
    pub fn swap01(&self) -> Tensor {
        let (a, b, c) = (self.shape[0], self.shape[1], self.shape[2]);
        let mut out = vec![0.0; self.data.len()];
        for i in 0..a {
            for j in 0..b {
                let src = &self.data[(i * b + j) * c..(i * b + j + 1) * c];
                out[(j * a + i) * c..(j * a + i + 1) * c].copy_from_slice(src);
            }
        }
        Tensor::new(&[b, a, c], out)
    }

    /// 2-D transpose: (A, B) -> (B, A).
    pub fn t(&self) -> Tensor {
        let (a, b) = (self.shape[0], self.shape[1]);
        let mut out = vec![0.0; self.data.len()];
        const BS: usize = 32;
        for i0 in (0..a).step_by(BS) {
            for j0 in (0..b).step_by(BS) {
                for i in i0..(i0 + BS).min(a) {
                    for j in j0..(j0 + BS).min(b) {
                        out[j * a + i] = self.data[i * b + j];
                    }
                }
            }
        }
        Tensor::new(&[b, a], out)
    }
}
