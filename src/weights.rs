//! Loading safetensors files (the converted weights, and the reference dumps in the parity test).
//! Format: u64 little-endian header size, JSON header {name: {dtype, shape, data_offsets}}, data.

use std::collections::HashMap;
use std::path::Path;

use crate::tensor::Tensor;

pub struct SafeTensors {
    pub tensors: HashMap<String, Tensor>,
    pub metadata: HashMap<String, String>,
    /// Names in file order.
    pub order: Vec<String>,
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // Subnormal half: renormalise.
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

impl SafeTensors {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&bytes[8..8 + n]).map_err(|e| e.to_string())?;
        let data = &bytes[8 + n..];
        let mut tensors = HashMap::new();
        let mut metadata = HashMap::new();
        let mut entries = vec![];
        for (name, h) in header {
            if name == "__metadata__" {
                for (k, v) in h.as_object().unwrap() {
                    metadata.insert(k.clone(), v.as_str().unwrap_or_default().to_string());
                }
                continue;
            }
            let dtype = h["dtype"].as_str().unwrap().to_string();
            let shape: Vec<usize> =
                h["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
            let off = h["data_offsets"].as_array().unwrap();
            let (a, b) = (off[0].as_u64().unwrap() as usize, off[1].as_u64().unwrap() as usize);
            let raw = &data[a..b];
            let values: Vec<f32> = match dtype.as_str() {
                "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
                "F16" => raw
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                    .collect(),
                "F64" => raw
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect(),
                d => return Err(format!("{name}: unsupported dtype {d}")),
            };
            entries.push((a, name.clone()));
            tensors.insert(name, Tensor::new(&shape, values));
        }
        entries.sort();
        Ok(SafeTensors { tensors, metadata, order: entries.into_iter().map(|e| e.1).collect() })
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Removes and returns a tensor; panics with the name if it is missing.
    pub fn take(&mut self, name: &str) -> Tensor {
        self.tensors.remove(name).unwrap_or_else(|| panic!("missing weight {name}"))
    }
}

#[cfg(test)]
mod tests {
    use super::f16_to_f32;
    #[test]
    fn half() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x0200), 2f32.powi(-15));
    }
}
