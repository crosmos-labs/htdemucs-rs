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

/// One tensor's header entry: name, dtype width (bytes), shape, data byte range.
struct Entry {
    name: String,
    width: usize,
    shape: Vec<usize>,
    a: usize,
    b: usize,
}

/// Parses and validates the JSON header against `data_len` bytes of tensor data. Entries are
/// returned in data order.
fn parse_header(header: &[u8], data_len: usize) -> Result<(Vec<Entry>, HashMap<String, String>), String> {
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(header).map_err(|e| format!("bad header: {e}"))?;
    let mut metadata = HashMap::new();
    let mut entries = vec![];
    for (name, h) in header {
        if name == "__metadata__" {
            for (k, v) in h.as_object().ok_or("bad __metadata__")? {
                metadata.insert(k.clone(), v.as_str().unwrap_or_default().to_string());
            }
            continue;
        }
        let bad = |what: &str| format!("{name}: bad {what}");
        let dtype = h.get("dtype").and_then(|v| v.as_str()).ok_or_else(|| bad("dtype"))?.to_string();
        let shape: Vec<usize> = h
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| bad("shape"))?
            .iter()
            .map(|v| v.as_u64().and_then(|x| usize::try_from(x).ok()))
            .collect::<Option<_>>()
            .ok_or_else(|| bad("shape"))?;
        let off = h.get("data_offsets").and_then(|v| v.as_array()).filter(|o| o.len() == 2).ok_or_else(|| bad("data_offsets"))?;
        let at = |i: usize| off[i].as_u64().and_then(|x| usize::try_from(x).ok()).ok_or_else(|| bad("data_offsets"));
        let (a, b) = (at(0)?, at(1)?);
        if a > b || b > data_len {
            return Err(format!("{name}: data {a}..{b} outside the file (truncated?)"));
        }
        let width = match dtype.as_str() {
            "F32" => 4,
            "F16" => 2,
            "F64" => 8,
            d => return Err(format!("{name}: unsupported dtype {d}")),
        };
        let count = shape.iter().try_fold(1usize, |p, &d| p.checked_mul(d)).ok_or_else(|| bad("shape"))?;
        if b - a != count.checked_mul(width).ok_or_else(|| bad("shape"))? {
            return Err(format!("{name}: {} bytes for shape {shape:?} {dtype}", b - a));
        }
        entries.push(Entry { name, width, shape, a, b });
    }
    entries.sort_by_key(|e| e.a);
    Ok((entries, metadata))
}

fn decode(raw: &[u8], width: usize) -> Vec<f32> {
    match width {
        4 => raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        2 => raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap()))).collect(),
        _ => raw.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
    }
}

impl SafeTensors {
    /// Reads a safetensors file tensor by tensor (peak memory: the tensors plus one tensor's
    /// raw bytes, not a copy of the whole file).
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        use std::io::{Read, Seek, SeekFrom};
        let path = path.as_ref();
        let ctx = |e: String| format!("{}: {e}", path.display());
        let mut f = std::fs::File::open(path).map_err(|e| ctx(e.to_string()))?;
        let size = f.metadata().map_err(|e| ctx(e.to_string()))?.len() as usize;
        let mut h8 = [0u8; 8];
        f.read_exact(&mut h8).map_err(|_| ctx("truncated: no header length".into()))?;
        let n = u64::from_le_bytes(h8);
        let end = usize::try_from(n).ok().and_then(|n| n.checked_add(8)).filter(|&e| e <= size).ok_or_else(|| ctx("truncated: header runs past the end".into()))?;
        let mut header = vec![0u8; end - 8];
        f.read_exact(&mut header).map_err(|e| ctx(e.to_string()))?;
        let (entries, metadata) = parse_header(&header, size - end).map_err(ctx)?;
        let mut tensors = HashMap::new();
        let mut raw = vec![];
        for e in &entries {
            raw.resize(e.b - e.a, 0);
            f.seek(SeekFrom::Start((end + e.a) as u64)).map_err(|e| ctx(e.to_string()))?;
            f.read_exact(&mut raw).map_err(|e| ctx(e.to_string()))?;
            tensors.insert(e.name.clone(), Tensor::new(&e.shape, decode(&raw, e.width)));
        }
        Ok(SafeTensors { tensors, metadata, order: entries.into_iter().map(|e| e.name).collect() })
    }

    /// Parses a safetensors file. Malformed input (truncated, bad header, offsets out of range,
    /// shape / size mismatch) is an error, never a panic.
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let n = bytes.get(..8).map(|h| u64::from_le_bytes(h.try_into().unwrap())).ok_or("truncated: no header length")?;
        let end = usize::try_from(n).ok().and_then(|n| n.checked_add(8)).filter(|&e| e <= bytes.len()).ok_or("truncated: header runs past the end")?;
        let data = &bytes[end..];
        let (entries, metadata) = parse_header(&bytes[8..end], data.len())?;
        let tensors = entries.iter().map(|e| (e.name.clone(), Tensor::new(&e.shape, decode(&data[e.a..e.b], e.width)))).collect();
        Ok(SafeTensors { tensors, metadata, order: entries.into_iter().map(|e| e.name).collect() })
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
    use super::{f16_to_f32, SafeTensors};

    #[test]
    fn malformed_files_are_errors() {
        // A valid two-tensor file, then every truncation of it and a few corruptions.
        let header = br#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F16","shape":[1,2],"data_offsets":[8,12]}}"#;
        let mut f = (header.len() as u64).to_le_bytes().to_vec();
        f.extend_from_slice(header);
        f.extend_from_slice(&1.0f32.to_le_bytes());
        f.extend_from_slice(&2.0f32.to_le_bytes());
        f.extend_from_slice(&[0x00, 0x3c, 0x00, 0xc0]);
        let ok = SafeTensors::parse(&f).unwrap();
        assert_eq!(ok.get("a").unwrap().data, [1.0, 2.0]);
        assert_eq!(ok.get("b").unwrap().data, [1.0, -2.0]);
        for n in 0..f.len() {
            assert!(SafeTensors::parse(&f[..n]).is_err(), "truncated to {n} bytes");
        }
        let mut huge = f.clone();
        huge[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(SafeTensors::parse(&huge).is_err());
        let bad = |from: &str, to: &str| {
            let h = String::from_utf8(header.to_vec()).unwrap().replace(from, to);
            let mut g = (h.len() as u64).to_le_bytes().to_vec();
            g.extend_from_slice(h.as_bytes());
            g.extend_from_slice(&f[8 + header.len()..]);
            SafeTensors::parse(&g)
        };
        assert!(bad("[0,8]", "[0,80]").is_err());
        assert!(bad("[0,8]", "[8,0]").is_err());
        assert!(bad("\"shape\":[2]", "\"shape\":[3]").is_err());
        assert!(bad("F16", "I8").is_err());
        assert!(bad("\"shape\":[2]", "\"shape\":\"x\"").is_err());
    }
    #[test]
    fn half() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x0200), 2f32.powi(-15));
    }
}

#[cfg(test)]
mod file_tests {
    use super::SafeTensors;

    #[test]
    fn load_reads_tensor_by_tensor_like_parse() {
        let header = br#"{"__metadata__":{"m":"x"},"b":{"dtype":"F16","shape":[1,2],"data_offsets":[8,12]},"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut f = (header.len() as u64).to_le_bytes().to_vec();
        f.extend_from_slice(header);
        f.extend_from_slice(&1.5f32.to_le_bytes());
        f.extend_from_slice(&(-2.0f32).to_le_bytes());
        f.extend_from_slice(&[0x00, 0x3c, 0x00, 0xc0]);
        let dir = std::env::temp_dir().join(format!("st-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.safetensors");
        std::fs::write(&p, &f).unwrap();
        let (a, b) = (SafeTensors::load(&p).unwrap(), SafeTensors::parse(&f).unwrap());
        assert_eq!(a.order, ["a", "b"]);
        assert_eq!(a.order, b.order);
        assert_eq!(a.metadata["m"], "x");
        for n in ["a", "b"] {
            assert_eq!(a.get(n).unwrap().data, b.get(n).unwrap().data);
            assert_eq!(a.get(n).unwrap().shape, b.get(n).unwrap().shape);
        }
        for n in [0, 7, 9, f.len() - 1] {
            std::fs::write(&p, &f[..n]).unwrap();
            assert!(SafeTensors::load(&p).is_err(), "truncated to {n}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
