//! The model hyperparameters written by tools/convert.py (HTDemucs' constructor kwargs).

use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct Fraction {
    pub num: u64,
    pub den: u64,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    pub name: String,
    pub sources: Vec<String>,
    pub audio_channels: usize,
    pub samplerate: usize,
    pub segment: Fraction,
    pub training_length: usize,
    pub channels: usize,
    pub channels_time: Option<usize>,
    pub growth: usize,
    pub nfft: usize,
    pub wiener_iters: i64,
    pub cac: bool,
    pub depth: usize,
    pub rewrite: bool,
    pub multi_freqs: Vec<f64>,
    pub freq_emb: f64,
    pub emb_scale: f64,
    pub kernel_size: usize,
    pub stride: usize,
    pub time_stride: usize,
    pub context: usize,
    pub context_enc: usize,
    pub norm_starts: usize,
    pub dconv_mode: usize,
    pub dconv_depth: usize,
    pub dconv_comp: usize,
    pub bottom_channels: usize,
    pub t_layers: usize,
    pub t_hidden_scale: f64,
    pub t_heads: usize,
    pub t_layer_scale: bool,
    pub t_gelu: bool,
    pub t_emb: String,
    pub t_max_period: f64,
    pub t_weight_pos_embed: f64,
    pub t_sin_random_shift: usize,
    pub t_norm_in: bool,
    pub t_norm_in_group: bool,
    pub t_group_norm: bool,
    pub t_norm_first: bool,
    pub t_norm_out: bool,
    pub t_sparse_self_attn: bool,
    pub t_sparse_cross_attn: bool,
    pub t_cross_first: bool,
}

impl Config {
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let cfg: Config = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        cfg.check()?;
        Ok(cfg)
    }

    /// We implement the configuration the pretrained htdemucs / htdemucs_6s use; refuse others
    /// rather than silently computing something different.
    fn check(&self) -> Result<(), String> {
        let mut bad = vec![];
        let mut need = |ok: bool, what: &str| {
            if !ok {
                bad.push(what.to_string())
            }
        };
        need(self.cac, "cac");
        need(self.wiener_iters == 0, "wiener_iters == 0");
        need(self.multi_freqs.is_empty(), "no multi_freqs");
        need(self.norm_starts >= self.depth, "no GroupNorm in encoder/decoder layers");
        need(self.dconv_mode == 3, "dconv_mode == 3");
        need(self.rewrite, "rewrite");
        need(self.context_enc == 0 && self.context == 1, "context_enc 0, context 1");
        need(self.channels_time.is_none(), "channels_time unset");
        need(self.t_emb == "sin" && self.t_sin_random_shift == 0, "sin time embedding, no shift");
        need(self.t_norm_in && !self.t_norm_in_group && !self.t_group_norm, "LayerNorm transformer");
        need(self.t_norm_first && self.t_norm_out && self.t_layer_scale && self.t_gelu, "norm_first / norm_out / layer_scale / gelu");
        need(!self.t_sparse_self_attn && !self.t_sparse_cross_attn, "dense attention");
        need(!self.t_cross_first, "t_cross_first false");
        // Every encoder layer is a frequency layer (no branch merge before the transformer).
        let mut freqs = self.nfft / 2;
        for _ in 0..self.depth {
            need(freqs > self.kernel_size, "all layers in the frequency domain");
            freqs /= self.stride;
        }
        need(self.training_length as u64 == self.segment.num * self.samplerate as u64 / self.segment.den, "training length");
        if bad.is_empty() { Ok(()) } else { Err(format!("unsupported model config: {}", bad.join(", "))) }
    }
}
