//! The contract between the runtime and a model implementation.
//!
//! Changed in sys1rust from laya-r-mlx 914c9a7: `BackendOptions::tuning`, and the one setting
//! of it that the runtime reads itself, `parallel_load` ([`BackendOptions::parallel_load`]).
//!
//! A backend owns the encoder, the decision-head transformer layers, the type embedding and
//! the scorer. It receives a collated batch and returns the masked scorer logits plus the
//! pooled `[CLS]` hidden state of the head output; the action head runs on the CPU in
//! [`crate::decode`].

use crate::{Error, Result};

/// Which compute device a backend should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Auto,
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Default)]
pub struct BackendOptions {
    pub device: Device,
    /// Run the transformer in f32 instead of the checkpoint's f16 (slower, closer to the
    /// Python CPU/MPS numerics).
    pub f32: bool,
    /// Backend-specific settings as a comma list of `key=value` (laya-mlx: see its `Knobs`).
    /// `None` falls back to the `SYS1_MLX` environment variable. One setting is the runtime's,
    /// not the backend's: `parallel_load` (see [`Self::parallel_load`]).
    pub tuning: Option<String>,
}

impl BackendOptions {
    /// The settings in force: `tuning`, or `SYS1_MLX` when `tuning` is `None` (empty when
    /// that is unset too).
    pub fn settings(&self) -> String {
        match &self.tuning {
            Some(t) => t.clone(),
            None => std::env::var("SYS1_MLX").unwrap_or_default(),
        }
    }

    /// The `parallel_load` setting, which [`crate::Agent::load`] reads: load the tokenizer on
    /// a second thread while the calling thread opens the weights and builds the backend. On
    /// when bare or `=1`, off when `=0` or absent, and the last mention wins. Any other value is
    /// an [`Error::Config`], as a bad value of a backend setting is. A backend that checks its
    /// settings strictly has to accept the key (laya-mlx does).
    pub fn parallel_load(&self) -> Result<bool> {
        let mut on = false;
        for kv in self.settings().split(',') {
            match kv.split_once('=') {
                None if kv == "parallel_load" => on = true,
                Some(("parallel_load", v)) => {
                    on = match v {
                        "1" => true,
                        "0" => false,
                        _ => {
                            return Err(Error::Config(format!(
                                "settings: `{kv}`: `{v}` is not 0 or 1"
                            )))
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(on)
    }
}

/// A padded batch, row-major. Row `i` covers `input_ids[i*len .. (i+1)*len]`.
#[derive(Debug, Clone)]
pub struct Batch {
    pub n: usize,
    pub len: usize,
    pub kmax: usize,
    pub input_ids: Vec<u32>,
    /// 1 for real tokens, 0 for padding (`n * len`).
    pub attention_mask: Vec<u32>,
    /// Unpadded length of each row.
    pub seq_lens: Vec<usize>,
    /// Marker token positions (`n * kmax`, zero-filled past `marker_count`).
    pub marker_pos: Vec<u32>,
    /// Valid markers per row.
    pub marker_count: Vec<usize>,
    /// Question type index per row (0 choice, 1 score, 2 noul).
    pub qtype: Vec<u32>,
}

impl Batch {
    pub fn marker_mask(&self, row: usize, k: usize) -> bool {
        k < self.marker_count[row]
    }
    pub fn total_tokens(&self) -> usize {
        self.seq_lens.iter().sum()
    }
}

/// Backend output for one batch.
#[derive(Debug, Clone)]
pub struct BackendOutput {
    /// `n * kmax` scorer logits, `-1e4` where the marker mask is false.
    pub logits: Vec<f32>,
    /// `n * hidden` head-output hidden state at position 0, in f32.
    pub pooled: Vec<f32>,
}

pub trait Backend: Send + Sync {
    /// Human-readable backend/device description, e.g. `mlx(gpu,f16)`.
    fn name(&self) -> String;
    fn forward(&self, batch: &Batch) -> Result<BackendOutput>;
    /// The settings whose custom kernel is active: requested, built and checked at load. A
    /// setting that fell back to the library ops is not listed, so a test of a kernel can
    /// tell a run through it from one through the fallback.
    fn active_kernels(&self) -> &'static [&'static str] {
        &[]
    }
    /// Sequence length to pad a batch of `rows` rows whose longest row is `len` tokens to.
    /// Backends that keep a fixed set of shapes round `len` up to a bucket.
    fn padded_len(&self, len: usize, _rows: usize) -> usize {
        len
    }
    /// Hint that the backend may want to warm up compiled kernels for a shape.
    fn warmup(&self) -> Result<()> {
        Ok(())
    }
    /// Debug hook: the encoder's `last_hidden_state` for a batch (`n * len * hidden`, f32),
    /// used by the parity harness against `encoder_hidden_item0` fixtures.
    fn encoder_hidden(&self, _batch: &Batch) -> Result<Option<Vec<f32>>> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(tuning: &str) -> BackendOptions {
        BackendOptions {
            tuning: Some(tuning.into()),
            ..Default::default()
        }
    }

    /// `parallel_load` parses like a backend flag: bare or `=1` is on, `=0` is off, the last
    /// mention wins, another value is an error, and other keys (`parallel_loader`) do not count.
    #[test]
    fn parallel_load_parses_as_a_flag() {
        for (spec, want) in [
            ("", false),
            ("parallel_load", true),
            ("parallel_load=1", true),
            ("parallel_load=0", false),
            ("f16gelu,parallel_load,cache=512", true),
            ("parallel_load,parallel_load=0", false),
            ("parallel_load=0,parallel_load", true),
            ("parallel_loader,xparallel_load=1", false),
        ] {
            assert_eq!(opts(spec).parallel_load().unwrap(), want, "{spec}");
        }
        for spec in [
            "parallel_load=2",
            "parallel_load=",
            "f16gelu,parallel_load=yes",
        ] {
            let e = opts(spec).parallel_load().unwrap_err().to_string();
            assert!(
                e.contains("parallel_load") && e.contains("is not 0 or 1"),
                "{spec}: {e}"
            );
        }
    }
}
