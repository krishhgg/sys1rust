//! HF `tokenizers` wrapper exposing exactly what the sequence builder needs.

use crate::{Error, Result};
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

pub struct LayaTokenizer {
    tok: Tokenizer,
    pub cls_id: u32,
    pub sep_id: u32,
    pub mask_id: u32,
    pub pad_id: u32,
    pub mask_token: String,
}

impl std::fmt::Debug for LayaTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayaTokenizer")
            .field("cls_id", &self.cls_id)
            .field("sep_id", &self.sep_id)
            .field("mask_id", &self.mask_id)
            .field("pad_id", &self.pad_id)
            .field("mask_token", &self.mask_token)
            .finish()
    }
}

fn special_str(cfg: &Value, key: &str) -> Option<String> {
    match cfg.get(key)? {
        Value::String(s) => Some(s.clone()),
        Value::Object(o) => o.get("content").and_then(Value::as_str).map(str::to_owned),
        _ => None,
    }
}

impl LayaTokenizer {
    /// The tokenizer file [`Self::load`] reads, `<dir>/tokenizer/tokenizer.json`.
    pub fn file(model_dir: &Path) -> PathBuf {
        model_dir.join("tokenizer").join("tokenizer.json")
    }

    /// Load `<dir>/tokenizer/tokenizer.json` (+ `tokenizer_config.json` for the special tokens).
    pub fn load(model_dir: &Path) -> Result<Self> {
        let tdir = model_dir.join("tokenizer");
        let file = Self::file(model_dir);
        let tok = Tokenizer::from_file(&file)
            .map_err(|e| Error::Tokenizer(format!("{}: {e}", file.display())))?;
        let cfg: Value = match std::fs::read(tdir.join("tokenizer_config.json")) {
            Ok(b) => serde_json::from_slice(&b)?,
            Err(_) => Value::Null,
        };
        Self::from_tokenizer(tok, &cfg)
    }

    /// Wrap a loaded tokenizer. `cfg` is the `tokenizer_config.json` object naming the special
    /// tokens, or `Null` to find them by their usual names.
    pub fn from_tokenizer(tok: Tokenizer, cfg: &Value) -> Result<Self> {
        let pick = |key: &str, fallbacks: &[&str]| -> Result<(u32, String)> {
            let mut names: Vec<String> = special_str(cfg, key).into_iter().collect();
            names.extend(fallbacks.iter().map(|s| s.to_string()));
            for n in &names {
                if let Some(id) = tok.token_to_id(n) {
                    return Ok((id, n.clone()));
                }
            }
            Err(Error::Tokenizer(format!(
                "special token {key} not found (tried {names:?})"
            )))
        };
        let (cls_id, _) = pick("cls_token", &["[CLS]", "<bos>", "<s>"])?;
        let (sep_id, _) = pick("sep_token", &["[SEP]", "<eos>", "</s>"])?;
        let (mask_id, mask_token) = pick("mask_token", &["[MASK]", "<mask>"])?;
        let (pad_id, _) = pick("pad_token", &["[PAD]", "<pad>"])?;
        Ok(Self {
            tok,
            cls_id,
            sep_id,
            mask_id,
            pad_id,
            mask_token,
        })
    }

    /// `tok(text, add_special_tokens=False)["input_ids"]`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tok
            .encode(text, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn inner(&self) -> &Tokenizer {
        &self.tok
    }
}
