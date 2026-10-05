//! Memory-mapped `model.safetensors` access shared by every backend.

use crate::{Error, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use std::path::Path;

pub struct Weights {
    mmap: Mmap,
    path: std::path::PathBuf,
}

impl std::fmt::Debug for Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Weights({})", self.path.display())
    }
}

impl Weights {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("model.safetensors");
        if !path.exists() {
            return Err(Error::Weights(format!(
                "'model.safetensors' not found in {}",
                model_dir.display()
            )));
        }
        let file = std::fs::File::open(&path)?;
        // SAFETY: the checkpoint file is treated as read-only for the life of the mapping.
        let mmap = unsafe { Mmap::map(&file)? };
        // Validate the header eagerly. It also checks that the file is as long as the header
        // says, so a file cut short (an interrupted download) fails here, with its path.
        if let Err(e) = SafeTensors::deserialize(&mmap) {
            return Err(Error::Weights(format!(
                "{} ({} bytes) is truncated or not a safetensors file: {e}",
                path.display(),
                mmap.len()
            )));
        }
        Ok(Self { mmap, path })
    }

    /// Parse the safetensors header. Cheap; call per use rather than caching a self-borrow.
    pub fn view(&self) -> Result<SafeTensors<'_>> {
        Ok(SafeTensors::deserialize(&self.mmap)?)
    }

    pub fn names(&self) -> Result<Vec<String>> {
        Ok(self
            .view()?
            .names()
            .into_iter()
            .map(|s| s.to_string())
            .collect())
    }

    pub fn tensor_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let st = self.view()?;
        let t = st
            .tensor(name)
            .map_err(|e| Error::Weights(format!("{name}: {e}")))?;
        Ok((t.shape().to_vec(), to_f32(&t)?))
    }

    /// Verify the checkpoint carries the parameter families the decision model needs.
    pub fn verify(&self) -> Result<()> {
        let names = self.names()?;
        for prefix in ["encoder.", "type_emb.", "scorer.", "act_head."] {
            if !names.iter().any(|n| n.starts_with(prefix)) {
                return Err(Error::Weights(format!(
                    "checkpoint is missing '{prefix}' parameters; expected an RL Agent decision model"
                )));
            }
        }
        Ok(())
    }
}

/// Convert any float tensor view to f32.
pub fn to_f32(t: &TensorView<'_>) -> Result<Vec<f32>> {
    let data = t.data();
    Ok(match t.dtype() {
        Dtype::F32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::F16 => data
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::BF16 => data
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::F64 => data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        other => return Err(Error::Weights(format!("unsupported dtype {other:?}"))),
    })
}

/// Convert any float tensor view to f16.
pub fn to_f16(t: &TensorView<'_>) -> Result<Vec<f16>> {
    let data = t.data();
    Ok(match t.dtype() {
        Dtype::F16 => data
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]))
            .collect(),
        _ => to_f32(t)?.into_iter().map(f16::from_f32).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A fresh directory under the system temp dir, removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            static N: AtomicUsize = AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!(
                "laya-core-weights-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A missing file and a file cut anywhere (empty, inside the length prefix, inside the
    /// header, inside the data, one byte short) are each an error naming the file, never a
    /// panic; the whole file opens.
    #[test]
    fn open_rejects_a_missing_or_truncated_file() {
        let data = vec![0u8; 4096];
        let tensors = [
            (
                "encoder.w",
                TensorView::new(Dtype::F16, vec![32, 32], &data[..2048]).unwrap(),
            ),
            (
                "act_head.0.weight",
                TensorView::new(Dtype::F32, vec![16, 32], &data[2048..]).unwrap(),
            ),
        ];
        let bytes = safetensors::serialize(tensors, None).unwrap();
        let header_end = 8 + u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let dir = TempDir::new();
        let e = Weights::open(&dir.0).unwrap_err().to_string();
        assert!(e.contains("'model.safetensors' not found in"), "{e}");
        let path = dir.0.join("model.safetensors");
        for cut in [
            0,
            4,
            8,
            header_end / 2,
            header_end,
            header_end + 100,
            bytes.len() - 1,
        ] {
            std::fs::write(&path, &bytes[..cut]).unwrap();
            let e = Weights::open(&dir.0).unwrap_err().to_string();
            assert!(
                e.contains(&format!(
                    "model.safetensors ({cut} bytes) is truncated or not a safetensors file"
                )),
                "cut at {cut}: {e}"
            );
        }
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(Weights::open(&dir.0).unwrap().names().unwrap().len(), 2);
    }
}
