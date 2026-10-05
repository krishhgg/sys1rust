//! Shared by the `sys1-bench` and `sys1-probe` binaries: the checkpoint directory of a model
//! pinned in `bench/models.lock.json`, inside the local Hugging Face hub cache.

use anyhow::{bail, Context, Result};
use laya_core::resolve::hf_cache_dir;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// The pinned snapshot of `model` and its sha: `<hub cache>/models--org--name/snapshots/<sha>`
/// for the `repo` and `sha` of `<bench>/models.lock.json`. The hub cache is the directory
/// huggingface_hub downloads into, resolved as it does ([`hf_cache_dir`]: `HF_HUB_CACHE`, else
/// `HF_HOME/hub`, else `XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub`), the
/// same lookup sys1rust uses. Never downloads: a snapshot that is not there is an error naming the
/// directory and the cache it looked in.
pub fn pinned_model_dir(bench: &Path, model: &str) -> Result<(PathBuf, String)> {
    pinned_model_dir_in(&hf_cache_dir(), bench, model)
}

/// [`pinned_model_dir`] with the hub cache given.
fn pinned_model_dir_in(cache: &Path, bench: &Path, model: &str) -> Result<(PathBuf, String)> {
    let lock_path = bench.join("models.lock.json");
    let lock = std::fs::read_to_string(&lock_path).with_context(|| format!("read {}", lock_path.display()))?;
    let lock: Value = serde_json::from_str(&lock).with_context(|| format!("parse {}", lock_path.display()))?;
    let pin = lock
        .get(model)
        .with_context(|| format!("{model} is not in {}", lock_path.display()))?;
    let repo = pin["repo"].as_str().with_context(|| format!("{model}: repo"))?;
    let sha = pin["sha"].as_str().with_context(|| format!("{model}: sha"))?;
    let dir = snapshot_dir(cache, repo, sha);
    if !dir.is_dir() {
        bail!(
            "{model} ({repo} at {sha}) is not downloaded: {} does not exist (hub cache {}, from \
             HF_HUB_CACHE, HF_HOME/hub, XDG_CACHE_HOME/huggingface/hub or ~/.cache/huggingface/hub)",
            dir.display(),
            cache.display()
        );
    }
    Ok((dir, sha.to_string()))
}

/// `<cache>/models--org--name/snapshots/<sha>`, the layout huggingface_hub writes.
fn snapshot_dir(cache: &Path, repo: &str, sha: &str) -> PathBuf {
    cache
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots")
        .join(sha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A fresh bench root with a lock file and an empty hub cache under the system temp dir,
    /// removed on drop.
    struct Fake {
        root: PathBuf,
    }

    impl Fake {
        fn new() -> Self {
            static N: AtomicUsize = AtomicUsize::new(0);
            let root = std::env::temp_dir().join(format!(
                "sys1-bench-pin-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(root.join("bench")).unwrap();
            std::fs::create_dir_all(root.join("hub")).unwrap();
            std::fs::write(
                root.join("bench/models.lock.json"),
                r#"{"_note": "pins", "typed-decisions": {"repo": "convaiinnovations/laya-typed-decisions", "sha": "abc123", "license": "apache-2.0"}, "broken": {"repo": "x/y"}}"#,
            )
            .unwrap();
            Self { root }
        }
        fn bench(&self) -> PathBuf {
            self.root.join("bench")
        }
        fn hub(&self) -> PathBuf {
            self.root.join("hub")
        }
        fn resolve(&self, model: &str) -> Result<(PathBuf, String)> {
            pinned_model_dir_in(&self.hub(), &self.bench(), model)
        }
    }

    impl Drop for Fake {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn snapshot_dir_is_the_hub_layout() {
        assert_eq!(
            snapshot_dir(Path::new("/c"), "convaiinnovations/laya", "55cf4c4"),
            PathBuf::from("/c/models--convaiinnovations--laya/snapshots/55cf4c4")
        );
    }

    #[test]
    fn pinned_snapshot_is_found_under_the_given_cache() {
        let f = Fake::new();
        let snap = snapshot_dir(&f.hub(), "convaiinnovations/laya-typed-decisions", "abc123");
        std::fs::create_dir_all(&snap).unwrap();
        assert_eq!(f.resolve("typed-decisions").unwrap(), (snap, "abc123".to_string()));
    }

    /// A pin that is not downloaded names the missing directory and the cache that was searched,
    /// so a run that looked in the wrong cache says which one it used.
    #[test]
    fn missing_snapshot_names_the_directory_and_the_cache() {
        let f = Fake::new();
        let e = f.resolve("typed-decisions").unwrap_err().to_string();
        assert!(e.contains("is not downloaded"), "{e}");
        assert!(e.contains(&snapshot_dir(&f.hub(), "convaiinnovations/laya-typed-decisions", "abc123").display().to_string()), "{e}");
        assert!(e.contains(&format!("hub cache {}", f.hub().display())), "{e}");
        assert!(e.contains("HF_HUB_CACHE"), "{e}");
    }

    #[test]
    fn unknown_model_and_incomplete_pin_are_errors() {
        let f = Fake::new();
        let e = f.resolve("english").unwrap_err().to_string();
        assert!(e.contains("english is not in"), "{e}");
        let e = f.resolve("broken").unwrap_err().to_string();
        assert!(e.contains("broken: sha"), "{e}");
        let e = pinned_model_dir_in(&f.hub(), &f.root.join("nowhere"), "typed-decisions")
            .unwrap_err()
            .to_string();
        assert!(e.contains("models.lock.json"), "{e}");
    }
}
