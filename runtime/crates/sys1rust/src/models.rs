//! The three Laya models sys1rust can download, each at the revision pinned in
//! `bench/models.lock.json`, with the five files the loader reads. A file's `blob` is the name
//! huggingface_hub gives it under `blobs/`: its sha256 for a Git LFS file, its git blob sha1
//! otherwise. `packaging/gen_manifest.py` prints the table from the pinned snapshots and checks
//! every blob name against the file's bytes.

use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct ModelFile {
    /// Path inside the snapshot, `/`-separated.
    pub path: &'static str,
    pub size: u64,
    /// sha256 (64 hex digits) or git blob sha1 (40).
    pub blob: &'static str,
}

#[derive(Debug)]
pub struct LayaModel {
    pub name: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    pub files: &'static [ModelFile],
}

const fn f(path: &'static str, size: u64, blob: &'static str) -> ModelFile {
    ModelFile { path, size, blob }
}

#[rustfmt::skip]
pub static MODELS: [LayaModel; 3] = [
    LayaModel {
        name: "typed-decisions",
        repo: "convaiinnovations/laya-typed-decisions",
        revision: "1a793eb568e6718f15941d08f85432581df534e3",
        files: &[
            f("model.safetensors", 842609220, "4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e"),
            f("rl_agent_config.json", 847, "5f0e1d5f2366fe8ba2ff330dffaeed53b469e97e"),
            f("encoder/config.json", 2084, "d4be4829750fb04c0aa8b9897c3ea827f76c0109"),
            f("tokenizer/tokenizer.json", 3583228, "2f4d8583e507b7466d2490e2d6c045647a822698"),
            f("tokenizer/tokenizer_config.json", 337, "ed1ffabc2ce11120754705709569e365e46da71a"),
        ],
    },
    LayaModel {
        name: "multilingual",
        repo: "convaiinnovations/laya-multilingual",
        revision: "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
        files: &[
            f("model.safetensors", 643835514, "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204"),
            f("rl_agent_config.json", 472, "00e35f88bb731bb9126a914666ab1cdac8a204c8"),
            f("encoder/config.json", 1938, "0de0e2d30638873790cf962def52e2acf4db3eef"),
            f("tokenizer/tokenizer.json", 34363188, "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f"),
            f("tokenizer/tokenizer_config.json", 502, "eea1ed61121530d74c1722dc2aab6ec917a75909"),
        ],
    },
    LayaModel {
        name: "english",
        repo: "convaiinnovations/laya",
        revision: "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
        files: &[
            f("model.safetensors", 842609210, "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"),
            f("rl_agent_config.json", 745, "3e4fcbf12cf36164ce18a1398aa9f35f58375ae0"),
            f("encoder/config.json", 2083, "5881ba831f2db5ce0f606bbaa1f2668e1e6cb706"),
            f("tokenizer/tokenizer.json", 3583228, "2f4d8583e507b7466d2490e2d6c045647a822698"),
            f("tokenizer/tokenizer_config.json", 308, "9fd800115c5c92353220aa66addfce67a9135f32"),
        ],
    },
];

/// What the cache holds of a model's pinned snapshot.
#[derive(Debug, PartialEq, Eq)]
pub enum Status {
    /// Every file is in the snapshot at its expected size.
    Complete,
    /// `have` bytes are there, counting snapshot files, blobs not yet linked and partial
    /// downloads.
    Partial {
        have: u64,
    },
    Missing,
}

impl LayaModel {
    pub fn total_size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }

    /// `<cache>/models--org--name`.
    pub fn repo_dir(&self, cache: &Path) -> PathBuf {
        cache.join(repo_folder(self.repo))
    }

    pub fn snapshot_dir(&self, cache: &Path) -> PathBuf {
        self.repo_dir(cache).join("snapshots").join(self.revision)
    }

    pub fn blob_path(&self, cache: &Path, file: &ModelFile) -> PathBuf {
        self.repo_dir(cache).join("blobs").join(file.blob)
    }

    /// Checks sizes only, so a call costs a few `stat`s even for an 846 MB snapshot.
    pub fn status(&self, cache: &Path) -> Status {
        let snapshot = self.snapshot_dir(cache);
        let mut complete = true;
        let mut have = 0;
        for file in self.files {
            if file_size(&snapshot.join(file.path)) == Some(file.size) {
                have += file.size;
                continue;
            }
            complete = false;
            let blob = self.blob_path(cache, file);
            if file_size(&blob) == Some(file.size) {
                have += file.size;
            } else if let Some(n) = file_size(&incomplete_path(&blob)) {
                have += n.min(file.size);
            }
        }
        match (complete, have) {
            (true, _) => Status::Complete,
            (false, 0) => Status::Missing,
            (false, have) => Status::Partial { have },
        }
    }
}

/// A Laya model by name or repo id, as `--model` takes them.
pub fn find(name_or_repo: &str) -> Option<&'static LayaModel> {
    let name = crate::config::cli_checkpoint_name(name_or_repo)?;
    MODELS.iter().find(|m| m.name == name)
}

/// `models--org--name`, huggingface_hub's folder for a model repo.
pub fn repo_folder(repo: &str) -> String {
    format!("models--{}", repo.replace('/', "--"))
}

/// `<blob>.incomplete`, where huggingface_hub and sys1rust write a blob while downloading it.
pub fn incomplete_path(blob: &Path) -> PathBuf {
    let mut name = blob
        .file_name()
        .expect("a blob path ends in a file name")
        .to_os_string();
    name.push(".incomplete");
    blob.with_file_name(name)
}

/// Size of the regular file at `path`, following symlinks.
pub fn file_size(path: &Path) -> Option<u64> {
    std::fs::metadata(path)
        .ok()
        .filter(|m| m.is_file())
        .map(|m| m.len())
}

/// Megabytes as macOS and Homebrew count them (10^6 bytes), rounded.
pub fn mb(bytes: u64) -> u64 {
    (bytes + 500_000) / 1_000_000
}

/// The `sys1rust models` listing.
pub fn table(cache: &Path) -> String {
    let mut out = format!(
        "{:<16} {:<39} {:<8} {}\n",
        "MODEL", "REPO", "REVISION", "STATUS"
    );
    for m in &MODELS {
        let total = mb(m.total_size());
        let status = match m.status(cache) {
            Status::Complete => format!("downloaded, {total} MB"),
            Status::Partial { have } => format!("partial, {} of {total} MB", mb(have)),
            Status::Missing => format!("not downloaded, {total} MB"),
        };
        out.push_str(&format!(
            "{:<16} {:<39} {:<8} {status}\n",
            m.name,
            m.repo,
            &m.revision[..7]
        ));
    }
    out.push_str(&format!("cache: {}\n", cache.display()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::fs;

    /// The files laya-core reads from a checkpoint directory.
    const LOADER_FILES: [&str; 5] = [
        "model.safetensors",
        "rl_agent_config.json",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ];

    #[test]
    fn manifest_matches_models_lock() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../bench/models.lock.json"
        );
        let lock: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        for m in &MODELS {
            assert_eq!(lock[m.name]["repo"], m.repo, "{}", m.name);
            assert_eq!(lock[m.name]["sha"], m.revision, "{}", m.name);
        }
    }

    #[test]
    fn manifest_lists_the_served_checkpoints_in_order() {
        let pairs: Vec<_> = MODELS.iter().map(|m| (m.name, m.repo)).collect();
        assert_eq!(pairs, crate::config::CHECKPOINTS.to_vec());
    }

    #[test]
    fn every_model_lists_the_loader_files_with_hash_names() {
        for m in &MODELS {
            let paths: BTreeSet<_> = m.files.iter().map(|f| f.path).collect();
            assert_eq!(paths, LOADER_FILES.into_iter().collect(), "{}", m.name);
            for f in m.files {
                assert!(matches!(f.blob.len(), 40 | 64), "{} {}", m.name, f.path);
                assert!(
                    f.blob
                        .bytes()
                        .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
                    "{} {}",
                    m.name,
                    f.path
                );
            }
        }
    }

    #[test]
    fn download_sizes_in_mb() {
        let sizes: Vec<_> = MODELS
            .iter()
            .map(|m| (m.name, mb(m.total_size())))
            .collect();
        assert_eq!(
            sizes,
            [
                ("typed-decisions", 846),
                ("multilingual", 678),
                ("english", 846)
            ]
        );
    }

    #[test]
    fn find_takes_names_and_repo_ids() {
        assert_eq!(find("typed-decisions").unwrap().name, "typed-decisions");
        assert_eq!(find("convaiinnovations/laya").unwrap().name, "english");
        assert_eq!(find(" Multilingual ").unwrap().name, "multilingual");
        assert!(find("jev-1").is_none());
    }

    static TINY: LayaModel = LayaModel {
        name: "tiny",
        repo: "test/tiny",
        revision: "0123456789abcdef0123456789abcdef01234567",
        files: &[
            f("model.safetensors", 4, "aa"),
            f("tokenizer/tokenizer.json", 2, "bb"),
        ],
    };

    fn write(path: &Path, len: usize) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, vec![0u8; len]).unwrap();
    }

    #[test]
    fn status_counts_snapshot_files_blobs_and_partial_downloads() {
        let cache = tempfile::tempdir().unwrap();
        let c = cache.path();
        assert_eq!(TINY.status(c), Status::Missing);
        let blob = TINY.blob_path(c, &TINY.files[0]);
        write(&incomplete_path(&blob), 3);
        assert_eq!(TINY.status(c), Status::Partial { have: 3 });
        fs::remove_file(incomplete_path(&blob)).unwrap();
        write(&blob, 4);
        assert_eq!(TINY.status(c), Status::Partial { have: 4 });
        let snap = TINY.snapshot_dir(c);
        write(&snap.join("model.safetensors"), 4);
        write(&snap.join("tokenizer/tokenizer.json"), 1);
        assert_eq!(TINY.status(c), Status::Partial { have: 4 });
        write(&snap.join("tokenizer/tokenizer.json"), 2);
        assert_eq!(TINY.status(c), Status::Complete);
    }

    #[test]
    fn paths_follow_huggingface_hub() {
        let c = Path::new("/c");
        assert_eq!(TINY.repo_dir(c), Path::new("/c/models--test--tiny"));
        assert_eq!(
            TINY.snapshot_dir(c),
            Path::new("/c/models--test--tiny/snapshots/0123456789abcdef0123456789abcdef01234567")
        );
        assert_eq!(
            incomplete_path(&TINY.blob_path(c, &TINY.files[0])),
            Path::new("/c/models--test--tiny/blobs/aa.incomplete")
        );
    }

    #[test]
    fn table_lists_every_model_and_the_cache() {
        let cache = tempfile::tempdir().unwrap();
        let t = table(cache.path());
        assert!(
            t.contains("typed-decisions  convaiinnovations/laya-typed-decisions  1a793eb  not downloaded, 846 MB"),
            "{t}"
        );
        assert!(t.contains("multilingual") && t.contains("english"), "{t}");
        assert_eq!(t.lines().count(), 5, "{t}");
        assert!(
            t.ends_with(&format!("cache: {}\n", cache.path().display())),
            "{t}"
        );
    }
}
