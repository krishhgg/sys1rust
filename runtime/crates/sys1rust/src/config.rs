//! Server configuration: CLI flags, each falling back to an environment variable, and the
//! served checkpoint (a known name resolved in the local Hugging Face cache, or a directory).

use crate::models::{self, LayaModel, Status};
use anyhow::{bail, Context, Result};
use clap::Args;
use laya_core::resolve::{hf_cache_dir, resolve_model_dir};
use laya_core::BackendOptions;
use std::path::{Path, PathBuf};

/// Upstream's default admission bound, also the fallback for an invalid `LAYA_MAX_CONCURRENT`.
pub const DEFAULT_MAX_CONCURRENT: usize = 16;
/// Engine settings. From `results/SPIKE.md`: fp16 GELU, a 512 MiB MLX buffer cache (unbounded,
/// it grows to about RAM size and pushes the machine into swap), 2 GiB wired. From
/// `results/SPEED.md`: dense local attention up to 1,024 tokens, the last head layer only at
/// the rows the scorer reads, and no computing on padding; together 0.90 of the time of the
/// SPIKE settings, with the same answers (probabilities within 0.0005 on all three models).
/// Round 2 of `SPEED.md`: `fuserope`, the encoder's split, RoPE and unpad expand as one Metal
/// kernel, bit-identical to the MLX ops; if the kernel cannot be built on a machine, laya-mlx
/// says so once on stderr at load and runs the MLX ops.
/// Round 3 of `SPEED.md`, all bit-identical to the round 2 default: `band=512`, local attention
/// by chunks from 512 tokens; `nax=all`, the encoder's and head's projections on MLX's NAX gemm
/// loop (macOS 26.2 and a GPU of architecture generation 17 or later; elsewhere, or if the
/// load-time bit check fails, laya-mlx says why on stderr and uses MLX's gemms); and the
/// loading settings `directload,sharehead,parallel_load`. `main` also sets
/// [`laya_mlx::MLX_ENV_DEFAULTS`] unless the user has.
pub const DEFAULT_TUNING: &str = "f16gelu,cache=512,wired=2048,dense_upto=1024,headprune,unpad,fuserope,band=512,nax=all,directload,sharehead,parallel_load";

/// Checkpoint names and their Hugging Face repos, in upstream's naming.
pub const CHECKPOINTS: [(&str, &str); 3] = [
    ("typed-decisions", "convaiinnovations/laya-typed-decisions"),
    ("multilingual", "convaiinnovations/laya-multilingual"),
    ("english", "convaiinnovations/laya"),
];

/// Published ids a client may put in `model`. `convaiinnovations/laya` is deliberately absent,
/// as upstream: it means "let the server choose", which here is the only checkpoint served.
/// The CLI's `--model` has no such meaning and takes every repo in [`CHECKPOINTS`], see
/// [`cli_checkpoint_name`].
pub const PUBLISHED_MODEL_IDS: [(&str, &str); 2] = [
    ("convaiinnovations/laya-multilingual", "multilingual"),
    ("convaiinnovations/laya-typed-decisions", "typed-decisions"),
];

#[derive(Args, Debug, Clone)]
pub struct Config {
    /// Checkpoint to serve: `typed-decisions`, `multilingual`, `english`, one of their repo
    /// ids (`convaiinnovations/laya` is `english`), or a local checkpoint directory. Hub ids
    /// are only looked up in the local HF cache.
    #[arg(long, env = "SYS1_MODEL", default_value = "typed-decisions")]
    pub model: String,
    /// Load `snapshots/<sha>` of the cached repo instead of the revision pinned in
    /// bench/models.lock.json. sys1rust only ever downloads the pinned revision. A single
    /// directory name: letters, digits, `.`, `_` and `-`, not `.` or `..`.
    #[arg(long, env = "SYS1_REVISION")]
    pub revision: Option<String>,
    /// Bind address. Upstream defaults to 0.0.0.0; this is a local runtime.
    #[arg(long, env = "LAYA_HOST", default_value = "127.0.0.1")]
    pub host: String,
    /// Bind port; 0 picks a free port (see the ready line on stdout).
    #[arg(long, env = "LAYA_PORT", default_value_t = 8000)]
    pub port: u16,
    /// If set, `/v1/systemone` requires `Authorization: Bearer <key>`.
    #[arg(long, env = "LAYA_API_KEY")]
    pub api_key: Option<String>,
    /// Requests admitted past auth at once; excess gets 503. Invalid values fall back to 16.
    #[arg(long, env = "LAYA_MAX_CONCURRENT")]
    pub max_concurrent: Option<String>,
    /// laya-mlx settings (comma list, see `Knobs` in crates/laya-mlx).
    #[arg(long, env = "SYS1_MLX_TUNING", default_value = DEFAULT_TUNING)]
    pub tuning: String,
    /// Run the transformer in f32 instead of the checkpoint's f16.
    #[arg(long, env = "SYS1_F32")]
    pub f32: bool,
}

impl Config {
    /// The api key, or `None` when unset or empty (upstream: `os.environ.get(...) or None`).
    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref().filter(|k| !k.is_empty())
    }

    pub fn revision(&self) -> Option<&str> {
        self.revision
            .as_deref()
            .map(str::trim)
            .filter(|r| !r.is_empty())
    }

    pub fn max_concurrent(&self) -> usize {
        resolve_max_concurrent(self.max_concurrent.as_deref())
    }

    pub fn backend_options(&self) -> BackendOptions {
        BackendOptions {
            f32: self.f32,
            tuning: Some(self.tuning.clone()),
            ..Default::default()
        }
    }
}

/// The most permits a tokio semaphore, and so the job channel, can hold. Upstream hands any
/// positive `int` to `asyncio.Semaphore`, which has no ceiling, so a larger
/// `LAYA_MAX_CONCURRENT` is clamped to this instead of refused; the server then admits
/// every request, as upstream would.
pub const MAX_CONCURRENT_CAP: usize = tokio::sync::Semaphore::MAX_PERMITS;

/// Upstream `_resolve_max_concurrent`: unset, unparseable or non-positive means the default.
/// The value is read like Python's `int()` and clamped to [`MAX_CONCURRENT_CAP`].
pub fn resolve_max_concurrent(raw: Option<&str>) -> usize {
    match raw.filter(|s| !s.is_empty()).map(python_int) {
        Some(Some(n)) if n > 0 => n.min(MAX_CONCURRENT_CAP as i128) as usize,
        None => DEFAULT_MAX_CONCURRENT,
        Some(_) => {
            crate::log(format!(
                "invalid LAYA_MAX_CONCURRENT {:?}; falling back to {DEFAULT_MAX_CONCURRENT}",
                raw.unwrap_or_default()
            ));
            DEFAULT_MAX_CONCURRENT
        }
    }
}

/// Python's `int(str)` for a decimal literal: surrounding whitespace, an optional sign, and
/// ASCII digits with single underscores between them (`1_6`). A value past `i128` saturates,
/// which the caller clamps anyway. `int()` also takes non-ASCII decimal digits; those are
/// refused here.
fn python_int(s: &str) -> Option<i128> {
    let s = s.trim();
    let (negative, digits) = match s.as_bytes().first()? {
        b'-' => (true, &s[1..]),
        b'+' => (false, &s[1..]),
        _ => (false, s),
    };
    let mut n: i128 = 0;
    let mut after_digit = false;
    for b in digits.bytes() {
        match b {
            b'0'..=b'9' => {
                n = n.saturating_mul(10).saturating_add(i128::from(b - b'0'));
                after_digit = true;
            }
            b'_' if after_digit => after_digit = false,
            _ => return None,
        }
    }
    // Empty, or ending in an underscore.
    if !after_digit {
        return None;
    }
    Some(if negative { -n } else { n })
}

/// Map a checkpoint name or published id (case-insensitive, trimmed) to the canonical name.
pub fn checkpoint_name(s: &str) -> Option<&'static str> {
    let key = s.trim().to_ascii_lowercase();
    CHECKPOINTS
        .iter()
        .find(|(name, _)| *name == key)
        .map(|(name, _)| *name)
        .or_else(|| {
            PUBLISHED_MODEL_IDS
                .iter()
                .find(|(id, _)| *id == key)
                .map(|(_, name)| *name)
        })
}

/// What `--model` accepts: everything [`checkpoint_name`] does, plus every checkpoint's repo
/// id. On the command line `convaiinnovations/laya` selects the English checkpoint; only in a
/// request's `model` field does it mean "let the server choose".
pub fn cli_checkpoint_name(s: &str) -> Option<&'static str> {
    let key = s.trim().to_ascii_lowercase();
    checkpoint_name(&key).or_else(|| {
        CHECKPOINTS
            .iter()
            .find(|(_, repo)| *repo == key)
            .map(|(name, _)| *name)
    })
}

/// A revision must name one directory under `snapshots/`: `..`, path separators or an empty
/// name would resolve to a checkpoint outside the selected repo while `/health` and routing
/// still report the requested one.
pub fn check_revision(rev: &str) -> Result<()> {
    let plain = rev
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
    if rev.is_empty() || rev == "." || rev == ".." || !plain {
        bail!(
            "invalid revision {rev:?}: expected a snapshot directory name \
             (letters, digits, '.', '_' and '-')"
        );
    }
    Ok(())
}

/// The one checkpoint this process serves.
#[derive(Debug, Clone)]
pub struct ServedModel {
    /// Upstream's checkpoint name (`typed-decisions`), or the directory name for a local path.
    pub name: String,
    /// Hugging Face repo id, or the local path.
    pub repo: String,
    /// Snapshot sha when loaded from the hub cache.
    pub revision: Option<String>,
    pub dir: PathBuf,
}

/// A Laya model's pinned snapshot is not complete in the cache.
#[derive(Debug)]
pub struct NotDownloaded {
    pub model: &'static LayaModel,
    pub dir: PathBuf,
}

impl std::fmt::Display for NotDownloaded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({} at revision {}) is not fully downloaded in {}; get it with `hf download {} --revision {}`",
            self.model.name,
            self.model.repo,
            self.model.revision,
            self.dir.display(),
            self.model.repo,
            self.model.revision
        )
    }
}

impl std::error::Error for NotDownloaded {}

/// Resolve `--model` / `--revision` in the local HF cache. Never downloads.
pub fn resolve_served(model: &str, revision: Option<&str>) -> Result<ServedModel> {
    resolve_served_in(&hf_cache_dir(), model, revision)
}

/// [`resolve_served`] in `cache`. A Laya model without `--revision`, or with its pinned one,
/// loads the pinned snapshot, and the error is a [`NotDownloaded`] when the cache does not
/// have all of it. Any other revision must already be in the cache.
pub fn resolve_served_in(cache: &Path, model: &str, revision: Option<&str>) -> Result<ServedModel> {
    let model = model.trim();
    if let Some(laya) = models::find(model) {
        let dir = match revision {
            Some(rev) => {
                check_revision(rev)?;
                if rev == laya.revision {
                    pinned_dir(cache, laya)?
                } else {
                    let snap = laya.repo_dir(cache).join("snapshots").join(rev);
                    if !snap.is_dir() {
                        bail!(
                            "revision {rev} of {} is not in the local HF cache ({})",
                            laya.repo,
                            snap.display()
                        );
                    }
                    resolve_model_dir(&snap.to_string_lossy(), None)?
                }
            }
            None => pinned_dir(cache, laya)?,
        };
        let revision = snapshot_sha(&dir);
        return Ok(ServedModel {
            name: laya.name.into(),
            repo: laya.repo.into(),
            revision,
            dir,
        });
    }
    if !Path::new(model).exists() {
        bail!(
            "{model:?} is neither a checkpoint name ({}) nor an existing directory",
            CHECKPOINTS
                .iter()
                .map(|(n, _)| *n)
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if revision.is_some() {
        crate::log("--revision is ignored for a local checkpoint directory");
    }
    let dir = resolve_model_dir(model, None)?;
    let dir = dir
        .canonicalize()
        .with_context(|| format!("canonicalize {}", dir.display()))?;
    // A path into the hub cache still gets its repo and sha, so routing and /health say what
    // is really served; any other directory is named after itself.
    if let Some((repo, sha)) = hub_layout(&dir) {
        let name = CHECKPOINTS
            .iter()
            .find(|(_, r)| *r == repo)
            .map(|(n, _)| n.to_string())
            .unwrap_or_else(|| repo.rsplit('/').next().unwrap_or(&repo).to_string());
        return Ok(ServedModel {
            name,
            repo,
            revision: Some(sha),
            dir,
        });
    }
    let name = dir
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_else(|| model.to_string());
    Ok(ServedModel {
        name,
        repo: dir.to_string_lossy().into_owned(),
        revision: None,
        dir,
    })
}

fn pinned_dir(cache: &Path, laya: &'static LayaModel) -> Result<PathBuf> {
    let dir = laya.snapshot_dir(cache);
    if laya.status(cache) != Status::Complete {
        return Err(NotDownloaded { model: laya, dir }.into());
    }
    Ok(resolve_model_dir(&dir.to_string_lossy(), None)?)
}

/// `.../models--org--name/snapshots/<sha>` -> (`org/name`, sha).
fn hub_layout(dir: &Path) -> Option<(String, String)> {
    let sha = dir.file_name()?.to_str()?;
    let snapshots = dir.parent()?;
    if snapshots.file_name()?.to_str()? != "snapshots" {
        return None;
    }
    let repo_dir = snapshots.parent()?.file_name()?.to_str()?;
    let repo = repo_dir.strip_prefix("models--")?.replace("--", "/");
    Some((repo, sha.to_string()))
}

fn snapshot_sha(dir: &Path) -> Option<String> {
    hub_layout(dir).map(|(_, sha)| sha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{self, LayaModel};
    use std::fs;

    fn serve_args(args: &[&str]) -> std::result::Result<Config, clap::Error> {
        use clap::Parser;
        let all = ["sys1rust", "serve"]
            .into_iter()
            .chain(args.iter().copied());
        match crate::cli::Cli::try_parse_from(all)?.command {
            crate::cli::Command::Serve(c) => Ok(c),
            other => panic!("not serve: {other:?}"),
        }
    }

    #[test]
    fn max_concurrent_falls_back_like_upstream() {
        assert_eq!(resolve_max_concurrent(None), 16);
        assert_eq!(resolve_max_concurrent(Some("")), 16);
        assert_eq!(resolve_max_concurrent(Some("abc")), 16);
        assert_eq!(resolve_max_concurrent(Some("0")), 16);
        assert_eq!(resolve_max_concurrent(Some("-3")), 16);
        assert_eq!(resolve_max_concurrent(Some(" 4 ")), 4);
    }

    /// Everything Python's `int()` takes is taken (`+4`, `1_2`), everything it refuses falls
    /// back, and a value tokio cannot hold (upstream's `asyncio.Semaphore` can) is clamped
    /// to the cap instead of panicking in the semaphore or channel constructor.
    #[test]
    fn max_concurrent_reads_python_ints_and_clamps_to_tokio() {
        assert_eq!(resolve_max_concurrent(Some("+4")), 4);
        assert_eq!(resolve_max_concurrent(Some("1_2")), 12);
        for bad in ["1__2", "_12", "12_", "-", "+", "0x10", "12.0", "1e3", " "] {
            assert_eq!(resolve_max_concurrent(Some(bad)), 16, "{bad:?}");
        }
        assert_eq!(
            resolve_max_concurrent(Some("9223372036854775807")),
            MAX_CONCURRENT_CAP
        );
        assert_eq!(
            resolve_max_concurrent(Some(&"9".repeat(40))),
            MAX_CONCURRENT_CAP
        );
        assert_eq!(
            resolve_max_concurrent(Some(&(MAX_CONCURRENT_CAP + 1).to_string())),
            MAX_CONCURRENT_CAP
        );
        assert_eq!(
            resolve_max_concurrent(Some(&(MAX_CONCURRENT_CAP - 1).to_string())),
            MAX_CONCURRENT_CAP - 1
        );
        // The cap is what tokio accepts.
        let _ = tokio::sync::Semaphore::new(MAX_CONCURRENT_CAP);
    }

    #[test]
    fn checkpoint_names_and_published_ids() {
        assert_eq!(checkpoint_name("typed-decisions"), Some("typed-decisions"));
        assert_eq!(checkpoint_name(" English "), Some("english"));
        assert_eq!(
            checkpoint_name("CONVAIINNOVATIONS/LAYA-MULTILINGUAL"),
            Some("multilingual")
        );
        assert_eq!(checkpoint_name("convaiinnovations/laya"), None);
        assert_eq!(checkpoint_name("jev-1"), None);
    }

    #[test]
    fn cli_model_takes_every_repo_id() {
        assert_eq!(cli_checkpoint_name("english"), Some("english"));
        assert_eq!(
            cli_checkpoint_name("convaiinnovations/laya"),
            Some("english")
        );
        assert_eq!(
            cli_checkpoint_name(" Convaiinnovations/Laya-Multilingual "),
            Some("multilingual")
        );
        assert_eq!(
            cli_checkpoint_name("convaiinnovations/laya-typed-decisions"),
            Some("typed-decisions")
        );
        assert_eq!(cli_checkpoint_name("jev-1"), None);
        assert_eq!(cli_checkpoint_name("./laya"), None);
    }

    #[test]
    fn revision_must_be_one_directory_name() {
        assert!(check_revision("1a793eb568e6718f15941d08f85432581df534e3").is_ok());
        assert!(check_revision("v1.2_rc-3").is_ok());
        for bad in [
            "",
            ".",
            "..",
            "../laya-multilingual/snapshots/abc",
            "a/b",
            "a\\b",
            "é",
        ] {
            let e = check_revision(bad).unwrap_err();
            assert!(
                e.to_string().starts_with("invalid revision"),
                "{bad:?}: {e}"
            );
        }
        // Checked before the cache is touched, so a traversal never reaches the filesystem.
        let e = resolve_served("typed-decisions", Some("../other/snapshots/x")).unwrap_err();
        assert!(e.to_string().starts_with("invalid revision"), "{e}");
    }

    /// Every manifest file at its size, as sparse files, so an 842 MB weights file costs
    /// nothing.
    fn fake_snapshot(cache: &Path, m: &LayaModel, rev: &str) -> PathBuf {
        let snap = m.repo_dir(cache).join("snapshots").join(rev);
        for f in m.files {
            let p = snap.join(f.path);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::File::create(&p).unwrap().set_len(f.size).unwrap();
        }
        snap
    }

    #[test]
    fn a_laya_model_loads_its_pinned_revision() {
        let cache = tempfile::tempdir().unwrap();
        let m = models::find("typed-decisions").unwrap();
        let snap = fake_snapshot(cache.path(), m, m.revision);
        for (model, rev) in [
            ("typed-decisions", None),
            ("convaiinnovations/laya-typed-decisions", Some(m.revision)),
        ] {
            let s = resolve_served_in(cache.path(), model, rev).unwrap();
            assert_eq!(s.dir, snap);
            assert_eq!(s.revision.as_deref(), Some(m.revision));
            assert_eq!((s.name.as_str(), s.repo.as_str()), (m.name, m.repo));
        }
    }

    #[test]
    fn refs_main_no_longer_picks_the_snapshot() {
        let cache = tempfile::tempdir().unwrap();
        let m = models::find("english").unwrap();
        let other = "0000000000000000000000000000000000000000";
        fake_snapshot(cache.path(), m, other);
        let refs = m.repo_dir(cache.path()).join("refs");
        fs::create_dir_all(&refs).unwrap();
        fs::write(refs.join("main"), other).unwrap();
        let e = resolve_served_in(cache.path(), "english", None).unwrap_err();
        let missing = e.downcast_ref::<NotDownloaded>().expect("NotDownloaded");
        assert_eq!(missing.model.name, "english");
        assert_eq!(missing.dir, m.snapshot_dir(cache.path()));
        // Naming the other snapshot still loads it.
        let s = resolve_served_in(cache.path(), "english", Some(other)).unwrap();
        assert_eq!(s.revision.as_deref(), Some(other));
    }

    /// With both snapshots complete, the pin wins over the one `refs/main` names.
    #[test]
    fn the_pin_beats_refs_main_when_both_are_cached() {
        let cache = tempfile::tempdir().unwrap();
        let m = models::find("english").unwrap();
        let other = "0000000000000000000000000000000000000000";
        let pinned = fake_snapshot(cache.path(), m, m.revision);
        fake_snapshot(cache.path(), m, other);
        let refs = m.repo_dir(cache.path()).join("refs");
        fs::create_dir_all(&refs).unwrap();
        fs::write(refs.join("main"), other).unwrap();
        let s = resolve_served_in(cache.path(), "english", None).unwrap();
        assert_eq!(s.dir, pinned);
        assert_eq!(s.revision.as_deref(), Some(m.revision));
    }

    #[test]
    fn a_partial_pinned_snapshot_is_not_downloaded() {
        let cache = tempfile::tempdir().unwrap();
        let m = models::find("multilingual").unwrap();
        let snap = fake_snapshot(cache.path(), m, m.revision);
        fs::remove_file(snap.join("tokenizer/tokenizer.json")).unwrap();
        let e = resolve_served_in(cache.path(), "multilingual", None).unwrap_err();
        assert!(e.downcast_ref::<NotDownloaded>().is_some(), "{e:#}");
        let msg = e.to_string();
        assert!(
            msg.contains(m.revision) && msg.contains("not fully downloaded"),
            "{msg}"
        );
    }

    #[test]
    fn an_unpinned_revision_must_be_cached() {
        let cache = tempfile::tempdir().unwrap();
        let e = resolve_served_in(cache.path(), "typed-decisions", Some("abc123")).unwrap_err();
        assert!(e.downcast_ref::<NotDownloaded>().is_none());
        assert!(e.to_string().contains("not in the local HF cache"), "{e}");
    }

    #[test]
    fn hub_layout_is_detected() {
        let p =
            Path::new("/x/hub/models--convaiinnovations--laya-typed-decisions/snapshots/abc123");
        assert_eq!(
            hub_layout(p),
            Some((
                "convaiinnovations/laya-typed-decisions".into(),
                "abc123".into()
            ))
        );
        assert_eq!(hub_layout(Path::new("/x/my-checkpoint")), None);
    }

    #[test]
    fn defaults_and_env_style_flags_parse() {
        let c = serve_args(&[]).unwrap();
        assert_eq!(c.model, "typed-decisions");
        assert_eq!((c.host.as_str(), c.port), ("127.0.0.1", 8000));
        assert_eq!(c.max_concurrent(), 16);
        assert_eq!(c.tuning, DEFAULT_TUNING);
        // Both readers of the settings accept the default: laya-mlx at load, laya-core for
        // `parallel_load`.
        laya_mlx::check_settings(DEFAULT_TUNING).unwrap();
        assert!(c.backend_options().parallel_load().unwrap());
        assert!(!c.f32 && c.api_key().is_none() && c.revision().is_none());
        let c = serve_args(&[
            "--port",
            "0",
            "--max-concurrent",
            "x",
            "--api-key",
            "",
            "--f32",
        ])
        .unwrap();
        assert_eq!(c.port, 0);
        assert_eq!(c.max_concurrent(), 16);
        assert!(c.api_key().is_none());
        assert!(c.backend_options().f32);
        assert!(serve_args(&["--port", "70000"]).is_err());
    }
}
