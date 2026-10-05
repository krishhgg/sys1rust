//! Model downloads into the Hugging Face cache, in huggingface_hub's layout so Python and
//! sys1rust share files. `blobs/<blob>` holds a file's bytes, and
//! `snapshots/<revision>/<path>` is a relative symlink to it. A download writes
//! `blobs/<blob>.incomplete` while holding `.locks/<repo folder>/<blob>.lock` (the lock
//! huggingface_hub takes), resumes with `Range`, and renames the file into place only after
//! its size and hash match the manifest. When an earlier run's partial file hashes wrong, the
//! downloader starts that file over once. It also hashes a blob that is in the cache but not
//! linked from the snapshot before it links to that blob.

use crate::models::{file_size, incomplete_path, mb, repo_folder, LayaModel, ModelFile};
use anyhow::{anyhow, bail, Context, Result};
use sha1::Sha1;
use sha2::{Digest, Sha256};
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use ureq::config::RedirectAuthHeaders;
use ureq::tls::{RootCerts, TlsConfig};

pub const DEFAULT_ENDPOINT: &str = "https://huggingface.co";
/// GETs in a row that add no bytes before a file fails: the first try and 3 retries. A GET
/// that adds bytes resets the count, so a slow download that keeps moving never runs out.
const ATTEMPTS: u32 = 4;
/// ureq has no per-read timeout, only a budget for a whole response body. Each GET gets this
/// long; a download still running then resumes with `Range` on a new GET, and the downloader
/// notices a stalled one within this time.
const BODY_BUDGET: Duration = Duration::from_secs(120);
const CHUNK: usize = 1 << 20;

/// A Hugging Face endpoint to download from.
pub struct Hub {
    agent: ureq::Agent,
    endpoint: String,
    token: Option<String>,
    backoff: Duration,
}

/// Why one GET did not finish a file.
enum Failure {
    /// Another try may work: a connection error, a cut-off body, 429 or 5xx.
    Retry(anyhow::Error),
    /// Another try would fail the same way.
    Fatal(anyhow::Error),
}

impl Hub {
    /// `HF_ENDPOINT` (default `https://huggingface.co`) and `HF_TOKEN`. Empty values count as
    /// unset.
    pub fn from_env() -> Hub {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        let endpoint = var("HF_ENDPOINT");
        Hub::new(
            endpoint.as_deref().unwrap_or(DEFAULT_ENDPOINT),
            var("HF_TOKEN"),
        )
    }

    pub fn new(endpoint: &str, token: Option<String>) -> Hub {
        let agent = ureq::Agent::config_builder()
            .tls_config(
                TlsConfig::builder()
                    .root_certs(RootCerts::PlatformVerifier)
                    .build(),
            )
            .http_status_as_error(false)
            // ureq drops HF_TOKEN on every redirect, because `SameHost` would keep it for another
            // port. The hub sends a large file to a CDN host whose signed URL needs no token.
            .redirect_auth_headers(RedirectAuthHeaders::Never)
            // Ranges and sizes count the file's bytes, so no compressed transfer.
            .accept_encoding("identity")
            .user_agent(concat!("sys1rust/", env!("CARGO_PKG_VERSION")))
            .timeout_resolve(Some(Duration::from_secs(30)))
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_send_request(Some(Duration::from_secs(30)))
            .timeout_recv_response(Some(Duration::from_secs(60)))
            .timeout_recv_body(Some(BODY_BUDGET))
            .build()
            .new_agent();
        Hub {
            agent,
            endpoint: endpoint.trim_end_matches('/').to_string(),
            token,
            backoff: Duration::from_secs(1),
        }
    }

    /// Retry `n` waits `backoff * 2^(n-1)`. Tests set zero.
    pub fn with_backoff(mut self, backoff: Duration) -> Hub {
        self.backoff = backoff;
        self
    }

    pub fn file_url(&self, model: &LayaModel, file: &ModelFile) -> String {
        format!(
            "{}/{}/resolve/{}/{}",
            self.endpoint, model.repo, model.revision, file.path
        )
    }

    /// Put every file of `model` into `cache` and return the snapshot directory. A file
    /// already in the snapshot at its size costs no request and no hashing.
    pub fn ensure<W: Write>(
        &self,
        cache: &Path,
        model: &LayaModel,
        progress: &mut Progress<W>,
    ) -> Result<PathBuf> {
        for file in model.files {
            self.ensure_file(cache, model, file, progress)
                .with_context(|| {
                    format!(
                        "download {} {} at {}",
                        model.repo, file.path, model.revision
                    )
                })?;
        }
        Ok(model.snapshot_dir(cache))
    }

    fn ensure_file<W: Write>(
        &self,
        cache: &Path,
        model: &LayaModel,
        file: &ModelFile,
        progress: &mut Progress<W>,
    ) -> Result<()> {
        let link = model.snapshot_dir(cache).join(file.path);
        if file_size(&link) == Some(file.size) {
            return Ok(());
        }
        let blob = model.blob_path(cache, file);
        let lock_path = cache
            .join(".locks")
            .join(repo_folder(model.repo))
            .join(format!("{}.lock", file.blob));
        create_parent(&lock_path)?;
        create_parent(&blob)?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)
            .with_context(|| format!("open {}", lock_path.display()))?;
        lock.lock()
            .with_context(|| format!("lock {}", lock_path.display()))?;
        // Another process may have finished this file while this one waited. It hashed the
        // blob before it made the link, so this one checks the size only.
        if file_size(&link) == Some(file.size) {
            return Ok(());
        }
        if !cached_blob_ok(&blob, file, progress)? {
            self.download_blob(model, file, &blob, progress)?;
        }
        // The lock also covers the link, so 2 downloaders never replace each other's link.
        link_into_snapshot(&link, file)
        // `lock` drops here, which unlocks.
    }

    fn download_blob<W: Write>(
        &self,
        model: &LayaModel,
        file: &ModelFile,
        blob: &Path,
        progress: &mut Progress<W>,
    ) -> Result<()> {
        let part = incomplete_path(blob);
        let url = self.file_url(model, file);
        // Whether `part` holds bytes an earlier run wrote. Only the hash at the end shows
        // whether they are right. `fetch` clears this when it starts `part` over from byte 0.
        let mut inherited = file_size(&part).is_some_and(|n| n > 0);
        loop {
            self.fetch_all(&url, &part, file, &mut inherited, progress)?;
            let got = blob_name_of(&part, file.blob)
                .with_context(|| format!("hash {}", part.display()))?;
            if got == file.blob {
                return fs::rename(&part, blob)
                    .with_context(|| format!("rename {} to {}", part.display(), blob.display()));
            }
            if !inherited {
                // This call wrote every byte from 0, so the server sent the wrong file.
                let _ = fs::remove_file(&part);
                bail!(
                    "{} hashes to {got}, expected {}; deleted the download",
                    file.path,
                    file.blob
                );
            }
            // An earlier run left wrong bytes in `part`. Start the file over. `inherited` stays
            // false from here on, so a second mismatch fails above instead of looping.
            progress.note(&format!(
                "{}: the partial download from an earlier run hashes to {got}, expected {}; starting it over",
                file.path, file.blob
            ));
            fs::remove_file(&part).with_context(|| format!("remove {}", part.display()))?;
            inherited = false;
        }
    }

    /// GETs into `part` until it holds `file.size` bytes or the attempts run out.
    fn fetch_all<W: Write>(
        &self,
        url: &str,
        part: &Path,
        file: &ModelFile,
        inherited: &mut bool,
        progress: &mut Progress<W>,
    ) -> Result<()> {
        let mut attempt = 1;
        // A GET counts as progress only when it takes `part` past the most bytes it has held
        // in this call, starting from what an earlier run left. A server that ignores `Range`
        // restarts the file on every GET, and this way it still runs out of attempts.
        let mut furthest = file_size(part).filter(|&n| n <= file.size).unwrap_or(0);
        loop {
            match self.fetch(url, part, file, inherited, progress) {
                Ok(()) => return Ok(()),
                // The GET added bytes before it stopped, for example at BODY_BUDGET, so go on
                // from there at once.
                Err(Failure::Retry(_)) if file_size(part).unwrap_or(0) > furthest => {
                    furthest = file_size(part).unwrap_or(0);
                    attempt = 1;
                }
                Err(Failure::Retry(e)) if attempt < ATTEMPTS => {
                    progress.note(&format!("{}: {e:#}; retrying", file.path));
                    std::thread::sleep(self.backoff * 2u32.pow(attempt - 1));
                    attempt += 1;
                }
                Err(Failure::Retry(e) | Failure::Fatal(e)) => return Err(e),
            }
        }
    }

    /// One GET into `part`, continuing from the bytes already there. Sets `inherited` to false
    /// when it starts `part` over from byte 0.
    fn fetch<W: Write>(
        &self,
        url: &str,
        part: &Path,
        file: &ModelFile,
        inherited: &mut bool,
        progress: &mut Progress<W>,
    ) -> std::result::Result<(), Failure> {
        let fatal =
            |e: io::Error| Failure::Fatal(anyhow!(e).context(format!("write {}", part.display())));
        let mut have = file_size(part).unwrap_or(0);
        if have == file.size {
            // All bytes arrived in an earlier run; the hash check decides.
            return Ok(());
        }
        if have > file.size {
            fs::remove_file(part).map_err(fatal)?;
            *inherited = false;
            have = 0;
        }
        let mut req = self.agent.get(url);
        if let Some(token) = &self.token {
            req = req.header("Authorization", format!("Bearer {token}"));
        }
        if have > 0 {
            req = req.header("Range", format!("bytes={have}-"));
        }
        let mut resp = req
            .call()
            .map_err(|e| Failure::Retry(anyhow!(e).context(format!("GET {url}"))))?;
        let status = resp.status().as_u16();
        let append = match status {
            200 => false,
            206 if have > 0 && content_range_start(&resp) == Some(have) => true,
            206 => {
                // The server sent a range this request did not ask for, so start the file over.
                // A fresh download has no `part` yet.
                match fs::remove_file(part) {
                    Ok(()) => {}
                    Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                    Err(e) => return Err(fatal(e)),
                }
                *inherited = false;
                return Err(Failure::Retry(anyhow!("unexpected Content-Range from {url}")));
            }
            429 | 500..=599 => return Err(Failure::Retry(anyhow!("HTTP {status} from {url}"))),
            401 | 403 => {
                return Err(Failure::Fatal(anyhow!(
                    "HTTP {status} from {url}: the hub refused the download; if HF_TOKEN is set, check it"
                )))
            }
            404 => {
                return Err(Failure::Fatal(anyhow!(
                    "HTTP 404 from {url}: not found; if HF_ENDPOINT is set, check it"
                )))
            }
            _ => return Err(Failure::Fatal(anyhow!("HTTP {status} from {url}"))),
        };
        let mut out = if append {
            OpenOptions::new().append(true).open(part)
        } else {
            File::create(part)
        }
        .map_err(fatal)?;
        if !append {
            *inherited = false;
        }
        let mut done = if append { have } else { 0 };
        progress.start(file.path, file.size, done);
        let mut body = resp.body_mut().as_reader();
        let mut buf = vec![0u8; CHUNK];
        loop {
            let n = match body.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    progress.end(done);
                    return Err(Failure::Retry(anyhow!(e).context(format!("read {url}"))));
                }
            };
            if done + n as u64 > file.size {
                drop(out);
                let _ = fs::remove_file(part);
                progress.end(done);
                return Err(Failure::Fatal(anyhow!(
                    "{url} sent more than the expected {} bytes",
                    file.size
                )));
            }
            out.write_all(&buf[..n]).map_err(fatal)?;
            done += n as u64;
            progress.advance(done);
        }
        out.sync_all().map_err(fatal)?;
        progress.end(done);
        if done < file.size {
            return Err(Failure::Retry(anyhow!(
                "{url} stopped at {done} of {} bytes",
                file.size
            )));
        }
        Ok(())
    }
}

/// The first byte of `Content-Range: bytes <first>-<last>/<total>`.
fn content_range_start(resp: &ureq::http::Response<ureq::Body>) -> Option<u64> {
    let value = resp.headers().get("content-range")?.to_str().ok()?;
    value
        .strip_prefix("bytes ")?
        .split('-')
        .next()?
        .trim()
        .parse()
        .ok()
}

/// Whether `blob` already holds `file`'s bytes, by size and then by hash. This deletes a blob
/// of the right size that hashes wrong, so the caller downloads it again. The caller holds
/// the blob's lock.
fn cached_blob_ok<W: Write>(
    blob: &Path,
    file: &ModelFile,
    progress: &mut Progress<W>,
) -> Result<bool> {
    if file_size(blob) != Some(file.size) {
        return Ok(false);
    }
    let got = blob_name_of(blob, file.blob).with_context(|| format!("hash {}", blob.display()))?;
    if got == file.blob {
        return Ok(true);
    }
    progress.note(&format!(
        "{}: the cached blob hashes to {got}, expected {}; downloading it again",
        file.path, file.blob
    ));
    fs::remove_file(blob).with_context(|| format!("remove {}", blob.display()))?;
    Ok(false)
}

/// `snapshots/<rev>/<path>` -> `../../blobs/<blob>`, with one more `..` per folder in `path`:
/// the relative link huggingface_hub makes. The caller holds the blob's lock.
fn link_into_snapshot(link: &Path, file: &ModelFile) -> Result<()> {
    create_parent(link)?;
    let mut target = PathBuf::new();
    for _ in 0..2 + file.path.matches('/').count() {
        target.push("..");
    }
    target.push("blobs");
    target.push(file.blob);
    // Another downloader may have made this link already, and a reader may be using it.
    if fs::read_link(link).is_ok_and(|t| t == target) && file_size(link) == Some(file.size) {
        return Ok(());
    }
    // Make the link under a name no other process uses, then rename it over what is there,
    // such as a link whose blob a user deleted. A rename replaces the old entry in 1 step, so
    // `link` never goes missing.
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let name = link
        .file_name()
        .expect("a snapshot path ends in a file name");
    let mut tmp_name = OsString::from(".");
    tmp_name.push(name);
    tmp_name.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let tmp = link.with_file_name(tmp_name);
    // Only a process with this one's ID, now gone, could have left a link at `tmp`.
    let _ = fs::remove_file(&tmp);
    std::os::unix::fs::symlink(&target, &tmp).with_context(|| format!("link {}", tmp.display()))?;
    fs::rename(&tmp, link).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        anyhow!(e).context(format!("rename {} to {}", tmp.display(), link.display()))
    })
}

fn create_parent(path: &Path) -> Result<()> {
    let dir = path.parent().expect("cache paths have a parent");
    fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))
}

/// The blob name huggingface_hub gives `path`'s bytes: sha256 when `expected` has 64 hex
/// digits (a Git LFS file), the git blob sha1 when it has 40.
pub fn blob_name_of(path: &Path, expected: &str) -> io::Result<String> {
    let mut f = File::open(path)?;
    let len = f.metadata()?.len();
    match expected.len() {
        64 => {
            let mut h = Sha256::new();
            io::copy(&mut f, &mut h)?;
            Ok(hex(&h.finalize()))
        }
        40 => {
            let mut h = Sha1::new();
            h.update(format!("blob {len}\0"));
            io::copy(&mut f, &mut h)?;
            Ok(hex(&h.finalize()))
        }
        n => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("a blob name has 40 or 64 characters, not {n}"),
        )),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Download progress. On a terminal, `Progress` rewrites one line per file in place at most
/// every 200 ms. Elsewhere, such as the `brew services` log, it prints a line at every 10%.
pub struct Progress<W: Write> {
    out: W,
    tty: bool,
    path: String,
    total: u64,
    first: u64,
    started: Instant,
    shown_step: u64,
    shown_at: Option<Instant>,
}

impl Progress<io::Stderr> {
    pub fn stderr() -> Self {
        let err = io::stderr();
        let tty = err.is_terminal();
        Progress::new(err, tty)
    }
}

impl<W: Write> Progress<W> {
    pub fn new(out: W, tty: bool) -> Self {
        Progress {
            out,
            tty,
            path: String::new(),
            total: 1,
            first: 0,
            started: Instant::now(),
            shown_step: 0,
            shown_at: None,
        }
    }

    pub fn into_inner(self) -> W {
        self.out
    }

    fn start(&mut self, path: &str, total: u64, done: u64) {
        self.path = path.to_string();
        self.total = total.max(1);
        self.first = done;
        self.started = Instant::now();
        self.shown_step = done * 10 / self.total;
        self.shown_at = None;
        self.show(done);
    }

    fn advance(&mut self, done: u64) {
        if self.tty {
            if self
                .shown_at
                .is_some_and(|t| t.elapsed() < Duration::from_millis(200))
            {
                return;
            }
            self.show(done);
        } else {
            let step = done * 10 / self.total;
            if step > self.shown_step {
                self.shown_step = step;
                self.show(done);
            }
        }
    }

    fn end(&mut self, done: u64) {
        if self.tty {
            self.show(done);
            let _ = writeln!(self.out);
        }
    }

    fn note(&mut self, msg: &str) {
        if self.tty {
            let _ = writeln!(self.out);
        }
        let _ = writeln!(self.out, "sys1rust: {msg}");
    }

    fn show(&mut self, done: u64) {
        let secs = self.started.elapsed().as_secs_f64().max(0.001);
        let rate = done.saturating_sub(self.first) as f64 / 1e6 / secs;
        let line = format!(
            "sys1rust: {} {:>3}% {}/{} MB {:.1} MB/s",
            self.path,
            done * 100 / self.total,
            mb(done),
            mb(self.total),
            rate
        );
        let _ = if self.tty {
            write!(self.out, "\r{line}\x1b[K")
        } else {
            writeln!(self.out, "{line}")
        };
        let _ = self.out.flush();
        self.shown_at = Some(Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_prints_every_ten_percent_off_a_terminal() {
        let mut p = Progress::new(Vec::new(), false);
        p.start("model.safetensors", 1000, 0);
        for done in (0..=1000).step_by(50) {
            p.advance(done);
        }
        p.end(1000);
        let text = String::from_utf8(p.into_inner()).unwrap();
        let percents: Vec<&str> = text
            .lines()
            .map(|l| l.split_whitespace().nth(2).unwrap())
            .collect();
        assert_eq!(
            percents,
            ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]
        );
    }

    #[test]
    fn blob_names_follow_huggingface_hub() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f");
        fs::write(&p, b"hello\n").unwrap();
        // `git hash-object` of "hello\n", and `shasum -a 256` of it.
        assert_eq!(
            blob_name_of(&p, &"0".repeat(40)).unwrap(),
            "ce013625030ba8dba906f756967f9e9ca394464a"
        );
        assert_eq!(
            blob_name_of(&p, &"0".repeat(64)).unwrap(),
            "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
        );
        assert!(blob_name_of(&p, "abc").is_err());
    }
}
