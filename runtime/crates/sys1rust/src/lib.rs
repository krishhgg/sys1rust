//! This library holds the `sys1rust` command's argument parser, HTTP server, configuration
//! and inference worker. The server speaks upstream Laya's `/v1/systemone` protocol (see
//! `laya_serve.py` and `docs/http-api.md` upstream).
//!
//! The binary in `main.rs` wires these pieces together; they are a library so the HTTP layer
//! can be tested against a fake [`worker::Predictor`] without MLX or a checkpoint:
//!
//! - [`cli`]: the `serve`, `pull` and `models` subcommands and the `--version` line.
//! - [`config`]: CLI flags with env-var fallbacks, and resolving the served checkpoint.
//! - [`worker`]: the single inference thread, its bounded job channel and reply oneshots.
//! - [`validate`]: upstream's request checks (400/413/422 and the `model` field rule).
//! - [`http`]: the axum router, admission, auth, body cap and error bodies.
//! - [`agent`]: the real predictor over `laya_core::Agent`, with its warm-up requests.
//! - [`models`]: the pinned manifest of the three Laya models and the `models` table.
//! - [`download`]: model downloads into the Hugging Face cache, with resume and hash checks.

pub mod agent;
pub mod cli;
pub mod config;
pub mod download;
pub mod http;
pub mod models;
pub mod validate;
pub mod worker;

pub use config::{Config, ServedModel};
pub use http::{router, serve, AppState};
pub use worker::{Predictor, Worker, WorkerHandle};

/// Write one human-readable log line to stderr (stdout is reserved for the ready line).
pub fn log(msg: impl AsRef<str>) {
    eprintln!("sys1rust: {}", msg.as_ref());
}

/// Fold line breaks in client-controlled text so it cannot forge log entries.
pub fn sanitize(s: &str) -> String {
    s.replace('\r', "\\r").replace('\n', "\\n")
}
