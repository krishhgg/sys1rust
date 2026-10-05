//! `sys1rust` entry point. `serve` sets the MLX environment defaults the user has not set,
//! resolves the checkpoint, loads and warms it on the inference thread, binds, prints the
//! one-line JSON ready event to stdout, serves until SIGINT/SIGTERM, then finishes in-flight
//! requests and exits 0. `models` prints the Laya models table to stdout. Everything
//! human-readable goes to stderr.

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::json;
use std::io::Write;
use std::process::ExitCode;
use std::time::Duration;
use sys1rust::agent::AgentPredictor;
use sys1rust::cli::{Cli, Command};
use sys1rust::config::{resolve_served, Config};
use sys1rust::{log, router, serve, AppState, Worker};

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match &cli.command {
        Command::Serve(cfg) => {
            // MLX reads its buffer limits once, when the inference thread's first GPU
            // operation creates the Metal device. Set the unset ones here, while this is the
            // only thread and before any MLX call, so they act as if the user had set them
            // before starting the server.
            let mlx_env = laya_mlx::set_mlx_env_defaults();
            serve_cmd(cfg, &mlx_env)
        }
        Command::Models => models_cmd(),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            log(format!("error: {e:#}"));
            ExitCode::FAILURE
        }
    }
}

fn models_cmd() -> Result<()> {
    // Filled in by Task 3.
    Ok(())
}

fn serve_cmd(cfg: &Config, mlx_env: &[(&str, &str)]) -> Result<()> {
    for (key, value) in mlx_env {
        log(format!("{key}={value} (default; set {key} to override)"));
    }
    let served = resolve_served(&cfg.model, cfg.revision())?;
    log(format!(
        "loading {} ({}{}) from {}",
        served.name,
        served.repo,
        served
            .revision
            .as_deref()
            .map(|r| format!(" @ {r}"))
            .unwrap_or_default(),
        served.dir.display()
    ));
    let opts = cfg.backend_options();
    let max_concurrent = cfg.max_concurrent();
    let (worker, ready) = Worker::spawn(
        AgentPredictor::factory(served.dir.clone(), opts),
        max_concurrent,
    );
    let ready = ready
        .recv()
        .context("inference thread exited before reporting")??;
    log(format!(
        "model loaded in {:.1} ms, warm-up in {:.1} ms, engine {}",
        ready.load_ms, ready.warmup_ms, ready.engine
    ));

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind((cfg.host.as_str(), cfg.port))
            .await
            .with_context(|| format!("bind {}:{}", cfg.host, cfg.port))?;
        let addr = listener.local_addr()?;
        let line = json!({
            "event": "listening",
            "addr": addr.to_string(),
            "model": served.name,
            "repo": served.repo,
            "revision": served.revision,
            "engine": ready.engine,
            "load_ms": round1(ready.load_ms),
            "warmup_ms": round1(ready.warmup_ms),
            "max_concurrent": max_concurrent,
            "pid": std::process::id(),
        });
        let mut out = std::io::stdout().lock();
        writeln!(out, "{line}")?;
        out.flush()?;
        drop(out);
        log(format!(
            "listening on http://{addr} (auth {})",
            if cfg.api_key().is_some() { "on" } else { "off" }
        ));

        let state = AppState::new(
            worker.handle.clone(),
            served.clone(),
            ready.engine.clone(),
            cfg.tuning.clone(),
            cfg.api_key().map(str::to_string),
            max_concurrent,
        );
        serve(listener, router(state), shutdown_signal()).await?;
        Ok::<(), anyhow::Error>(())
    })?;
    // All connections are done; the runtime and the app state go away with `rt`.
    drop(rt);
    if !worker.join(Duration::from_secs(10)) {
        log("inference thread did not exit in 10 s; exiting anyway");
    }
    log("stopped");
    Ok(())
}

/// Resolves on SIGINT or SIGTERM.
async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    tokio::select! {
        r = tokio::signal::ctrl_c() => { r.expect("install SIGINT handler"); }
        _ = term.recv() => {}
    }
    log("shutdown requested; finishing in-flight requests");
}

fn round1(ms: f64) -> f64 {
    (ms * 10.0).round() / 10.0
}
