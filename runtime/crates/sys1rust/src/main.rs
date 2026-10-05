//! `sys1rust` entry point. `serve` sets the MLX environment defaults the user has not set,
//! resolves the checkpoint, loads and warms it on the inference thread, binds, prints the
//! one-line JSON ready event to stdout, serves until SIGINT/SIGTERM, then finishes in-flight
//! requests and exits 0. `pull` downloads a model and prints its snapshot directory. With
//! `HF_HUB_OFFLINE` on, it prints the directory only when the snapshot is complete. `models`
//! prints the Laya models table to stdout. Everything human-readable goes to stderr.

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use laya_core::resolve::hf_cache_dir;
use serde_json::json;
use std::io::Write;
use std::process::ExitCode;
use std::time::Duration;
use sys1rust::agent::AgentPredictor;
use sys1rust::cli::{offline_from_env, Cli, Command, PullArgs};
use sys1rust::config::{resolve_or_download, Config};
use sys1rust::download::{Hub, Progress};
use sys1rust::{log, models, router, serve, AppState, Worker};

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
        Command::Pull(args) => pull_cmd(args),
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
    print_out(&models::table(&hf_cache_dir()), "the models table")
}

/// Write `text` to stdout. `what` names it in the error.
fn print_out(text: &str, what: &str) -> Result<()> {
    let mut out = std::io::stdout().lock();
    match out.write_all(text.as_bytes()).and_then(|()| out.flush()) {
        // A reader that quits early, such as `head`, closes the pipe. That is not an error.
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
        r => r.with_context(|| format!("writing {what} to stdout")),
    }
}

fn pull_cmd(args: &PullArgs) -> Result<()> {
    let model = models::find(&args.model).ok_or_else(|| {
        anyhow!(
            "unknown model {:?}; pull takes typed-decisions, multilingual, english or one of their repo ids",
            args.model
        )
    })?;
    let cache = hf_cache_dir();
    // A complete snapshot needs no download, so HF_HUB_OFFLINE matters only when it is not.
    let dir = if model.status(&cache) == models::Status::Complete {
        model.snapshot_dir(&cache)
    } else if offline_from_env() {
        bail!(
            "HF_HUB_OFFLINE is on, so nothing is downloaded; unset it to pull {}",
            model.name
        );
    } else {
        Hub::from_env().ensure(&cache, model, &mut Progress::stderr())?
    };
    print_out(&format!("{}\n", dir.display()), "the snapshot directory")
}

fn serve_cmd(cfg: &Config, mlx_env: &[(&str, &str)]) -> Result<()> {
    for (key, value) in mlx_env {
        log(format!("{key}={value} (default; set {key} to override)"));
    }
    let hub = (!cfg.offline()).then(Hub::from_env);
    let served = resolve_or_download(
        &hf_cache_dir(),
        &cfg.model,
        cfg.revision(),
        hub.as_ref(),
        &mut Progress::stderr(),
    )?;
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
