//! Bench adapter for the sys1rust runtime (bench/PLAN.md, "Adapter interface").
//!
//! run --list-variants
//! run --variant V --model M --workload FILE --out FILE [--warmup N] [--repeats R] [--duration S]
//!
//! In-process only. `latency_ms` covers `Agent::predict_batch_timed` for one request: sequence
//! building, tokenization, the forward pass with its GPU sync, and answer decoding. Each result
//! also records the time spent in each of those phases (`phase_us`).

use anyhow::{bail, Context, Result};
use laya_core::{Agent, BackendOptions};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const CONTENDER: &str = "sys1rust";
const MODELS: [&str; 3] = ["typed-decisions", "multilingual", "english"];
/// Limits of a model that every variant inherits, printed with `--list-variants` as
/// `model_notes`. English has upstream references for the smoke, short and cold workloads only:
/// its correctness runs report gold accuracy but cannot report the choice agreement with
/// upstream that qualifies a speed result (`bench/harness/compare.py` prints "gold only").
const MODEL_NOTES: [(&str, &str); 1] = [(
    "english",
    "no upstream correctness reference (bench/reference/english has smoke, short and cold only); \
     correctness runs are gold-only and do not qualify speed results",
)];

struct Variant {
    name: &'static str,
    notes: &'static str,
    opts: fn() -> BackendOptions,
    /// Set laya-mlx's MLX environment defaults (`MLX_ENV_DEFAULTS`) that the caller has not
    /// set, as sys1rust does.
    mlx_env: bool,
}

fn tuned(tuning: &str) -> BackendOptions {
    BackendOptions { tuning: Some(tuning.into()), ..Default::default() }
}

const VARIANTS: [Variant; 6] = [
    Variant {
        name: "mlx-fp16",
        notes: "laya-r-mlx 914c9a7 as forked, unchanged: fp16 weights, questions in one batch padded to the \
                longest row. Its GELU promotes activations to fp32 from the first MLP on.",
        opts: || tuned(""),
        mlx_env: false,
    },
    Variant {
        name: "mlx-fp16-fix",
        notes: "mlx-fp16 with GELU kept in fp16, so activations and gemms stay fp16.",
        opts: || tuned("f16gelu"),
        mlx_env: false,
    },
    Variant {
        name: "mlx-fp16-fast",
        notes: "mlx-fp16-fix with MLX's buffer cache capped at 512 MiB (the default lets freed buffers \
                grow to about the size of RAM when request lengths vary) and a 2 GiB wired limit so \
                the weights stay resident.",
        opts: || tuned("f16gelu,cache=512,wired=2048"),
        mlx_env: false,
    },
    Variant {
        name: "mlx-fp16-lean",
        notes: "mlx-fp16-fast plus the results/SPEED.md work reductions, the sys1rust default: dense local \
                attention up to 1,024 tokens, the last head layer only at the scorer's rows, no computing \
                on padding, (round 2) the split + RoPE + unpad expand as one Metal kernel, and (round 3) \
                local attention by chunks from 512 tokens, the projections on MLX's NAX gemm loop, the \
                loading settings, and MLX_MAX_MB_PER_BUFFER=10 unless set.",
        opts: || tuned("f16gelu,cache=512,wired=2048,dense_upto=1024,headprune,unpad,fuserope,band=512,nax=all,directload,sharehead,parallel_load"),
        mlx_env: true,
    },
    Variant {
        name: "mlx-env",
        notes: "Exploration only: backend settings from the SYS1_MLX environment variable.",
        opts: || tuned(&std::env::var("SYS1_MLX").unwrap_or_default()),
        mlx_env: false,
    },
    Variant {
        name: "mlx-fp32",
        notes: "Same as mlx-fp16 with the transformer in fp32.",
        opts: || BackendOptions { f32: true, ..tuned("") },
        mlx_env: false,
    },
];

/// MLX allocator state after a request: active, buffer cache and peak, in MiB. Metal buffers
/// do not show up in RSS, so this is the only view of the GPU-side memory.
fn mlx_mb() -> Value {
    let mb = |r: mlx_rs::error::Result<usize>| r.map(|b| b >> 20).unwrap_or(0);
    json!({
        "active": mb(mlx_rs::memory::active_memory()),
        "cache": mb(mlx_rs::memory::cache_memory()),
        "peak": mb(mlx_rs::memory::peak_memory()),
    })
}

/// The MLX environment variables in `MLX_ENV_DEFAULTS` as this process sees them, null when
/// unset, whoever set them.
fn mlx_env() -> Value {
    let vars = laya_mlx::MLX_ENV_DEFAULTS.iter().map(|(key, _)| (key.to_string(), json!(std::env::var(key).ok())));
    Value::Object(vars.collect())
}

fn now_unix() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

#[derive(Default)]
struct Args {
    list: bool,
    variant: String,
    model: String,
    workload: String,
    out: String,
    warmup: usize,
    repeats: usize,
    duration: Option<f64>,
    concurrency: usize,
}

fn parse_args() -> Result<Args> {
    let mut a = Args { warmup: 5, repeats: 1, concurrency: 1, ..Default::default() };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--list-variants" => a.list = true,
            "--variant" => a.variant = val()?,
            "--model" => a.model = val()?,
            "--workload" => a.workload = val()?,
            "--out" => a.out = val()?,
            "--warmup" => a.warmup = val()?.parse()?,
            "--repeats" => a.repeats = val()?.parse()?,
            "--duration" => a.duration = Some(parse_duration(&val()?)?),
            "--concurrency" => a.concurrency = val()?.parse()?,
            other => bail!("unknown argument {other}"),
        }
    }
    Ok(a)
}

/// `--duration` in seconds: a finite, positive number. Zero or a negative one would write only
/// the meta line and exit 0, so the harness would skip a run that measured nothing; NaN or
/// infinity would never end the run.
fn parse_duration(s: &str) -> Result<f64> {
    let d: f64 = s.parse().with_context(|| format!("--duration {s}: not a number"))?;
    if !d.is_finite() || d <= 0.0 {
        bail!("--duration {s}: must be a finite, positive number of seconds");
    }
    Ok(d)
}

/// One request per non-blank JSONL line. A line that cannot be read or parsed is an error, so
/// a run never finishes with fewer requests than the workload holds.
fn read_workload(path: &str) -> Result<Vec<Value>> {
    let file = std::fs::File::open(path).with_context(|| format!("open workload {path}"))?;
    let mut rows = Vec::new();
    for (i, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| format!("{path}:{}: read error", i + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        rows.push(serde_json::from_str(&line).with_context(|| format!("{path}:{}: invalid JSON", i + 1))?);
    }
    if rows.is_empty() {
        bail!("workload {path} has no requests");
    }
    Ok(rows)
}

fn main() -> Result<()> {
    let t_process_start = now_unix();
    let args = parse_args()?;
    if args.list {
        let v: Vec<Value> = VARIANTS
            .iter()
            .map(|v| json!({
                "variant": v.name, "models": MODELS, "mode": "inproc", "max_state_tokens": null, "notes": v.notes,
                "model_notes": MODEL_NOTES.iter().map(|(m, n)| (m.to_string(), Value::from(*n))).collect::<serde_json::Map<_, _>>(),
            }))
            .collect();
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }
    if args.concurrency > 1 {
        bail!("--concurrency is for http mode; this adapter is in-process");
    }
    if args.repeats == 0 && args.duration.is_none() {
        bail!("--repeats must be at least 1 unless --duration is given");
    }
    let variant = VARIANTS
        .iter()
        .find(|v| v.name == args.variant)
        .with_context(|| format!("unknown variant {}", args.variant))?;
    if variant.mlx_env {
        // Before any MLX call (MLX reads these once) and while this is the only thread.
        laya_mlx::set_mlx_env_defaults();
    }
    let opts = (variant.opts)();
    if let Some(spec) = &opts.tuning {
        // The meta line labels the run with these settings, so one the backend would not apply
        // is an error here, before anything is loaded or written.
        laya_mlx::check_settings(spec).with_context(|| format!("variant {}: settings `{spec}`", variant.name))?;
    }
    let bench = std::env::var("BENCH_ROOT").context("BENCH_ROOT is not set (source bench/env.sh)")?;
    let (dir, sha) = sys1_bench::pinned_model_dir(Path::new(&bench), &args.model)?;

    let rows = read_workload(&args.workload)?;

    let t_load = Instant::now();
    let agent = Agent::load(&dir, &opts, Box::new(laya_mlx::make_backend))?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    let mut out = std::io::BufWriter::new(std::fs::File::create(&args.out)?);
    let meta = json!({
        "type": "meta", "contender": CONTENDER, "variant": variant.name, "model": args.model,
        "model_sha": sha, "code_version": std::env::var("SYS1_CODE_VERSION").unwrap_or_else(|_| "unknown".into()),
        "backend": "mlx", "mode": "inproc", "load_ms": (load_ms * 10.0).round() / 10.0,
        "pid": std::process::id(), "t_process_start": t_process_start,
        "engine": agent.backend_name(), "tuning": opts.tuning, "mlx_env": mlx_env(), "warmup": args.warmup, "repeats": args.repeats, "duration": args.duration,
    });
    writeln!(out, "{meta}")?;

    let run_one = |row: &Value| -> (f64, f64, Result<(Value, laya_core::agent::Timing)>) {
        let body = &row["body"];
        let t_start = now_unix();
        let t0 = Instant::now();
        let r = agent
            .predict_batch_timed(std::slice::from_ref(&body["state"]), &body["questions"], None)
            .map(|(mut v, t)| (v.remove(0), t))
            .map_err(anyhow::Error::from);
        (t_start, t0.elapsed().as_secs_f64() * 1000.0, r)
    };

    for row in rows.iter().take(args.warmup) {
        if let (_, _, Err(e)) = run_one(row) {
            eprintln!("warmup {}: {e:#}", row["id"]);
        }
    }

    let t_measure = Instant::now();
    let mut repeat = 0usize;
    'outer: loop {
        for row in &rows {
            if let Some(d) = args.duration {
                if t_measure.elapsed().as_secs_f64() >= d {
                    break 'outer;
                }
            }
            let (t_start, latency_ms, r) = run_one(row);
            let line = match r {
                Ok((res, t)) => json!({
                    "type": "result", "id": row["id"], "repeat": repeat, "t_start": t_start,
                    "latency_ms": (latency_ms * 1000.0).round() / 1000.0,
                    "answers": res["answers"],
                    "input_tokens": res["usage"]["input_tokens"],
                    "phase_us": {"encode": t.encode_us, "forward": t.forward_us, "decode": t.decode_us},
                    "batch": {"rows": t.batch_rows, "len": t.batch_len},
                    "mlx_mb": mlx_mb(),
                }),
                Err(e) => json!({"type": "result", "id": row["id"], "repeat": repeat, "error": format!("{e:#}")}),
            };
            writeln!(out, "{line}")?;
            out.flush()?;
        }
        repeat += 1;
        if args.duration.is_none() && repeat >= args.repeats {
            break;
        }
    }
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_must_be_finite_and_positive() {
        assert_eq!(parse_duration("1.5").unwrap(), 1.5);
        assert_eq!(parse_duration("30").unwrap(), 30.0);
        for bad in ["0", "-1", "nan", "inf", "-inf", "abc", ""] {
            assert!(parse_duration(bad).is_err(), "{bad:?} was accepted");
        }
    }

    /// Every named variant's settings parse, so the spec printed in the meta line is the one
    /// the backend applies.
    #[test]
    fn every_variant_spec_is_valid() {
        for v in &VARIANTS {
            if v.name == "mlx-env" {
                continue;
            }
            let opts = (v.opts)();
            laya_mlx::check_settings(opts.tuning.as_deref().unwrap_or("")).unwrap_or_else(|e| panic!("{}: {e}", v.name));
        }
    }
}
