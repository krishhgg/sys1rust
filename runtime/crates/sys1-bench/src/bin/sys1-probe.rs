//! Steady-state latency per request shape, for comparing backend settings quickly.
//!
//! sys1-probe --workload FILE [--model M] [--tuning SPEC] [--f32] [--warmup N] [--iters N]
//!            [--order grouped|mixed] [--ab SPEC_A SPEC_B [--check]] [--all-rows] [--rows-out FILE]
//!
//! Takes the first row of each shape (`shape.state_tokens`, `shape.n_questions`), or with
//! `--all-rows` every row of the workload. `grouped` runs
//! each shape `iters` times in a row after `warmup` runs; `mixed` cycles through the shapes
//! `iters` times, so every request follows a different shape. Prints a header with the model,
//! the engine (device and precision) and the settings the timings were taken under, then min,
//! p50 and max ms.
//! Backend settings are a laya-mlx `Knobs` spec: `--tuning` (default: none, the upstream
//! reproduction), or `SYS1_MLX` when `--tuning` is absent. Every spec is checked at startup, so
//! a bad one is an error and not a run reported under settings that were never applied. The
//! probe takes specs, not sys1-bench variant names; `sys1-bench --list-variants` shows each
//! variant's spec.
//!
//! `--ab` loads two agents from the same checkpoint, one per settings spec, warms both and runs
//! them alternately (A, B, A, B, ...) on every shape so clock and thermal drift hit both the
//! same. Prints p50 of each and B/A per shape, then the geo-mean of B/A. `--check` also
//! compares A's and B's answers on every picked row: the largest absolute difference over the
//! reported probabilities, over `score` and over `action.act_probability`, and whether every
//! choice, score and noul answer is the same.
//! With `--ab --all-rows` each row is timed on its own (warm-up and iterations per row) and
//! every row's answers are checked; the table has one line per shape: the median over its rows
//! of each side's per-row p50, their ratio, and the geo-mean of the per-row B/A. Without `--ab`,
//! `--all-rows` prints one line per row. `--rows-out` writes the per-row p50s as TSV for later
//! analysis, also without `--ab` (one p50 column). Before any timing, it is an error if the TSV
//! file is the workload file, or if a row `id` has a tab, `\n` or `\r`, which would split its
//! TSV line.
//! Separate process runs on this machine differ by about 5%, which hides 3% effects; the
//! in-process alternation is what makes the comparison usable. The two specs must agree on
//! the process-wide MLX limits (`cache`, `wired`): both agents share one allocator, so the
//! probe refuses a pair that differs there instead of timing both under the second one's.

use anyhow::{Context, Result};
use laya_core::{Agent, BackendOptions};
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<()> {
    let mut workload = String::new();
    let mut model = "typed-decisions".to_string();
    let mut tuning: Option<String> = None;
    let mut f32 = false;
    let mut warmup = 3usize;
    let mut iters = 10usize;
    let mut mixed = false;
    let mut ab: Option<(String, String)> = None;
    let mut check = false;
    let mut all_rows = false;
    let mut rows_out: Option<String> = None;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--workload" => workload = val()?,
            "--model" => model = val()?,
            "--tuning" => tuning = Some(val()?),
            "--f32" => f32 = true,
            "--variant" => anyhow::bail!(
                "--variant {}: the probe takes a settings spec (--tuning SPEC, --f32), not a \
                 sys1-bench variant name; `sys1-bench --list-variants` shows each variant's spec",
                val()?
            ),
            "--warmup" => warmup = val()?.parse()?,
            "--iters" => iters = val()?.parse()?,
            "--order" => mixed = parse_order(&val()?)?,
            "--ab" => ab = Some((val()?, val()?)),
            "--check" => check = true,
            "--all-rows" => all_rows = true,
            "--rows-out" => rows_out = Some(val()?),
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    if check && ab.is_none() {
        anyhow::bail!("--check needs --ab");
    }
    if iters == 0 {
        anyhow::bail!("--iters must be at least 1 (there is no p50 of no runs)");
    }
    // The settings the run is reported under; a bad one is an error, not a silent default. The
    // single run's spec is resolved here (`--tuning`, else `SYS1_MLX`, else none) so that the
    // header prints exactly what the backend is given.
    let spec = match (&ab, tuning) {
        (Some((a, b)), _) => {
            laya_mlx::check_settings(a).with_context(|| format!("--ab A `{a}`"))?;
            laya_mlx::check_settings(b).with_context(|| format!("--ab B `{b}`"))?;
            if let Err(e) = same_process_wide_limits(a, b) {
                anyhow::bail!("--ab: {e}");
            }
            String::new()
        }
        (None, Some(spec)) => {
            laya_mlx::check_settings(&spec).with_context(|| format!("--tuning `{spec}`"))?;
            spec
        }
        (None, None) => {
            let spec = std::env::var("SYS1_MLX").unwrap_or_default();
            laya_mlx::check_settings(&spec).context("SYS1_MLX")?;
            spec
        }
    };
    let bench = std::env::var("BENCH_ROOT").context("source bench/env.sh")?;
    let (dir, sha) = sys1_bench::pinned_model_dir(Path::new(&bench), &model)?;

    let mut picked: Vec<(String, Value)> = Vec::new();
    let file = std::fs::File::open(&workload).with_context(|| format!("open workload {workload}"))?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)?;
        let shape = format!("s{}_q{}", row["shape"]["state_tokens"], row["shape"]["n_questions"]);
        if all_rows || !picked.iter().any(|(s, _)| *s == shape) {
            picked.push((shape, row));
        }
    }
    if picked.is_empty() {
        anyhow::bail!("{workload}: no requests, nothing to time");
    }
    if let Some(path) = rows_out.as_deref() {
        check_rows_out_path(&workload, path)?;
        check_row_ids(&picked, &workload, path)?;
    }

    if let Some((spec_a, spec_b)) = ab {
        return ab_run(&dir, &format!("{model}\t{sha}"), f32, &picked, &spec_a, &spec_b, warmup, iters, mixed, check, rows_out.as_deref());
    }

    let opts = BackendOptions { f32, tuning: Some(spec.clone()), ..Default::default() };
    let agent = Agent::load(&dir, &opts, Box::new(laya_mlx::make_backend))?;
    print!("{}", header(&format!("{model}\t{sha}"), &agent.backend_name(), &[("settings", &spec)]));
    let run = |row: &Value| -> Result<(f64, u64)> {
        let (ms, tok, _) = predict(&agent, row)?;
        Ok((ms, tok))
    };

    let mut times: Vec<Vec<f64>> = vec![Vec::new(); picked.len()];
    let mut tokens = vec![0u64; picked.len()];
    if mixed {
        for (_, row) in &picked {
            for _ in 0..warmup {
                run(row)?;
            }
        }
        for _ in 0..iters {
            for (i, (_, row)) in picked.iter().enumerate() {
                let (ms, tok) = run(row)?;
                times[i].push(ms);
                tokens[i] = tok;
            }
        }
    } else {
        for (i, (_, row)) in picked.iter().enumerate() {
            for _ in 0..warmup {
                run(row)?;
            }
            for _ in 0..iters {
                let (ms, tok) = run(row)?;
                times[i].push(ms);
                tokens[i] = tok;
            }
        }
    }
    println!("shape\ttokens\tmin\tp50\tmax");
    let mut p50s = Vec::with_capacity(picked.len());
    for (i, (shape, _)) in picked.iter().enumerate() {
        let mut t = times[i].clone();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = p50(&times[i]);
        p50s.push(p50);
        println!("{shape}\t{}\t{:.1}\t{:.1}\t{:.1}", tokens[i], t[0], p50, t[t.len() - 1]);
    }
    println!("geo_p50\t{:.1}", geo_mean(&p50s));
    if let Some(path) = rows_out.as_deref() {
        let mut tsv = String::from("row\tshape\ttokens\tp50\n");
        for (i, (shape, row)) in picked.iter().enumerate() {
            let id = row["id"].as_str().unwrap_or("");
            tsv.push_str(&format!("{id}\t{shape}\t{}\t{:.3}\n", tokens[i], p50s[i]));
        }
        std::fs::write(path, tsv).with_context(|| format!("write {path}"))?;
    }
    print_mlx_mb();
    Ok(())
}

/// One request through `agent`: latency in ms, input tokens and the answers object.
fn predict(agent: &Agent, row: &Value) -> Result<(f64, u64, Value)> {
    let t0 = Instant::now();
    let (mut out, _) =
        agent.predict_batch_timed(
            std::slice::from_ref(&row["body"]["state"]),
            &row["body"]["questions"],
            None,
        )?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mut out = out.remove(0);
    let tokens = out["usage"]["input_tokens"].as_u64().unwrap_or(0);
    Ok((ms, tokens, out["answers"].take()))
}

/// The lines above the timings that say what they were taken under: the model (name and
/// pinned sha), the engine name, which carries the device and the precision (`mlx(gpu,f16)`),
/// and one line per settings spec as applied (`(none)` for the upstream defaults). Saved runs
/// with different tuning are told apart by this header alone.
fn header(model: &str, engine: &str, settings: &[(&str, &str)]) -> String {
    let mut h = format!("model\t{model}\nengine\t{engine}\n");
    for (label, spec) in settings {
        let spec = if spec.is_empty() { "(none)" } else { spec };
        h.push_str(&format!("{label}\t{spec}\n"));
    }
    h
}

/// `--ab`: A and B on every shape, strictly alternating, plus the answer check.
#[allow(clippy::too_many_arguments)]
fn ab_run(
    dir: &std::path::Path,
    model: &str,
    f32: bool,
    picked: &[(String, Value)],
    spec_a: &str,
    spec_b: &str,
    warmup: usize,
    iters: usize,
    mixed: bool,
    check: bool,
    rows_out: Option<&str>,
) -> Result<()> {
    let load = |spec: &str| -> Result<Agent> {
        let opts = BackendOptions { f32, tuning: Some(spec.to_string()), ..Default::default() };
        Ok(Agent::load(dir, &opts, Box::new(laya_mlx::make_backend))?)
    };
    let agents = [load(spec_a)?, load(spec_b)?];
    print!("{}", header(model, &agents[0].backend_name(), &[("A", spec_a), ("B", spec_b)]));

    let mut times: [Vec<Vec<f64>>; 2] = [vec![Vec::new(); picked.len()], vec![Vec::new(); picked.len()]];
    let mut tokens = vec![0u64; picked.len()];
    let mut answers: Vec<[Value; 2]> = vec![[Value::Null, Value::Null]; picked.len()];
    let mut pair = |i: usize, row: &Value, record: bool| -> Result<()> {
        for (side, agent) in agents.iter().enumerate() {
            let (ms, tok, ans) = predict(agent, row)?;
            if record {
                times[side][i].push(ms);
                tokens[i] = tok;
                if answers[i][side].is_null() {
                    answers[i][side] = ans;
                }
            }
        }
        Ok(())
    };
    for (i, (_, row)) in picked.iter().enumerate() {
        for _ in 0..warmup {
            pair(i, row, false)?;
        }
        if !mixed {
            for _ in 0..iters {
                pair(i, row, true)?;
            }
        }
    }
    if mixed {
        for _ in 0..iters {
            for (i, (_, row)) in picked.iter().enumerate() {
                pair(i, row, true)?;
            }
        }
    }

    let row_a: Vec<f64> = times[0].iter().map(|t| p50(t)).collect();
    let row_b: Vec<f64> = times[1].iter().map(|t| p50(t)).collect();
    if let Some(path) = rows_out {
        let mut tsv = String::from("row\tshape\ttokens\tp50_A\tp50_B\n");
        for (i, (shape, row)) in picked.iter().enumerate() {
            let id = row["id"].as_str().unwrap_or("");
            tsv.push_str(&format!("{id}\t{shape}\t{}\t{:.3}\t{:.3}\n", tokens[i], row_a[i], row_b[i]));
        }
        std::fs::write(path, tsv).with_context(|| format!("write {path}"))?;
    }
    let groups = shape_groups(picked);
    let mut ratios = Vec::with_capacity(groups.len());
    if groups.len() == picked.len() {
        println!("shape\ttokens\tp50_A\tp50_B\tB/A");
        for (i, (shape, _)) in picked.iter().enumerate() {
            let (a, b) = (row_a[i], row_b[i]);
            ratios.push(b / a);
            println!("{shape}\t{}\t{a:.1}\t{b:.1}\t{:.3}", tokens[i], b / a);
        }
    } else {
        // Several rows per shape: the median over the rows of each side's per-row p50 (the
        // harness's per-shape figure), their ratio, and the geo-mean of the per-row ratios.
        println!("shape\trows\ttokens_med\tmed_p50_A\tmed_p50_B\tB/A\tgeo_rows_B/A");
        for (shape, idx) in &groups {
            let pick = |v: &[f64]| -> Vec<f64> { idx.iter().map(|&i| v[i]).collect() };
            let (a, b) = (median(&pick(&row_a)), median(&pick(&row_b)));
            let per_row: Vec<f64> = idx.iter().map(|&i| row_b[i] / row_a[i]).collect();
            let tok = median(&idx.iter().map(|&i| tokens[i] as f64).collect::<Vec<_>>());
            ratios.push(b / a);
            println!("{shape}\t{}\t{tok:.0}\t{a:.2}\t{b:.2}\t{:.4}\t{:.4}", idx.len(), b / a, geo_mean(&per_row));
        }
        let all: Vec<f64> = row_a.iter().zip(&row_b).map(|(a, b)| b / a).collect();
        println!("geo_rows_B/A\t{:.4}", geo_mean(&all));
    }
    println!("geo_B/A\t{:.3}", geo_mean(&ratios));
    if check {
        let mut total = AnswerCheck::default();
        for [a, b] in &answers {
            total.merge(&compare_answers(a, b));
        }
        println!(
            "check\trows {}\tanswers {}\tmax_prob_diff {:.6}\tmax_score_diff {:.6}\tmax_act_diff {:.6}\t\
             mismatch {}\t(choice {} score {} noul {})\tmissing_values {}",
            picked.len(),
            total.answers,
            total.max_prob_diff,
            total.max_score_diff,
            total.max_act_diff,
            total.mismatches(),
            total.choice_mismatch,
            total.score_mismatch,
            total.noul_mismatch,
            total.missing
        );
    }
    print_mlx_mb();
    Ok(())
}

/// The value of a knob in a spec (the last one wins, as laya-mlx `Knobs` parses it), `None`
/// when the spec does not set it.
fn knob_value(spec: &str, knob: &str) -> Option<String> {
    spec.split(',')
        .map(|kv| kv.split_once('=').unwrap_or((kv, "1")))
        .filter(|(k, _)| *k == knob)
        .next_back()
        .map(|(_, v)| v.to_string())
}

/// `Err` naming the knob when two `--ab` specs would run under different process-wide MLX
/// limits. `cache` and `wired` are applied to the allocator at load, so with two agents in one
/// process both would run under the second load's values.
fn same_process_wide_limits(a: &str, b: &str) -> std::result::Result<(), String> {
    for knob in ["cache", "wired"] {
        if knob_value(a, knob) != knob_value(b, knob) {
            return Err(format!(
                "A and B set different `{knob}` limits; these are process-wide, so B's would \
                 apply to both. Give both specs the same cache and wired values."
            ));
        }
    }
    Ok(())
}

/// `--order`: `mixed` cycles through the shapes, `grouped` runs each shape's iterations in a row.
fn parse_order(order: &str) -> Result<bool> {
    match order {
        "mixed" => Ok(true),
        "grouped" => Ok(false),
        other => anyhow::bail!("unknown order {other}: use mixed or grouped"),
    }
}

fn print_mlx_mb() {
    let mb = |r: mlx_rs::error::Result<usize>| r.map(|b| b >> 20).unwrap_or(0);
    println!(
        "mlx_mb\tactive {}\tcache {}\tpeak {}",
        mb(mlx_rs::memory::active_memory()),
        mb(mlx_rs::memory::cache_memory()),
        mb(mlx_rs::memory::peak_memory())
    );
}

/// Median as the upper middle element (`sorted[len / 2]`), the convention of the bench harness.
fn p50(times: &[f64]) -> f64 {
    let mut t = times.to_vec();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[t.len() / 2]
}

/// `--rows-out` must not name the workload, or the TSV would overwrite the requests it timed.
/// The same file counts by device and inode, so a link to the workload is refused too. A path
/// that does not exist yet passes.
fn check_rows_out_path(workload: &str, path: &str) -> Result<()> {
    use std::os::unix::fs::MetadataExt;
    let (Ok(w), Ok(out)) = (std::fs::metadata(workload), std::fs::metadata(path)) else {
        return Ok(());
    };
    if (w.dev(), w.ino()) == (out.dev(), out.ino()) {
        anyhow::bail!("--rows-out {path} is the workload file {workload}; the TSV would overwrite its requests");
    }
    Ok(())
}

/// `--rows-out` writes each picked row's `id` into one TSV cell, so an id with a tab, `\n` or
/// `\r` is an error naming the id, the workload and the TSV file.
fn check_row_ids(picked: &[(String, Value)], workload: &str, path: &str) -> Result<()> {
    for (_, row) in picked {
        let id = row["id"].as_str().unwrap_or("");
        if id.contains(['\t', '\n', '\r']) {
            anyhow::bail!("--rows-out {path}: the row of {workload} with id {id:?} has a tab or line break in its id, which would split its TSV line");
        }
    }
    Ok(())
}

/// Median of a sample, the mean of the two middle values for an even count.
fn median(xs: &[f64]) -> f64 {
    let mut t = xs.to_vec();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = t.len();
    if n % 2 == 1 {
        t[n / 2]
    } else {
        (t[n / 2 - 1] + t[n / 2]) / 2.0
    }
}

/// The shapes of the picked rows in first-seen order, each with the indices of its rows.
fn shape_groups(picked: &[(String, Value)]) -> Vec<(String, Vec<usize>)> {
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for (i, (shape, _)) in picked.iter().enumerate() {
        match groups.iter_mut().find(|(s, _)| s == shape) {
            Some(g) => g.1.push(i),
            None => groups.push((shape.clone(), vec![i])),
        }
    }
    groups
}

fn geo_mean(xs: &[f64]) -> f64 {
    (xs.iter().map(|x| x.ln()).sum::<f64>() / xs.len() as f64).exp()
}

/// How two answer objects for the same request differ.
#[derive(Debug, Default, Clone, PartialEq)]
struct AnswerCheck {
    /// Questions compared.
    answers: usize,
    /// Largest absolute difference over `probabilities.*` and `noul`.
    max_prob_diff: f64,
    /// Largest absolute difference of the expected `score` (score questions only).
    max_score_diff: f64,
    /// Largest absolute difference of `action.act_probability`, which comes from the pooled
    /// output rather than the marker logits, so it can move when nothing else does.
    max_act_diff: f64,
    /// Answers whose reported `choice`, `score` or `noul` (4 decimals) is not the same.
    choice_mismatch: usize,
    score_mismatch: usize,
    noul_mismatch: usize,
    /// Values (`probabilities.*`, `noul`, `score`, `action.act_probability`) that one side
    /// reports as a number and the other side not at all or not as a number. These never count
    /// as a difference of zero.
    missing: usize,
}

impl AnswerCheck {
    fn mismatches(&self) -> usize {
        self.choice_mismatch + self.score_mismatch + self.noul_mismatch
    }
    fn merge(&mut self, other: &AnswerCheck) {
        self.answers += other.answers;
        self.max_prob_diff = self.max_prob_diff.max(other.max_prob_diff);
        self.max_score_diff = self.max_score_diff.max(other.max_score_diff);
        self.max_act_diff = self.max_act_diff.max(other.max_act_diff);
        self.choice_mismatch += other.choice_mismatch;
        self.score_mismatch += other.score_mismatch;
        self.noul_mismatch += other.noul_mismatch;
        self.missing += other.missing;
    }
}

/// Compare the `answers` objects of two results for the same request, question by question.
/// A question missing or of another type on the B side counts as a mismatch of A's type; a
/// probability key on one side only, or a non-numeric value, counts in `missing`.
fn compare_answers(a: &Value, b: &Value) -> AnswerCheck {
    let mut c = AnswerCheck::default();
    let Some(qa) = a.as_object() else {
        return c;
    };
    for (qid, ans_a) in qa {
        c.answers += 1;
        let ans_b = &b[qid];
        let same = |key: &str| ans_a[key] == ans_b[key] && !ans_a[key].is_null();
        match ans_a["type"].as_str() {
            Some("choice") => c.choice_mismatch += usize::from(!same("choice")),
            Some("score") => c.score_mismatch += usize::from(!same("score")),
            Some("noul") => c.noul_mismatch += usize::from(!same("noul")),
            _ => {}
        }
        // `None` when neither side has the value (a key the type does not report); `Some(None)`
        // when only one side has a number.
        let diff = |x: &Value, y: &Value| -> Option<Option<f64>> {
            match (x.as_f64(), y.as_f64()) {
                (Some(x), Some(y)) => Some(Some((x - y).abs())),
                (None, None) if x.is_null() && y.is_null() => None,
                _ => Some(None),
            }
        };
        let mut missing = 0usize;
        let mut note = |d: Option<Option<f64>>, worst: &mut f64| match d {
            Some(Some(d)) => *worst = worst.max(d),
            Some(None) => missing += 1,
            None => {}
        };
        let (pa, pb) = (ans_a["probabilities"].as_object(), ans_b["probabilities"].as_object());
        if let Some(pa) = pa {
            for (k, pv) in pa {
                note(diff(pv, &ans_b["probabilities"][k]), &mut c.max_prob_diff);
            }
        }
        note(diff(&ans_a["noul"], &ans_b["noul"]), &mut c.max_prob_diff);
        note(diff(&ans_a["score"], &ans_b["score"]), &mut c.max_score_diff);
        note(diff(&ans_a["action"]["act_probability"], &ans_b["action"]["act_probability"]), &mut c.max_act_diff);
        // Keys B reports that A does not.
        let extra = pb.map_or(0, |pb| pb.keys().filter(|k| !pa.is_some_and(|pa| pa.contains_key(*k))).count());
        c.missing += missing + extra;
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn p50_is_the_upper_middle_of_the_sorted_times() {
        assert_eq!(p50(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(p50(&[4.0, 1.0, 3.0, 2.0]), 3.0);
        assert_eq!(p50(&[5.0]), 5.0);
    }

    #[test]
    fn geo_mean_of_ratios() {
        assert!((geo_mean(&[2.0, 0.5]) - 1.0).abs() < 1e-12);
        assert!((geo_mean(&[4.0, 1.0]) - 2.0).abs() < 1e-12);
        assert!((geo_mean(&[0.9, 0.9, 0.9]) - 0.9).abs() < 1e-12);
    }

    #[test]
    fn compare_answers_reports_prob_diff_and_mismatches() {
        let a = json!({
            "pick": {"type": "choice", "choice": "x", "probabilities": {"x": 0.7, "y": 0.3}, "action": {"act_probability": 0.8}},
            "rate": {"type": "score", "score": 1.25, "probabilities": {"0": 0.25, "1": 0.25, "2": 0.5}, "action": {"act_probability": 0.5}},
            "yes": {"type": "noul", "noul": 0.61, "action": {"act_probability": 0.3}},
        });
        let same = compare_answers(&a, &a);
        assert_eq!(
            same,
            AnswerCheck { answers: 3, max_prob_diff: 0.0, ..Default::default() }
        );

        let b = json!({
            "pick": {"type": "choice", "choice": "y", "probabilities": {"x": 0.45, "y": 0.55}, "action": {"act_probability": 0.8}},
            "rate": {"type": "score", "score": 1.2501, "probabilities": {"0": 0.25, "1": 0.25, "2": 0.5}, "action": {"act_probability": 0.52}},
            "yes": {"type": "noul", "noul": 0.6, "action": {"act_probability": 0.3}},
        });
        let c = compare_answers(&a, &b);
        assert_eq!(c.answers, 3);
        assert!((c.max_prob_diff - 0.25).abs() < 1e-12);
        assert!((c.max_score_diff - 0.0001).abs() < 1e-12);
        // The action probability is compared on its own: it comes from the pooled output.
        assert!((c.max_act_diff - 0.02).abs() < 1e-12);
        assert_eq!((c.choice_mismatch, c.score_mismatch, c.noul_mismatch), (1, 1, 1));
        assert_eq!(c.mismatches(), 3);

        // Only the action probability differs: no mismatch, but a nonzero act diff.
        let mut act_only = a.clone();
        act_only["yes"]["action"]["act_probability"] = json!(0.31);
        let c = compare_answers(&a, &act_only);
        assert_eq!((c.mismatches(), c.missing, c.max_prob_diff), (0, 0, 0.0));
        assert!((c.max_act_diff - 0.01).abs() < 1e-12);

        // A question B does not have counts as a mismatch of its type, and every number A
        // reported for it as missing (5 probabilities, 1 noul, 1 score, 3 action probabilities).
        let c = compare_answers(&a, &json!({}));
        assert_eq!((c.choice_mismatch, c.score_mismatch, c.noul_mismatch), (1, 1, 1));
        assert_eq!((c.max_prob_diff, c.max_score_diff, c.max_act_diff), (0.0, 0.0, 0.0));
        assert_eq!(c.missing, 10);
    }

    /// A probability B leaves out, reports as something other than a number, or adds is not a
    /// difference of zero: it shows up in `missing` even when the choice is the same.
    #[test]
    fn compare_answers_counts_missing_and_non_numeric_probabilities() {
        let a = json!({"pick": {"type": "choice", "choice": "x", "probabilities": {"x": 0.7, "y": 0.3}}});
        let dropped = json!({"pick": {"type": "choice", "choice": "x", "probabilities": {"x": 0.7}}});
        let c = compare_answers(&a, &dropped);
        assert_eq!((c.mismatches(), c.missing, c.max_prob_diff), (0, 1, 0.0));
        let text = json!({"pick": {"type": "choice", "choice": "x", "probabilities": {"x": 0.7, "y": "0.3"}}});
        assert_eq!(compare_answers(&a, &text).missing, 1);
        let extra = json!({"pick": {"type": "choice", "choice": "x", "probabilities": {"x": 0.7, "y": 0.3, "z": 0.0}}});
        assert_eq!(compare_answers(&a, &extra).missing, 1);
        // Keys the answer type does not report (no `score` on a choice) are not missing, and
        // neither is an `action` block absent on both sides.
        assert_eq!(compare_answers(&a, &a).missing, 0);
        // An action probability on one side only is missing.
        let mut with_act = a.clone();
        with_act["pick"]["action"] = json!({"act_probability": 0.9});
        assert_eq!(compare_answers(&with_act, &a).missing, 1);
        assert_eq!(compare_answers(&a, &with_act).missing, 1);
        let mut t = AnswerCheck::default();
        t.merge(&c);
        assert_eq!(t.missing, 1);
    }

    /// The header names the model, the engine (with its precision) and every settings spec; an
    /// empty spec is `(none)`.
    #[test]
    fn header_says_what_the_timings_were_taken_under() {
        let h = header("typed-decisions\t1a793eb", "mlx(gpu,f16)", &[("settings", "f16gelu,cache=512")]);
        assert_eq!(h, "model\ttyped-decisions\t1a793eb\nengine\tmlx(gpu,f16)\nsettings\tf16gelu,cache=512\n");
        let h = header("multilingual\te4e9ddf", "mlx(gpu,f32)", &[("settings", "")]);
        assert!(h.contains("engine\tmlx(gpu,f32)\n") && h.ends_with("settings\t(none)\n"), "{h}");
        let h = header("english\t55cf4c4", "mlx(gpu,f16)", &[("A", "f16gelu"), ("B", "f16gelu,unpad")]);
        assert!(h.ends_with("A\tf16gelu\nB\tf16gelu,unpad\n"), "{h}");
    }

    /// An id with a tab, `\n` or `\r` fails `--rows-out` with the id as Rust prints it, the
    /// workload and the TSV file; other ids pass.
    #[test]
    fn rows_out_rejects_ids_that_split_a_line() {
        let row = |id: &str| ("s64_q1".to_string(), serde_json::json!({ "id": id }));
        assert!(check_row_ids(&[row("a-1"), row("b 2"), row("")], "w.jsonl", "rows.tsv").is_ok());
        for (id, shown) in [("a\tb", r#""a\tb""#), ("a\nb", r#""a\nb""#), ("a\rb", r#""a\rb""#)] {
            let e = check_row_ids(&[row("ok"), row(id)], "w.jsonl", "rows.tsv").unwrap_err().to_string();
            assert!(e.contains(shown) && e.contains("w.jsonl") && e.contains("rows.tsv"), "{e}");
        }
    }

    /// `--rows-out` naming the workload, by its path or through a link, fails; another file
    /// or a path that does not exist passes.
    #[test]
    fn rows_out_must_not_be_the_workload() {
        let dir = std::env::temp_dir().join(format!("sys1-probe-rows-out-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = |name: &str| dir.join(name).to_str().unwrap().to_string();
        std::fs::write(path("w.jsonl"), "{}\n").unwrap();
        std::fs::write(path("other.tsv"), "").unwrap();
        std::os::unix::fs::symlink(path("w.jsonl"), path("link.tsv")).unwrap();
        std::fs::hard_link(path("w.jsonl"), path("hard.tsv")).unwrap();
        for out in [path("w.jsonl"), path("link.tsv"), path("hard.tsv"), format!("{}/./w.jsonl", dir.display())] {
            let e = check_rows_out_path(&path("w.jsonl"), &out).unwrap_err().to_string();
            assert!(e.contains("is the workload file"), "{out}: {e}");
        }
        assert!(check_rows_out_path(&path("w.jsonl"), &path("other.tsv")).is_ok());
        assert!(check_rows_out_path(&path("w.jsonl"), &path("new.tsv")).is_ok());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn median_and_shape_groups() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        let row = |s: &str| (s.to_string(), Value::Null);
        let picked = [row("s64_q1"), row("s64_q1"), row("s128_q1"), row("s64_q1")];
        assert_eq!(
            shape_groups(&picked),
            vec![("s64_q1".to_string(), vec![0, 1, 3]), ("s128_q1".to_string(), vec![2])]
        );
    }

    #[test]
    fn order_is_mixed_or_grouped() {
        assert!(parse_order("mixed").unwrap());
        assert!(!parse_order("grouped").unwrap());
        for bad in ["Mixed", "mixd", "random", ""] {
            assert!(parse_order(bad).is_err(), "{bad:?} was accepted");
        }
    }

    #[test]
    fn ab_specs_must_share_the_process_wide_limits() {
        assert!(same_process_wide_limits("f16gelu,cache=512,wired=2048", "f16gelu,cache=512,wired=2048,unpad").is_ok());
        assert!(same_process_wide_limits("wired=2048,cache=512", "cache=512,wired=2048").is_ok());
        assert!(same_process_wide_limits("", "f16gelu").is_ok());
        let e = same_process_wide_limits("cache=512", "cache=1024").unwrap_err();
        assert!(e.contains("`cache`"), "{e}");
        let e = same_process_wide_limits("cache=512,wired=2048", "cache=512").unwrap_err();
        assert!(e.contains("`wired`"), "{e}");
        assert!(same_process_wide_limits("", "wired=1024").is_err());
    }

    #[test]
    fn answer_check_merges() {
        let mut t = AnswerCheck { answers: 1, max_prob_diff: 0.1, ..Default::default() };
        t.merge(&AnswerCheck {
            answers: 2,
            max_prob_diff: 0.3,
            max_score_diff: 0.2,
            max_act_diff: 0.05,
            score_mismatch: 1,
            ..Default::default()
        });
        assert_eq!(t.answers, 3);
        assert_eq!((t.max_prob_diff, t.max_score_diff, t.max_act_diff), (0.3, 0.2, 0.05));
        assert_eq!(t.mismatches(), 1);
    }
}
