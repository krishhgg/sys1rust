//! Helpers shared by the model tests of this crate: the checkpoints, the bench files and the
//! answer comparison. Included with `mod common;` from each test file that needs it.

#![allow(dead_code)]

use laya_core::resolve::resolve_model_dir;
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// The three published checkpoints, as `bench/models.lock.json` names them.
pub const CHECKPOINTS: [(&str, &str); 3] = [
    ("typed-decisions", "convaiinnovations/laya-typed-decisions"),
    ("multilingual", "convaiinnovations/laya-multilingual"),
    ("english", "convaiinnovations/laya"),
];

/// Set to `1` for the strict mode of the model tests: a checkpoint that does not resolve fails
/// the test instead of being skipped, and every test must run all of [`CHECKPOINTS`]. Use it
/// for the runs that back a release.
pub const ALL_CHECKPOINTS_ENV: &str = "SYS1_TEST_ALL_CHECKPOINTS";

fn strict() -> bool {
    std::env::var_os(ALL_CHECKPOINTS_ENV).is_some_and(|v| v == "1")
}

/// The cached directory of checkpoint `repo`, or `None` with a note when it does not resolve.
/// In strict mode ([`ALL_CHECKPOINTS_ENV`]) that is a panic naming the error.
pub fn checkpoint_dir(name: &str, repo: &str) -> Option<PathBuf> {
    match resolve_model_dir(repo, None) {
        Ok(dir) => Some(dir),
        Err(e) if strict() => panic!("{name}: {repo} does not resolve ({e}); {ALL_CHECKPOINTS_ENV}=1 needs every checkpoint"),
        Err(e) => {
            eprintln!("{name:<16} skipped: {repo} does not resolve ({e})");
            None
        }
    }
}

/// The end of a test over [`CHECKPOINTS`]: `ran` of them ran. At least one must, and in strict
/// mode all of them.
pub fn assert_ran(ran: usize) {
    assert!(ran > 0, "no checkpoint in the HF cache; source bench/env.sh and download one");
    if strict() {
        assert_eq!(ran, CHECKPOINTS.len(), "{ALL_CHECKPOINTS_ENV}=1: {ran} of {} checkpoints ran", CHECKPOINTS.len());
    }
}

/// `bench/` of this checkout.
pub fn bench_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../bench")
        .canonicalize()
        .unwrap()
}

pub fn read_jsonl(path: &Path) -> Vec<Value> {
    BufReader::new(std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
        .lines()
        .map(|l| l.unwrap())
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(&l).unwrap())
        .collect()
}

/// What an answer decides, by the rule of `bench/harness/compare.py` (`categorical`): the
/// `choice` label (the most probable option when `choice` is missing), the noul side
/// (`noul >= 0.5`), or the most probable level of a score answer as an integer (without
/// `probabilities`, an integer `score`; a fractional one decides nothing). `Null` when the
/// answer decides nothing, which never agrees with anything (see [`agree`]).
pub fn categorical(answer: &Value) -> Value {
    match answer["type"].as_str() {
        Some("choice") => match &answer["choice"] {
            Value::Null => argmax_key(&answer["probabilities"]).map_or(Value::Null, Value::from),
            choice => choice.clone(),
        },
        Some("noul") => answer["noul"].as_f64().map_or(Value::Null, |v| Value::from(v >= 0.5)),
        Some("score") => {
            if let Some(level) = argmax_key(&answer["probabilities"]) {
                return level.parse::<i64>().map_or(Value::Null, Value::from);
            }
            match answer["score"].as_f64() {
                Some(s) if s == s.trunc() => Value::from(s as i64),
                _ => Value::Null,
            }
        }
        _ => Value::Null,
    }
}

/// The key with the largest value; the first one on a tie, as Python's `max(probs, key=...)`
/// picks it (`serde_json` keeps the JSON order). `None` for no object, an empty one, or a
/// value that is not a number.
fn argmax_key(probs: &Value) -> Option<&str> {
    let mut best: Option<(&str, f64)> = None;
    for (k, v) in probs.as_object()? {
        let v = v.as_f64()?;
        if best.is_none_or(|(_, b)| v > b) {
            best = Some((k, v));
        }
    }
    best.map(|(k, _)| k)
}

/// Whether two answers agree, as `compare.py` (`agree`) counts it: the same `type` and the
/// same [`categorical`] decision. A score answer without `probabilities` and with a fractional
/// `score` agrees when the two scores are within 0.05.
pub fn agree(a: &Value, r: &Value) -> bool {
    if a["type"].is_null() || a["type"] != r["type"] {
        return false;
    }
    let (ca, cr) = (categorical(a), categorical(r));
    if a["type"] == "score" && ca.is_null() && a["score"].is_number() {
        return match (a["score"].as_f64(), r["score"].as_f64()) {
            (Some(x), Some(y)) => (x - y).abs() <= 0.05,
            _ => false,
        };
    }
    !ca.is_null() && ca == cr
}

/// Largest absolute difference between two answers' reported probabilities: the values of
/// `probabilities` for choice and score answers, `noul` for noul answers, and
/// `action.act_probability` when either answer reports it. The action probability comes from
/// the pooled `[CLS]` output rather than the marker logits, so it can move when no other
/// probability does; the upstream references report it on every answer.
///
/// Both answers must report the same keys with finite numbers, or this is an `Err` naming the
/// first offending key. A missing or non-numeric value is never a difference of zero.
pub fn prob_diff(a: &Value, b: &Value) -> Result<f64, String> {
    let finite = |v: &Value, what: &str| -> Result<f64, String> {
        match v.as_f64() {
            Some(x) if x.is_finite() => Ok(x),
            _ => Err(format!("{what} is {v}, not a finite number")),
        }
    };
    let mut worst = match (a["probabilities"].as_object(), b["probabilities"].as_object()) {
        (Some(pa), Some(pb)) => {
            if let Some(k) = pb.keys().find(|k| !pa.contains_key(*k)) {
                return Err(format!("probabilities.{k} only on one side"));
            }
            let mut worst = 0.0f64;
            for (k, va) in pa {
                let vb = pb.get(k).ok_or_else(|| format!("probabilities.{k} only on one side"))?;
                let d = (finite(va, &format!("probabilities.{k}"))? - finite(vb, &format!("probabilities.{k}"))?).abs();
                worst = worst.max(d);
            }
            worst
        }
        (None, None) => (finite(&a["noul"], "noul")? - finite(&b["noul"], "noul")?).abs(),
        _ => return Err("probabilities only on one side".into()),
    };
    let (act_a, act_b) = (&a["action"]["act_probability"], &b["action"]["act_probability"]);
    if !act_a.is_null() || !act_b.is_null() {
        let d = (finite(act_a, "action.act_probability")? - finite(act_b, "action.act_probability")?).abs();
        worst = worst.max(d);
    }
    Ok(worst)
}
