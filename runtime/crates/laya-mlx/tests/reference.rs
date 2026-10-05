//! The smoke workload (`bench/workloads/smoke.jsonl`) through the engine with the production
//! settings, against the upstream fp32 CPU reference `bench/reference/<name>/smoke.jsonl`, for
//! every checkpoint in the HF cache. This is the check that the f16 GPU path serves upstream's
//! answers: the same decision on every answer and every reported probability within `TOL`,
//! `action.act_probability` included.
//!
//! Ignored by default: it needs the downloaded checkpoints (`source bench/env.sh` first). A
//! checkpoint that is not in the cache is skipped with a note; at least one must run, and with
//! `SYS1_TEST_ALL_CHECKPOINTS=1` all three.
//!
//! Run from the repo root with:
//! `cargo test --manifest-path runtime/Cargo.toml -p laya-mlx --release --test reference -- --ignored --nocapture`

mod common;

use common::{agree, assert_ran, bench_root, categorical, checkpoint_dir, prob_diff, read_jsonl, CHECKPOINTS};
use laya_core::{Agent, BackendOptions};
use serde_json::{json, Value};

/// The settings answers are served with: GELU kept in f16 (upstream promotes it to f32), a
/// 512 MiB MLX buffer cache and 2 GiB wired. The bench's `mlx-fp16-fast` variant.
const PRODUCTION: &str = "f16gelu,cache=512,wired=2048";

/// The f16 GPU path against the fp32 CPU reference. typed-decisions measures 0.0008 on the
/// smoke set, multilingual 0.0102 on one near-tie answer. The parity fixtures use the same 0.02.
const TOL: f64 = 0.02;

/// The bench harness flags a run below this agreement (`bench/harness/compare.py`).
const MIN_AGREEMENT: f64 = 0.99;

#[test]
#[ignore = "needs the checkpoints in the HF cache; source bench/env.sh first"]
fn smoke_matches_upstream_reference() {
    smoke_against_reference(PRODUCTION);
}

/// The same check with boolean masks. The smoke set's 512-token states pad past four window
/// widths, so they take the chunked local-attention path: this covers `mask=bool` on both the
/// dense and the chunked masks, padded query rows (all-false rows would be NaN) included.
#[test]
#[ignore = "needs the checkpoints in the HF cache; source bench/env.sh first"]
fn smoke_matches_upstream_reference_with_bool_masks() {
    smoke_against_reference(&format!("{PRODUCTION},mask=bool"));
}

/// The smoke set with `tuning` against the reference, for every checkpoint in the cache.
fn smoke_against_reference(tuning: &str) {
    let bench = bench_root();
    let smoke = read_jsonl(&bench.join("workloads/smoke.jsonl"));
    let mut ran = 0;
    for (name, repo) in CHECKPOINTS {
        let Some(dir) = checkpoint_dir(name, repo) else {
            continue;
        };
        let reference = read_jsonl(&bench.join(format!("reference/{name}/smoke.jsonl")));
        let opts = BackendOptions { tuning: Some(tuning.into()), ..Default::default() };
        let agent = Agent::load(&dir, &opts, Box::new(laya_mlx::make_backend)).unwrap();
        let (mut worst, mut worst_at, mut agreed, mut total) = (0.0f64, String::new(), 0usize, 0usize);
        for row in &smoke {
            let id = row["id"].as_str().unwrap();
            let want = &reference
                .iter()
                .find(|r| r["id"] == *id)
                .unwrap_or_else(|| panic!("{name}: {id} is not in the reference"))["answers"];
            let got = agent.predict(&row["body"]["state"], &row["body"]["questions"]).unwrap();
            let got = got["answers"].as_object().unwrap();
            assert_eq!(
                got.len(),
                want.as_object().unwrap().len(),
                "{name}/{id}: answered questions differ from the reference"
            );
            for (qid, ans) in got {
                let d = prob_diff(ans, &want[qid]).unwrap_or_else(|e| panic!("{name}/{id}/{qid}: {e}"));
                assert!(d <= TOL, "{name}/{id}/{qid}: probability diff {d} vs reference: {ans} vs {}", want[qid]);
                if d > worst {
                    (worst, worst_at) = (d, format!("{id}/{qid}"));
                }
                total += 1;
                agreed += usize::from(agree(ans, &want[qid]));
            }
        }
        let agreement = agreed as f64 / total as f64;
        eprintln!(
            "{name:<16} [{tuning}] {total} answers over {} requests: agreement {agreed}/{total} ({:.1}%), max probability diff {worst:.4} at {worst_at}",
            smoke.len(),
            100.0 * agreement
        );
        assert!(agreement >= MIN_AGREEMENT, "{name}: agreement {:.1}% is below {:.0}%", 100.0 * agreement, 100.0 * MIN_AGREEMENT);
        ran += 1;
    }
    assert_ran(ran);
}

#[test]
fn prob_diff_is_the_largest_difference_over_the_same_keys() {
    let a = json!({"type": "choice", "probabilities": {"x": 0.7, "y": 0.3}});
    let b = json!({"type": "choice", "probabilities": {"x": 0.65, "y": 0.35}});
    assert!((prob_diff(&a, &b).unwrap() - 0.05).abs() < 1e-12);
    assert_eq!(prob_diff(&a, &a).unwrap(), 0.0);
    let n = json!({"type": "noul", "noul": 0.61});
    let m = json!({"type": "noul", "noul": 0.6});
    assert!((prob_diff(&n, &m).unwrap() - 0.01).abs() < 1e-12);
}

#[test]
fn prob_diff_rejects_missing_or_non_numeric_values() {
    let a = json!({"type": "choice", "probabilities": {"x": 0.7, "y": 0.3}});
    // A key missing on either side, or an extra one, is an error, not a zero.
    assert!(prob_diff(&a, &json!({"probabilities": {"x": 0.7}})).is_err());
    assert!(prob_diff(&a, &json!({"probabilities": {"x": 0.7, "y": 0.3, "z": 0.0}})).is_err());
    assert!(prob_diff(&a, &json!({"probabilities": {"x": 0.7, "y": null}})).is_err());
    assert!(prob_diff(&a, &json!({"probabilities": {"x": 0.7, "y": "0.3"}})).is_err());
    // Probabilities on one side only, or a noul without its number.
    assert!(prob_diff(&a, &json!({"type": "noul", "noul": 0.5})).is_err());
    assert!(prob_diff(&json!({"noul": 0.5}), &json!({})).is_err());
    assert!(prob_diff(&json!({"noul": f64::NAN}), &json!({"noul": 0.5})).is_err());
}

/// `action.act_probability` counts like any other probability once either side reports it: it
/// comes from the pooled output, so it can move while every marker probability stays put.
#[test]
fn prob_diff_includes_the_action_probability() {
    let a = json!({"type": "choice", "probabilities": {"x": 0.7, "y": 0.3}, "action": {"act_probability": 0.9}});
    let mut b = a.clone();
    b["action"]["act_probability"] = json!(0.85);
    assert!((prob_diff(&a, &b).unwrap() - 0.05).abs() < 1e-12);
    assert_eq!(prob_diff(&a, &a).unwrap(), 0.0);
    // The larger of the two differences is reported.
    b["probabilities"]["x"] = json!(0.6);
    b["probabilities"]["y"] = json!(0.4);
    assert!((prob_diff(&a, &b).unwrap() - 0.1).abs() < 1e-12);
    let n = json!({"type": "noul", "noul": 0.6, "action": {"act_probability": 0.2}});
    let mut m = n.clone();
    m["action"]["act_probability"] = json!(0.25);
    assert!((prob_diff(&n, &m).unwrap() - 0.05).abs() < 1e-12);
    // On one side only, not a number, or not finite: an error, never a difference of zero.
    let mut without = a.clone();
    without.as_object_mut().unwrap().remove("action");
    assert!(prob_diff(&a, &without).unwrap_err().contains("act_probability"));
    assert!(prob_diff(&without, &a).unwrap_err().contains("act_probability"));
    b["action"]["act_probability"] = json!("0.9");
    assert!(prob_diff(&a, &b).is_err());
    b["action"]["act_probability"] = json!(f64::NAN);
    assert!(prob_diff(&a, &b).is_err());
    b["action"] = json!({});
    assert!(prob_diff(&a, &b).is_err());
}

/// The decision of each answer type is the one `bench/harness/compare.py` counts: the choice
/// label, the noul side, and for a score the most probable level, not the rounded expected score.
#[test]
fn categorical_is_the_decision_of_each_answer_type() {
    assert_eq!(categorical(&json!({"type": "choice", "choice": "refund"})), json!("refund"));
    // Without `choice`, the most probable option; the first one on a tie.
    assert_eq!(categorical(&json!({"type": "choice", "probabilities": {"a": 0.3, "b": 0.4, "c": 0.3}})), json!("b"));
    assert_eq!(categorical(&json!({"type": "choice", "probabilities": {"a": 0.5, "b": 0.5}})), json!("a"));
    assert_eq!(categorical(&json!({"type": "noul", "noul": 0.51})), json!(true));
    assert_eq!(categorical(&json!({"type": "noul", "noul": 0.49})), json!(false));
    assert_eq!(categorical(&json!({"type": "noul"})), Value::Null);
    // A score decides its most probable level: expected 2.4 rounds to 2, but level 3 is the
    // most probable here, and a shift of probability can flip the level without moving the
    // rounded score.
    let score = json!({"type": "score", "score": 2.4, "probabilities": {"0": 0.1, "1": 0.1, "2": 0.35, "3": 0.45}});
    assert_eq!(categorical(&score), json!(3));
    assert_eq!(categorical(&json!({"type": "score", "score": 2.6, "probabilities": {"0": 0.0, "1": 0.0, "2": 0.6, "3": 0.4}})), json!(2));
    // Without probabilities an integer score is the level and a fractional one decides nothing.
    assert_eq!(categorical(&json!({"type": "score", "score": 2.0})), json!(2));
    assert_eq!(categorical(&json!({"type": "score", "score": 2.4})), Value::Null);
    assert_eq!(categorical(&json!({"type": "other"})), Value::Null);
}

/// `agree` is compare.py's rule: same type, same decision, and nothing agrees with a null
/// decision.
#[test]
fn agree_needs_the_same_type_and_a_decision() {
    let choice = json!({"type": "choice", "choice": "refund", "probabilities": {"refund": 0.6, "deny": 0.4}});
    assert!(agree(&choice, &json!({"type": "choice", "probabilities": {"refund": 0.51, "deny": 0.49}})));
    assert!(!agree(&choice, &json!({"type": "choice", "choice": "deny"})));
    let level1 = json!({"type": "score", "score": 1.0, "probabilities": {"0": 0.2, "1": 0.8}});
    let yes = json!({"type": "noul", "noul": 1.0});
    assert!(!agree(&level1, &yes), "score level 1 is not noul true");
    assert!(agree(&level1, &json!({"type": "score", "score": 1.2, "probabilities": {"0": 0.45, "1": 0.55}})));
    assert!(!agree(&level1, &json!({"type": "score", "score": 1.0, "probabilities": {"0": 0.55, "1": 0.45}})));
    assert!(!agree(&json!({"type": "noul"}), &json!({"type": "noul"})), "two null decisions do not agree");
    assert!(!agree(&json!({}), &json!({})));
    // A fractional score without probabilities is compared as a value within 0.05.
    assert!(agree(&json!({"type": "score", "score": 2.4}), &json!({"type": "score", "score": 2.44})));
    assert!(!agree(&json!({"type": "score", "score": 2.4}), &json!({"type": "score", "score": 2.5})));
    assert!(!agree(&json!({"type": "score", "score": 2.4}), &json!({"type": "score"})));
}
