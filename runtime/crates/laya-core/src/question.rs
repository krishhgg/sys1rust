//! Question schema validation, normalization and option rendering (`Agent._check_question`,
//! `Agent._to_internal`, `common.render_options`).

use crate::pyjson;
use crate::{Error, Result};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QType {
    Choice,
    Score,
    Noul,
}

impl QType {
    pub fn index(self) -> usize {
        match self {
            QType::Choice => 0,
            QType::Score => 1,
            QType::Noul => 2,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }
    pub fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(QType::Choice),
            1 => Some(QType::Score),
            2 => Some(QType::Noul),
            _ => None,
        }
    }
}

/// A validated, normalized question with its rendered option texts.
#[derive(Debug, Clone)]
pub struct Question {
    pub id: String,
    pub qtype: QType,
    /// Instruction text as the model sees it (non-string instructions are `json.dumps`ed).
    pub instructions: String,
    /// Option texts in label-index order; for noul always `[false, true]`.
    pub options: Vec<String>,
    /// Choice labels in option order as the caller wrote them: dict-form keys are strings,
    /// list-form items keep their JSON type (`1`, `true`, `1.5`). Upstream's `_to_internal`
    /// makes these the dict keys, so the answer's `choice` is the raw label and its
    /// `probabilities` key is the label's [`pyjson::dumps_key`]. Empty for other types.
    pub choice_labels: Vec<Value>,
    /// Score level descriptions as given (raw JSON values), for the `legend` output.
    pub score_legend: Vec<Value>,
}

fn is_blank(v: Option<&Value>) -> bool {
    matches!(v, None | Some(Value::Null)) || matches!(v, Some(Value::String(s)) if s.is_empty())
}

/// `render_criterion`: strings pass through, anything else is compact-ish JSON.
pub fn render_criterion(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => pyjson::dumps(other),
    }
}

fn resolve_noul_labels(labels: Option<&Value>) -> Result<(String, String)> {
    let bad = || {
        Error::Question(
            "noul labels must map exactly 'false' and 'true' to distinct non-empty strings".into(),
        )
    };
    let Some(labels) = labels else {
        return Ok(("false".into(), "true".into()));
    };
    let Value::Object(o) = labels else {
        return Err(bad());
    };
    if o.len() != 2 || !o.contains_key("false") || !o.contains_key("true") {
        return Err(bad());
    }
    let (Some(Value::String(f)), Some(Value::String(t))) = (o.get("false"), o.get("true")) else {
        return Err(bad());
    };
    let (f, t) = (f.trim().to_string(), t.trim().to_string());
    if f.is_empty() || t.is_empty() || f == t {
        return Err(bad());
    }
    Ok((f, t))
}

/// `Agent._check_question`, rule for rule and byte for byte in its messages, so a client sees
/// the same text from sys1rust as from upstream's server. Every `%r` is Python's `repr`.
fn check(qid: &str, qdef: &Value) -> Result<()> {
    let q = |msg: String| Error::Question(format!("question {}: {msg}", pyjson::repr_str(qid)));
    let Value::Object(o) = qdef else {
        return Err(q(format!(
            "definition must be a dict, got {}",
            pyjson::type_name(qdef)
        )));
    };
    let t = match o.get("type").and_then(Value::as_str) {
        Some("choice") => QType::Choice,
        Some("score") => QType::Score,
        Some("noul") => QType::Noul,
        // Upstream raises TypeError for a list or dict here (unhashable in `t not in QTYPES`);
        // naming the question is the better answer for the same input.
        _ => {
            return Err(q(format!(
                "unknown type {}; use one of ['choice', 'noul', 'score']",
                pyjson::repr(o.get("type").unwrap_or(&Value::Null))
            )))
        }
    };
    if !o.contains_key("instructions") {
        return Err(q(
            "no 'instructions'; add the text the model should answer".into()
        ));
    }
    let crit = o.get("criteria");
    match t {
        QType::Choice => match crit {
            Some(Value::Object(m)) if m.is_empty() => {
                return Err(q("a choice question needs at least one criterion".into()))
            }
            Some(Value::Array(a)) if a.is_empty() => {
                return Err(q("a choice question needs at least one criterion".into()))
            }
            // Dict keys are strings in JSON, so the label rules below cannot fire for them.
            Some(Value::Object(_)) => {}
            Some(Value::Array(a)) => {
                // A label is rendered as option text and becomes the answer key, so it must
                // be a scalar and not null.
                for (i, label) in a.iter().enumerate() {
                    match label {
                        Value::Array(_) | Value::Object(_) => {
                            return Err(q(format!(
                                "choice label {i} is a {}; a label is rendered as option text and used as the answer key, so it must be a scalar (a string, number or bool), got {}",
                                pyjson::type_name(label),
                                pyjson::repr(label)
                            )))
                        }
                        Value::Null => {
                            return Err(q(format!(
                                "choice label {i} is null; a label is rendered as option text and used as the answer key, so it must be a string, number or bool -- a null label renders as the text \"None\" while its answer key is \"null\""
                            )))
                        }
                        _ => {}
                    }
                }
                // `_to_internal` turns the list into `{label: None}`, so two labels that are
                // one Python dict key (1, 1.0 and True; 0, 0.0 and False) would silently
                // drop an option.
                for (i, label) in a.iter().enumerate() {
                    if let Some(first) = a[..i].iter().position(|l| pyjson::py_eq(l, label)) {
                        return Err(q(format!(
                            "choice label {i} ({}) repeats label {first}; the labels are the answer keys, so every option needs its own (1, 1.0 and True are one key)",
                            pyjson::repr(label)
                        )));
                    }
                }
            }
            _ => return Err(q("a choice question takes 'criteria' as a dict of label -> description, or a list of labels".into())),
        },
        QType::Score => match crit {
            Some(Value::Array(a)) if a.is_empty() => {
                return Err(q("a score question needs at least one level".into()))
            }
            Some(Value::Array(a)) => {
                if let Some(i) = a.iter().position(Value::is_null) {
                    return Err(q(format!(
                        "score level {i} is null; give every level a description, index 0 first"
                    )));
                }
            }
            _ => return Err(q("a score question takes 'criteria' as a list of level descriptions, index 0 first".into())),
        },
        QType::Noul => match crit {
            None | Some(Value::Null) => {}
            Some(Value::Object(m)) => {
                // `render_options` reads only `crit.get("false")` and `crit.get("true")`
                // after `str(k).lower()`, so any other key would be dropped without a word.
                let mut keys: Vec<Value> = m.keys().map(|k| Value::from(k.to_lowercase())).collect();
                keys.sort_by(|a, b| a.as_str().cmp(&b.as_str()));
                keys.dedup();
                if keys.iter().any(|k| !matches!(k.as_str(), Some("true" | "false"))) {
                    return Err(q(format!(
                        "a noul question takes 'criteria' keyed only 'true'/'false' (either or both, and omitted is fine), got {}. Those keys are the option texts the model reads; any other key was silently dropped and replaced with the defaults. If you want the answer worded differently, keep 'criteria' keyed 'true'/'false' and set 'labels' instead.",
                        pyjson::repr(&Value::Array(keys))
                    )));
                }
            }
            _ => return Err(q("a noul question takes 'criteria' as a dict with optional 'true'/'false' descriptions, or omits it".into())),
        },
    }
    if let Some(labels) = o.get("labels") {
        if t != QType::Noul {
            return Err(q("'labels' is only supported for noul questions".into()));
        }
        resolve_noul_labels(Some(labels)).map_err(|e| q(e.to_string()))?;
    }
    Ok(())
}

fn to_internal(qid: &str, qdef: &Map<String, Value>) -> Result<Question> {
    let qtype = match qdef.get("type").and_then(Value::as_str) {
        Some("choice") => QType::Choice,
        Some("score") => QType::Score,
        _ => QType::Noul,
    };
    let instructions = match &qdef["instructions"] {
        Value::String(s) => s.clone(),
        other => pyjson::dumps(other),
    };
    let crit = qdef.get("criteria");
    let mut options = Vec::new();
    let mut choice_labels = Vec::new();
    let mut score_legend = Vec::new();
    match qtype {
        QType::Choice => {
            // dict: label -> description; list: labels only, kept as the raw JSON value
            // (`check` has rejected labels that are one Python dict key). `render_options`
            // writes every label with `str(label)`, with or without a description.
            let mut entries: Vec<(Value, Value)> = Vec::new();
            match crit {
                Some(Value::Object(m)) => entries.extend(
                    m.iter()
                        .map(|(k, v)| (Value::String(k.clone()), v.clone())),
                ),
                Some(Value::Array(a)) => {
                    entries.extend(a.iter().map(|item| (item.clone(), Value::Null)))
                }
                _ => unreachable!("validated"),
            }
            for (label, v) in entries {
                let text = pyjson::py_str(&label);
                options.push(if is_blank(Some(&v)) {
                    text
                } else {
                    format!("{}: {}", text, render_criterion(&v))
                });
                choice_labels.push(label);
            }
        }
        QType::Score => {
            let Some(Value::Array(a)) = crit else {
                unreachable!("validated")
            };
            for (i, c) in a.iter().enumerate() {
                options.push(format!("level {}: {}", i, render_criterion(c)));
                score_legend.push(c.clone());
            }
        }
        QType::Noul => {
            let (false_label, true_label) = resolve_noul_labels(qdef.get("labels"))
                .map_err(|e| Error::Question(format!("question {}: {e}", pyjson::repr_str(qid))))?;
            // keys are normalized with str(k).lower()
            let mut norm: Map<String, Value> = Map::new();
            if let Some(Value::Object(m)) = crit {
                for (k, v) in m {
                    norm.insert(k.to_lowercase(), v.clone());
                }
            }
            let fc = norm.get("false");
            let tc = norm.get("true");
            options.push(format!(
                "{}: {}",
                false_label,
                if is_blank(fc) {
                    "no, the statement does not hold".to_string()
                } else {
                    render_criterion(fc.unwrap())
                }
            ));
            options.push(format!(
                "{}: {}",
                true_label,
                if is_blank(tc) {
                    "yes, the statement holds".to_string()
                } else {
                    render_criterion(tc.unwrap())
                }
            ));
        }
    }
    Ok(Question {
        id: qid.to_string(),
        qtype,
        instructions,
        options,
        choice_labels,
        score_legend,
    })
}

/// Validate and normalize a `{qid: definition}` object, preserving order.
pub fn parse_questions(questions: &Value) -> Result<Vec<Question>> {
    let Value::Object(map) = questions else {
        return Err(Error::Question(
            "questions must be a JSON object mapping question id -> definition".into(),
        ));
    };
    let mut out = Vec::with_capacity(map.len());
    for (qid, qdef) in map {
        check(qid, qdef)?;
        let Value::Object(o) = qdef else {
            unreachable!("validated")
        };
        out.push(to_internal(qid, o)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn renders_like_python() {
        let qs = parse_questions(&json!({
            "d": {"type": "choice", "instructions": "Which?", "criteria": {"billing": "invoices", "other": null, "zero": 0, "e": ""}},
            "u": {"type": "score", "instructions": "How?", "criteria": ["low", {"desc": "x"}]},
            "n": {"type": "noul", "instructions": {"a": 1}},
            "l": {"type": "noul", "instructions": "Money?", "criteria": {"True": "yes money", "FALSE": ""}, "labels": {"false": " B ", "true": "A"}},
            "c": {"type": "choice", "instructions": "x", "criteria": ["a", 7, 2.5, true, "it's"]},
        })).unwrap();
        assert_eq!(
            qs[0].options,
            ["billing: invoices", "other", "zero: 0", "e"]
        );
        assert_eq!(
            qs[1].options,
            ["level 0: low", "level 1: {\"desc\": \"x\"}"]
        );
        assert_eq!(qs[2].instructions, "{\"a\": 1}");
        assert_eq!(
            qs[2].options,
            [
                "false: no, the statement does not hold",
                "true: yes, the statement holds"
            ]
        );
        assert_eq!(
            qs[3].options,
            ["B: no, the statement does not hold", "A: yes money"]
        );
        // List labels render with `str(label)`: a number or bool as Python prints it.
        assert_eq!(qs[4].options, ["a", "7", "2.5", "True", "it's"]);
        assert_eq!(
            qs[4].choice_labels,
            [json!("a"), json!(7), json!(2.5), json!(true), json!("it's")]
        );
        // Dict-form labels are strings either way.
        assert_eq!(
            qs[0].choice_labels,
            [json!("billing"), json!("other"), json!("zero"), json!("e")]
        );
        assert!(qs[1].choice_labels.is_empty() && qs[2].choice_labels.is_empty());
    }

    /// A list-form label keeps its JSON type: the option text is `str(label)`, the answer's
    /// `choice` is the raw label and the `probabilities` key is what `json.dumps` writes for
    /// it as a dict key.
    #[test]
    fn list_labels_keep_their_json_type() {
        let qs = parse_questions(&json!({
            "q": {"type": "choice", "instructions": "x", "criteria": [2, "b", true, 1.5]},
        }))
        .unwrap();
        assert_eq!(qs[0].options, ["2", "b", "True", "1.5"]);
        assert_eq!(qs[0].choice_labels, [json!(2), json!("b"), json!(true), json!(1.5)]);
        assert!(qs[0].choice_labels[0].is_i64());
        assert!(qs[0].choice_labels[1].is_string());
        assert!(qs[0].choice_labels[2].is_boolean());
        assert!(qs[0].choice_labels[3].is_f64());
        let keys: Vec<String> = qs[0].choice_labels.iter().map(pyjson::dumps_key).collect();
        assert_eq!(keys, ["2", "b", "true", "1.5"]);
        // `1` and `True` are one Python dict key, so upstream refuses that list.
        assert_eq!(
            err("q", json!({"type": "choice", "instructions": "x", "criteria": [1, "b", true, 1.5]})),
            "question 'q': choice label 2 (True) repeats label 0; the labels are the answer keys, so every option needs its own (1, 1.0 and True are one key)"
        );
    }

    /// Two integer labels beyond `u64::MAX` that differ in the last digit are two Python dict
    /// keys, so they are accepted, and each renders with its exact digits: as option text
    /// (`str(label)`), as the raw label and as the `probabilities` key. Both would round to
    /// one f64; `arbitrary_precision` keeps the literals apart.
    #[test]
    fn adjacent_big_integer_labels_are_distinct() {
        let qs = parse_questions(
            &serde_json::from_str(
                r#"{"q": {"type": "choice", "instructions": "x",
                         "criteria": [18446744073709551616, 18446744073709551617]}}"#,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            qs[0].options,
            ["18446744073709551616", "18446744073709551617"]
        );
        let printed: Vec<String> = qs[0]
            .choice_labels
            .iter()
            .map(|l| serde_json::to_string(l).unwrap())
            .collect();
        assert_eq!(printed, ["18446744073709551616", "18446744073709551617"]);
        let keys: Vec<String> = qs[0].choice_labels.iter().map(pyjson::dumps_key).collect();
        assert_eq!(keys, ["18446744073709551616", "18446744073709551617"]);
        // The same digits twice is still a repeat.
        assert_eq!(
            err(
                "q",
                serde_json::from_str(
                    r#"{"type": "choice", "instructions": "x",
                        "criteria": [18446744073709551617, "b", 18446744073709551617]}"#
                )
                .unwrap()
            ),
            "question 'q': choice label 2 (18446744073709551617) repeats label 0; the labels are the answer keys, so every option needs its own (1, 1.0 and True are one key)"
        );
    }

    fn err(qid: &str, qdef: Value) -> String {
        parse_questions(&json!({ qid: qdef }))
            .unwrap_err()
            .to_string()
    }

    /// Every expected string is the `ValueError` text of upstream's `Agent._check_question`
    /// for the same input under Python 3.14.
    #[test]
    fn messages_match_upstream_check_question() {
        assert_eq!(
            err("q", json!("nope")),
            "question 'q': definition must be a dict, got str"
        );
        assert_eq!(
            err("q", json!([1])),
            "question 'q': definition must be a dict, got list"
        );
        assert_eq!(
            err("q", json!(1.5)),
            "question 'q': definition must be a dict, got float"
        );
        assert_eq!(
            err("q", Value::Null),
            "question 'q': definition must be a dict, got NoneType"
        );
        assert_eq!(
            err("q", json!(true)),
            "question 'q': definition must be a dict, got bool"
        );
        assert_eq!(
            err("q", json!({"type": "bool", "instructions": "x"})),
            "question 'q': unknown type 'bool'; use one of ['choice', 'noul', 'score']"
        );
        assert_eq!(
            err("q", json!({"instructions": "x"})),
            "question 'q': unknown type None; use one of ['choice', 'noul', 'score']"
        );
        assert_eq!(
            err("q", json!({"type": 1.0, "instructions": "x"})),
            "question 'q': unknown type 1.0; use one of ['choice', 'noul', 'score']"
        );
        assert_eq!(
            err("it's", json!({"type": "choice"})),
            "question \"it's\": no 'instructions'; add the text the model should answer"
        );
        let choice = |crit: Value| json!({"type": "choice", "instructions": "x", "criteria": crit});
        assert_eq!(
            err("q", choice(json!("abc"))),
            "question 'q': a choice question takes 'criteria' as a dict of label -> description, or a list of labels"
        );
        assert_eq!(
            err("q", choice(json!([]))),
            "question 'q': a choice question needs at least one criterion"
        );
        assert_eq!(
            err("q", choice(json!({}))),
            "question 'q': a choice question needs at least one criterion"
        );
        assert_eq!(
            err("q", choice(json!(["A", ["B"]]))),
            "question 'q': choice label 1 is a list; a label is rendered as option text and used as the answer key, so it must be a scalar (a string, number or bool), got ['B']"
        );
        assert_eq!(
            err("q", choice(json!(["A", {"k": [1, 2.5, null, true, "s"]}]))),
            "question 'q': choice label 1 is a dict; a label is rendered as option text and used as the answer key, so it must be a scalar (a string, number or bool), got {'k': [1, 2.5, None, True, 's']}"
        );
        assert_eq!(
            err("q", choice(json!(["A", null]))),
            "question 'q': choice label 1 is null; a label is rendered as option text and used as the answer key, so it must be a string, number or bool -- a null label renders as the text \"None\" while its answer key is \"null\""
        );
        let repeats = |i: usize, label: &str, first: usize| {
            format!("question 'q': choice label {i} ({label}) repeats label {first}; the labels are the answer keys, so every option needs its own (1, 1.0 and True are one key)")
        };
        assert_eq!(
            err("q", choice(json!(["a", "b", "a"]))),
            repeats(2, "'a'", 0)
        );
        assert_eq!(err("q", choice(json!([1, 2, 1.0]))), repeats(2, "1.0", 0));
        assert_eq!(err("q", choice(json!([true, 1]))), repeats(1, "1", 0));
        assert_eq!(err("q", choice(json!([0, false]))), repeats(1, "False", 0));
        assert_eq!(
            err("q", choice(json!([1e16, 10000000000000000u64]))),
            repeats(1, "10000000000000000", 0)
        );
        assert_eq!(
            err("q", choice(json!(["a", "b", "c", "b"]))),
            repeats(3, "'b'", 1)
        );
        assert_eq!(
            err("q", choice(json!(["a", "a", "a"]))),
            repeats(1, "'a'", 0)
        );
        // Distinct under Python `==`: accepted.
        assert!(parse_questions(&json!({"q": choice(json!(["1", 1]))})).is_ok());
        assert!(parse_questions(
            &json!({"q": choice(json!([9007199254740993u64, 9007199254740992.0]))})
        )
        .is_ok());

        let score = |crit: Value| json!({"type": "score", "instructions": "x", "criteria": crit});
        assert_eq!(
            err("q", score(json!({"a": 1}))),
            "question 'q': a score question takes 'criteria' as a list of level descriptions, index 0 first"
        );
        assert_eq!(
            err("q", score(json!([]))),
            "question 'q': a score question needs at least one level"
        );
        assert_eq!(
            err("q", score(json!(["low", null, null]))),
            "question 'q': score level 1 is null; give every level a description, index 0 first"
        );

        let noul = |crit: Value| json!({"type": "noul", "instructions": "x", "criteria": crit});
        assert_eq!(
            err("q", noul(json!(["a"]))),
            "question 'q': a noul question takes 'criteria' as a dict with optional 'true'/'false' descriptions, or omits it"
        );
        assert_eq!(
            err("q", noul(json!({"maybe": "m", "True": "t"}))),
            "question 'q': a noul question takes 'criteria' keyed only 'true'/'false' (either or both, and omitted is fine), got ['maybe', 'true']. Those keys are the option texts the model reads; any other key was silently dropped and replaced with the defaults. If you want the answer worded differently, keep 'criteria' keyed 'true'/'false' and set 'labels' instead."
        );
        // Keys that differ only in case are one key after `str(k).lower()`.
        assert!(parse_questions(&json!({"q": noul(json!({"True": "t", "TRUE": "u"}))})).is_ok());

        assert_eq!(
            err(
                "q",
                json!({"type": "choice", "instructions": "x", "criteria": ["a"], "labels": {"false": "a", "true": "b"}})
            ),
            "question 'q': 'labels' is only supported for noul questions"
        );
        assert_eq!(
            err("q", json!({"type": "noul", "instructions": "x", "labels": {"false": "a", "true": "a"}})),
            "question 'q': noul labels must map exactly 'false' and 'true' to distinct non-empty strings"
        );
    }

    /// The question id goes through Python's `repr`, escapes included.
    #[test]
    fn question_id_is_python_repr() {
        assert_eq!(
            err(
                "zero\u{200b}width",
                json!({"type": "choice", "instructions": "x", "criteria": []})
            ),
            "question 'zero\\u200bwidth': a choice question needs at least one criterion"
        );
        assert_eq!(
            err("tab\tq", json!({"type": "choice", "instructions": "x", "criteria": [null]})),
            "question 'tab\\tq': choice label 0 is null; a label is rendered as option text and used as the answer key, so it must be a string, number or bool -- a null label renders as the text \"None\" while its answer key is \"null\""
        );
    }

    #[test]
    fn rejects_malformed() {
        assert!(parse_questions(&json!({"q": {"type": "bool", "instructions": "x"}})).is_err());
        assert!(parse_questions(
            &json!({"q": {"type": "choice", "instructions": "x", "criteria": {}}})
        )
        .is_err());
        assert!(parse_questions(
            &json!({"q": {"type": "score", "instructions": "x", "criteria": {"a": 1}}})
        )
        .is_err());
        assert!(parse_questions(&json!({"q": {"type": "choice", "instructions": "x", "criteria": ["a"], "labels": {"false": "a", "true": "b"}}})).is_err());
        assert!(parse_questions(&json!({"q": {"type": "noul", "instructions": "x", "labels": {"false": "a", "true": "a"}}})).is_err());
        assert!(parse_questions(&json!({"q": {"type": "noul"}})).is_err());
        assert!(parse_questions(&json!({"q": "nope"})).is_err());
    }
}
