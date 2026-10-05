//! The real predictor: a `laya_core::Agent` on the MLX backend, plus the warm-up requests
//! run before the server binds (one question, then four) so the first client request does
//! not pay for the first forward's kernel compilation and buffer allocation.

use crate::worker::{Factory, Predictor};
use laya_core::{Agent, BackendOptions};
use serde_json::{json, Value};
use std::path::PathBuf;

pub struct AgentPredictor {
    agent: Agent,
}

impl AgentPredictor {
    /// A factory that loads the checkpoint on the inference thread.
    pub fn factory(dir: PathBuf, opts: BackendOptions) -> Factory {
        Box::new(move || {
            let agent = Agent::load(&dir, &opts, Box::new(laya_mlx::make_backend))?;
            Ok(Box::new(AgentPredictor { agent }) as Box<dyn Predictor>)
        })
    }
}

impl Predictor for AgentPredictor {
    fn predict(&self, state: &Value, questions: &Value) -> laya_core::Result<Value> {
        self.agent.predict(state, questions)
    }

    fn warmup(&self) -> laya_core::Result<()> {
        run_warmup(|state, questions| self.agent.predict(state, questions)).map(|_| ())
    }

    fn engine(&self) -> String {
        self.agent.backend_name()
    }
}

/// Run the warm-up sets through `predict` and return how many ran. Upstream `laya serve`
/// has no warm-up at all, so a set must not keep a checkpoint that can serve client
/// requests from starting. The one-question set is about as small as a real request: a
/// question error skips it with a log line (a local checkpoint with a short `head_max_len`
/// answers `options exceed head_max_len`), any other error is a load or backend failure
/// and stops startup. The four-question set is optional: any error, out of memory
/// included, is logged and skipped, since a machine that cannot fit four questions can
/// still serve one. A skip line quotes the error and does not guess at its cause.
pub fn run_warmup(
    mut predict: impl FnMut(&Value, &Value) -> laya_core::Result<Value>,
) -> laya_core::Result<usize> {
    let state = warmup_state();
    let mut ran = 0;
    for (name, questions, optional) in [
        ("one question", warmup_questions_1(), false),
        ("four questions", warmup_questions_4(), true),
    ] {
        match predict(&state, &questions) {
            Ok(_) => ran += 1,
            Err(e @ laya_core::Error::Question(_)) => {
                crate::log(format!("warm-up set ({name}) skipped: {e}"))
            }
            Err(e) if optional => crate::log(format!(
                "warm-up set ({name}) skipped, startup continues: {e}"
            )),
            Err(e) => return Err(e),
        }
    }
    Ok(ran)
}

/// A short customer-service state in the shape of the bench workloads.
pub fn warmup_state() -> Value {
    json!("{\"account\":{\"tier\":\"standard\",\"tenure_months\":3},\"thread\":[{\"role\":\"customer\",\"text\":\"My payment did not go through and now I am locked out of my account.\"}]}")
}

pub fn warmup_questions_1() -> Value {
    json!({
        "action": {
            "type": "choice",
            "instructions": "What should the assistant do next with this conversation?",
            "criteria": {
                "answer_directly": "The assistant can resolve this itself.",
                "escalate_to_human": "Hand off to a human agent.",
                "request_information": "More detail is needed from the customer."
            }
        }
    })
}

pub fn warmup_questions_4() -> Value {
    json!({
        "category": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "billing and refunds", "account": "login and access", "technical": "bugs and outages"}
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["calm", "firm", "angry", "furious"]
        },
        "refund": {
            "type": "noul",
            "instructions": "Is the customer asking for money back?",
            "criteria": {"true": "a refund or credit is requested", "false": "no money is asked for"}
        },
        "action": warmup_questions_1()["action"]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use laya_core::Error;

    /// A checkpoint whose budget fits one question but not four starts with one set run; one
    /// that fits neither starts with none.
    #[test]
    fn warmup_skips_sets_the_checkpoint_cannot_fit() {
        let ran = run_warmup(|_, questions| {
            if questions.as_object().unwrap().len() > 1 {
                Err(Error::Question(
                    "question 'urgency' options exceed head_max_len=24".into(),
                ))
            } else {
                Ok(json!({"answers": {}}))
            }
        })
        .unwrap();
        assert_eq!(ran, 1);
        let ran =
            run_warmup(|_, _| Err(Error::Question("question 'action': too big".into()))).unwrap();
        assert_eq!(ran, 0);
    }

    /// A backend failure on the four-question set is skipped, so a machine that runs out of
    /// memory on four questions still starts and serves smaller requests.
    #[test]
    fn warmup_skips_a_backend_failure_on_the_optional_set() {
        let ran = run_warmup(|_, questions| {
            if questions.as_object().unwrap().len() > 1 {
                Err(Error::Backend("metal: out of memory".into()))
            } else {
                Ok(json!({"answers": {}}))
            }
        })
        .unwrap();
        assert_eq!(ran, 1);
    }

    /// Anything but a question error on the one-question set is a load or backend failure
    /// and still stops startup, whatever the four-question set would have done.
    #[test]
    fn warmup_backend_failure_on_the_first_set_stops_startup() {
        let e = run_warmup(|_, _| Err(Error::Backend("metal: out of memory".into()))).unwrap_err();
        assert!(matches!(e, Error::Backend(_)), "{e}");
        let e = run_warmup(|_, _| Err(Error::Weights("missing tensor".into()))).unwrap_err();
        assert!(matches!(e, Error::Weights(_)), "{e}");
    }

    /// The warm-up sets are valid requests: both pass the server's own checks.
    #[test]
    fn warmup_requests_pass_validation() {
        for questions in [warmup_questions_1(), warmup_questions_4()] {
            let body = json!({"state": warmup_state(), "questions": questions});
            crate::validate::validate_body(body.to_string().as_bytes(), "typed-decisions").unwrap();
        }
    }
}
