---
name: sys1rust-use
description: Call a running sys1rust server from code, a script or another agent through its /v1/systemone API, read its choice, score and noul answers and their confidences, handle its errors, and choose the Laya model, port, API key or offline mode it runs with. Use when writing a client for sys1rust or laya serve, sending System 1 questions about a text, or changing how sys1rust serves.
---

# Use sys1rust

sys1rust answers questions about a text, the state, with one of Laya's System 1 models on the Mac's GPU. It speaks the `/v1/systemone` API of upstream `laya serve`, so a client written for one works with the other. Check that a server runs before you call it:

```sh
curl -s 127.0.0.1:8000/health
```

A running server answers 200 with `"status":"ok"` and names its model in `loaded`. If nothing answers, follow the setup skill, [`.agents/skills/sys1rust-setup/SKILL.md`](../sys1rust-setup/SKILL.md).

## Endpoints

- `GET /health` answers 200 with `{"status":"ok","loaded":["typed-decisions"],"revisions":{...},"device":"gpu","engine":"mlx(gpu,f16)","tuning":"...","worker":"running"}`. It answers 503 with `"status":"error"` when the inference thread has stopped.
- `POST /v1/systemone` takes one state and its questions and returns one answer per question.

## Request

```json
{
  "state": "Customer: I was charged twice for my subscription this month and I need it fixed today.",
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "payments, charges and refunds", "tech": "bugs and outages", "sales": "new purchases"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["none", "low", "high", "now"]},
    "money": {"type": "noul", "instructions": "Is money involved?"}
  }
}
```

- `state` is required. It is the text to judge, as a string or any other JSON value, up to 50,000 characters. A value that isn't a string counts as the length of Python's `str()` of it.
- `questions` is required. It maps each question id to a question, up to 64 of them. Every question needs `type` and `instructions`:
  - `choice` takes `criteria` as an object of label to description, or as a list of labels, up to 100 options.
  - `score` takes `criteria` as a list of level descriptions, level 0 first, up to 32 levels.
  - `noul` is a yes or no question. Its `criteria` is optional, an object with a `"true"` description, a `"false"` one or both.
  - All the questions together take at most 512 options.
- `model` is optional. Leave it out. A name of a Laya model other than the one the server runs gets 400.
- `max_len` and `head_max_len` get 422, because sys1rust doesn't support them.
- The body must be standard JSON in UTF-8, at most 2,097,152 bytes.

Put every question about one state into one request. They run through the model together, one row per question, which costs less than separate requests.

## Response

The request above returns:

```json
{
  "model": "laya-rl-agent",
  "answers": {
    "team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.7941, "tech": 0.1012, "sales": 0.1047}, "confidence": 0.4073, "answer_confidence": 0.7941, "action": {"act_probability": 1.0}},
    "urgency": {"type": "score", "score": 2.5407, "legend": {"0": "none", "1": "low", "2": "high", "3": "now"}, "probabilities": {"0": 0.0101, "1": 0.0452, "2": 0.3384, "3": 0.6062}, "confidence": 0.3821, "answer_confidence": 0.6062, "action": {"act_probability": 1.0}},
    "money": {"type": "noul", "noul": 0.7302, "confidence": 0.7302, "answer_confidence": 0.7302, "action": {"act_probability": 1.0}}
  },
  "usage": {"input_tokens": 150, "output_tokens": 0},
  "routing": {"model": "typed-decisions", "repo": "convaiinnovations/laya-typed-decisions", "reason": "only checkpoint served"}
}
```

`answers` has one entry per question id, in request order. Each entry has `type`, the answer, `confidence`, `answer_confidence` and `action`.

| type | answer fields | the decision |
| --- | --- | --- |
| `choice` | `choice`, the most likely label, and `probabilities` for every label | `choice` |
| `score` | `score`, the expected level, which can fall between levels; `probabilities` per level, keyed `"0"`, `"1"` and so on; `legend`, each level's description | `score` for a number, or the level with the highest probability for a level |
| `noul` | `noul`, the probability that the answer is yes | yes when `noul` is 0.5 or more |

- `answer_confidence` is the probability of the most likely option. For `choice` that is the chosen label, for `score` the most likely level, which can differ from the expected `score`, and for `noul` the larger of yes and no. Use it to decide whether to trust an answer.
- For `choice` and `score`, `confidence` is 1 minus the normalized entropy of the probabilities. It is 0 for an even split and 1 when everything is on one option. For `noul`, it equals `answer_confidence`, which is 0.5 for an even split. Don't compare the 2 against the same threshold.
- `action.act_probability` is the model's probability, from a second output, for acting on the answer rather than escalating.
- The server rounds every probability and score to 4 decimals. `usage.input_tokens` counts the tokens of all the rows, and `routing` names the model that answered.

The `X-Inference-Time-Ms` and `Server-Timing: inference;dur=<ms>` headers give the time spent tokenizing, running the model and decoding.

## Errors

Every error body is `{"detail": "<message>"}`, and `detail` says what was wrong.

| status | cause | what to do |
| --- | --- | --- |
| 400 | invalid JSON, no `state`, no `questions` object, or a `model` that names another Laya model | Fix the request. |
| 401 | the server has an API key, and the request sent no `Authorization: Bearer <key>` or a wrong one | Send the key. |
| 408 | the body took over 10 s to arrive | Send it again. |
| 413 | the body, the state, the questions or the options went over a limit | Split the request or shorten the state. |
| 422 | a malformed question, such as a `choice` without `criteria`, or `max_len` or `head_max_len` | Fix the question that `detail` names. |
| 503 with `Retry-After: 1` | the server already holds as many requests as `--max-concurrent` allows, 16 by default | Wait 1 s and retry. |
| 503 without `Retry-After` | the inference thread stopped | Restart the server. |
| 500 | inference failed | Read the server's log. |

The GPU runs one request at a time. The server holds up to 16, one running and the rest waiting their turn (`--max-concurrent`), and answers any request past that with 503 at once. More parallel clients don't get more answers per second.

## Examples

With curl, add `-i` to see the timing headers:

```sh
curl -s 127.0.0.1:8000/v1/systemone -H 'content-type: application/json' \
  -d '{"state": "My card was charged twice.", "questions": {"money": {"type": "noul", "instructions": "Is money involved?"}}}'
```

With Python's standard library:

```python
import json
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8000/v1/systemone"


def ask(state, questions, api_key=None, tries=5):
    """Send one state and its questions, and return the answers object."""
    body = json.dumps({"state": state, "questions": questions}).encode()
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    for attempt in range(tries):
        req = urllib.request.Request(URL, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)["answers"]
        except urllib.error.HTTPError as e:
            retry = e.headers.get("Retry-After")
            if e.code == 503 and retry and attempt + 1 < tries:
                time.sleep(int(retry))
                continue
            raise RuntimeError(f"{e.code}: {json.load(e)['detail']}") from e


answers = ask(
    "Customer: I was charged twice for my subscription this month and I need it fixed today.",
    {
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "payments, charges and refunds",
                              "tech": "bugs and outages", "sales": "new purchases"}},
        "money": {"type": "noul", "instructions": "Is money involved?"},
    },
)
print(answers["team"]["choice"], answers["team"]["answer_confidence"])
print("money involved:", answers["money"]["noul"] >= 0.5)
```

A program can also start its own server on a free port and read the address from the JSON line that `serve` prints on stdout once the model is loaded:

```python
import json
import subprocess

proc = subprocess.Popen(["sys1rust", "serve", "--port", "0"], stdout=subprocess.PIPE, text=True)
ready = json.loads(proc.stdout.readline())  # {"event": "listening", "addr": "127.0.0.1:<port>", ...}
URL = f"http://{ready['addr']}/v1/systemone"
# ... call ask() as above ...
proc.terminate()  # SIGTERM lets requests in flight finish before it exits
proc.wait()
```

If `serve` fails, it prints the error on stderr and exits without that line, so `readline()` returns an empty string. A server already running on the Mac shares the GPU with this one, so prefer one server per Mac.

## Models

One server runs one model, `typed-decisions` unless `--model` says otherwise.

| `--model` | Hugging Face repo | encoder | download |
| --- | --- | --- | --- |
| `typed-decisions` (default) | `convaiinnovations/laya-typed-decisions` | ModernBERT-large | 846 MB |
| `multilingual` | `convaiinnovations/laya-multilingual` | mmBERT-base | 678 MB |
| `english` | `convaiinnovations/laya` | ModernBERT-large | 846 MB |

To switch, stop the server and start it with the model you want:

```sh
sys1rust serve --model multilingual
```

That start downloads the model if the cache lacks it, and `sys1rust pull multilingual` downloads it ahead of time. `sys1rust models` lists what the cache holds. `--model` also takes a repo id from the table or a local checkpoint directory. The LaunchAgent that `install.sh --service` installs runs the default model. To serve another one, stop the service with `launchctl bootout gui/$(id -u)/io.github.krishhgg.sys1rust` and run `sys1rust serve --model <name>` yourself.

## Options

Each flag falls back to an environment variable. `sys1rust serve --help` lists all of them.

| flag | environment variable | default | effect |
| --- | --- | --- | --- |
| `--model` | `SYS1_MODEL` | `typed-decisions` | the model, from the table above |
| `--host` | `LAYA_HOST` | `127.0.0.1` | the bind address |
| `--port` | `LAYA_PORT` | `8000` | the port. `0` picks a free one and prints it in the `addr` of the stdout line |
| `--api-key` | `LAYA_API_KEY` | none | when set, `/v1/systemone` needs `Authorization: Bearer <key>` |
| `--max-concurrent` | `LAYA_MAX_CONCURRENT` | `16` | requests held at once, past which the server answers 503 |
| `--offline` | `HF_HUB_OFFLINE` | off | never download. A model that the cache lacks is then an error. `HF_HUB_OFFLINE` turns it on when set to `1`, `ON`, `YES` or `TRUE`, in any letter case |

Keep the default host. sys1rust speaks plain HTTP with no TLS, so an API key alone doesn't make it safe to reach from other machines. The README's "Serving other machines" section says how to put a reverse proxy in front.
