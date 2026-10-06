#!/usr/bin/env python3
"""Release smoke test. Starts `SYS1RUST serve --offline --port 0` with typed-decisions, sends the
24 requests in bench/workloads/smoke.jsonl, compares the answers with the fp32 CPU reference in
bench/reference/typed-decisions/smoke.jsonl by the rules of bench/harness/compare.py (each
question's answer agrees unless the reference is a near tie, and probabilities are within
MAX_DRIFT), then stops the server with SIGINT and requires exit code 0. confidence,
answer_confidence and action.act_probability must be within MAX_DRIFT too.

Every reply must also have the reference's shape, as the strict suite checks it
(runtime/crates/sys1rust/tests/live.rs): the reference's question ids, each answer's fields in
the reference's order, the reference's probability labels in its order and the reference's
legend. Probabilities, noul, the confidences and act_probability must be numbers in [0, 1], and
score a finite number. So a near tie or a missing field can't hide a broken answer. A request
that fails in transport, runs past REQUEST_TIMEOUT_S or returns bad JSON fails the run and stops
sending. The script always stops the server, also when it gets SIGINT or SIGTERM.

Run `SYS1RUST pull` first. Usage: packaging/smoke.py PATH_TO_SYS1RUST
"""
import http.client
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench", "harness"))
from compare import NEAR_TIE, agree, drift, margin, read_jsonl  # noqa: E402

MODEL = "typed-decisions"
REQUESTS = 24
MAX_DRIFT = 0.05
READY_TIMEOUT_S = 300
# Covers connect, send, headers and body of one request. CI runners have a slow paravirtual GPU.
REQUEST_TIMEOUT_S = 120
STOP_TIMEOUT_S = 30
KILL_TIMEOUT_S = 10


class Interrupted(Exception):
    """The SIGINT and SIGTERM handler raises this so the `finally` in main() stops the server."""


# While Popen starts the server, on_signal only records the signal. main() raises it once `proc`
# holds the server, so the cleanup in `finally` can't miss a child that Popen already started.
launch = {"active": False, "pending": None}


def on_signal(signum, _frame):
    name = signal.Signals(signum).name
    if launch["active"]:
        launch["pending"] = launch["pending"] or name
        return
    raise Interrupted(name)


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def is_probability(v):
    return is_number(v) and 0 <= v <= 1


def malformed(got, want):
    """Says why `got` can't stand in for the reference answer `want`, or returns None if it can.
    live.rs asserts the same key lists, so an extra field or label fails here too."""
    if not isinstance(got, dict):
        return f"answer is {type(got).__name__}, not an object"
    if list(got) != list(want):
        return f"fields {list(got)}, reference fields {list(want)}"
    if got["type"] != want["type"]:
        return f"type {got['type']!r}, reference type {want['type']!r}"
    labels = want.get("probabilities") or {}
    if "choice" in want and not (isinstance(got["choice"], str) and got["choice"] in labels):
        return f"choice {got['choice']!r} is not a reference label"
    if "score" in want and not is_number(got["score"]):
        return f"score {got['score']!r} is not a number"
    for field in ("noul", "confidence", "answer_confidence"):
        if field in want and not is_probability(got[field]):
            return f"{field} {got[field]!r} is not a number in [0, 1]"
    if "legend" in want and got["legend"] != want["legend"]:
        return f"legend {got['legend']!r}, reference legend {want['legend']!r}"
    if "action" in want:
        # The reference's action holds act_probability only.
        action = got["action"]
        if not isinstance(action, dict):
            return f"action {action!r} is not an object"
        for k in want["action"]:
            if not is_probability(action.get(k)):
                return f"action.{k} {action.get(k)!r} is not a number in [0, 1]"
    if "probabilities" in want:
        probs = got["probabilities"]
        if not isinstance(probs, dict) or list(probs) != list(labels):
            return f"probabilities {probs!r}, reference labels {list(labels)}"
        if not all(is_probability(p) for p in probs.values()):
            return "a probability is not a number in [0, 1]"
    return None


def field_drift(got, want):
    """The largest difference in confidence, answer_confidence and action.act_probability,
    the probabilities that compare.py's drift leaves out. malformed() has checked them."""
    pairs = [(got[f], want[f]) for f in ("confidence", "answer_confidence") if f in want]
    pairs += [(got["action"][k], v) for k, v in want.get("action", {}).items()]
    return max((abs(a - b) for a, b in pairs), default=0.0)


def post(host, port, body, out):
    """Sends one request on a fresh connection. Keeps the current stage in out["stage"] so a
    timeout can name it, and leaves out["error"] or out["status"], out["data"] and, after a 200,
    out["answers"]."""
    try:
        out["stage"] = "connect"
        conn = http.client.HTTPConnection(host, port, timeout=REQUEST_TIMEOUT_S)
        conn.connect()
        out["stage"] = "send"
        conn.request("POST", "/v1/systemone", body=body, headers={"content-type": "application/json"})
        out["stage"] = "status"
        resp = conn.getresponse()
        out["status"] = resp.status
        out["stage"] = "read"
        out["data"] = resp.read()
        conn.close()
        if resp.status == 200:
            out["stage"] = "parse"
            answers = json.loads(out["data"])["answers"]
            if not isinstance(answers, dict):
                raise ValueError(f"answers is {type(answers).__name__}, not an object")
            out["answers"] = answers
    except Exception as e:
        out["error"] = e


def check(rid, answers, want_all, failures, stats):
    if list(answers) != list(want_all):
        failures.append(f"{rid}: question ids {list(answers)}, reference {list(want_all)}")
    for q, want in want_all.items():
        stats["questions"] += 1
        where = f"{rid} {q}"
        if q not in answers:
            failures.append(f"{where}: missing")
            continue
        got = answers[q]
        bad = malformed(got, want)
        if bad:
            failures.append(f"{where}: malformed answer, {bad}: {got}")
            continue
        try:
            ok, d = agree(got, want), drift(got, want)
        except (TypeError, ValueError) as e:
            failures.append(f"{where}: cannot compare {got}: {e!r}")
            continue
        if not ok and margin(want) >= NEAR_TIE:
            failures.append(f"{where}: {got} vs reference {want}")
        if d is None:
            failures.append(f"{where}: no probability drift for {got}")
            continue
        d = max(d, field_drift(got, want))
        stats["worst"] = max(stats["worst"], d)
        if d > MAX_DRIFT:
            failures.append(f"{where}: drift {d:.4f} > {MAX_DRIFT}")


def run(proc, reqs, ref, failures, stats):
    lines = queue.Queue()

    def pump():
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    try:
        line = lines.get(timeout=READY_TIMEOUT_S)
    except queue.Empty:
        failures.append(f"no ready line within {READY_TIMEOUT_S} s")
        return
    if line is None:
        failures.append("server exited before printing the ready line")
        return
    ready = json.loads(line)
    host, port = ready["addr"].rsplit(":", 1)
    print(f"ready: {ready['model']} @ {ready['revision']}, engine {ready['engine']}, "
          f"load {ready['load_ms']} ms", flush=True)
    for r in reqs:
        out = {}
        body = json.dumps(dict(r["body"], model=MODEL))
        worker = threading.Thread(target=post, args=(host, int(port), body, out), daemon=True)
        worker.start()
        worker.join(REQUEST_TIMEOUT_S)
        if worker.is_alive():
            failures.append(f"{r['id']}: no full reply within {REQUEST_TIMEOUT_S} s, "
                            f"stuck at stage {out['stage']}")
            return
        if "error" in out:
            failures.append(f"{r['id']}: {out['stage']} failed: {out['error']!r}")
            return
        if out["status"] != 200:
            stats["requests"] += 1
            failures.append(f"{r['id']}: HTTP {out['status']} {out['data'][:200]!r}")
            continue
        stats["requests"] += 1
        check(r["id"], out["answers"], ref[r["id"]], failures, stats)


def stop(proc):
    """Stops the server with SIGINT and returns its exit status, or a note if it needed SIGKILL."""
    if proc.poll() is not None:
        return proc.returncode
    proc.send_signal(signal.SIGINT)
    try:
        return proc.wait(timeout=STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
    try:
        proc.wait(timeout=KILL_TIMEOUT_S)
        return f"killed after {STOP_TIMEOUT_S} s"
    except subprocess.TimeoutExpired:
        return f"pid {proc.pid} still running {KILL_TIMEOUT_S} s after SIGKILL"


def main():
    exe = sys.argv[1]
    reqs = read_jsonl(os.path.join(ROOT, "bench", "workloads", "smoke.jsonl"))
    ref = {r["id"]: r["answers"]
           for r in read_jsonl(os.path.join(ROOT, "bench", "reference", MODEL, "smoke.jsonl"))
           if r.get("type") == "result"}
    assert len(reqs) == REQUESTS and len(ref) == REQUESTS, (len(reqs), len(ref))

    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("LAYA_", "SYS1_")) and k != "HF_HUB_OFFLINE"}
    failures, stats = [], {"requests": 0, "questions": 0, "worst": 0.0}
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    proc, code = None, "not started"
    try:
        launch["active"] = True
        try:
            proc = subprocess.Popen([exe, "serve", "--offline", "--model", MODEL, "--port", "0"],
                                    stdout=subprocess.PIPE, text=True, env=env)
        finally:
            launch["active"] = False
        if launch["pending"]:
            raise Interrupted(launch["pending"])
        run(proc, reqs, ref, failures, stats)
    except Exception as e:
        failures.append(f"stopped early: {e!r}")
    finally:
        # A second signal must not cut the cleanup short. stop() is bounded.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if proc is not None:
            code = stop(proc)
    print(f"{stats['requests']} requests, {stats['questions']} questions, largest drift {stats['worst']:.4f}, "
          f"{len(failures)} failures, server exit {code}")
    for f in failures:
        print("FAIL", f)
    if failures or code != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
