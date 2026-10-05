#!/usr/bin/env python3
"""Release smoke test. Starts `SYS1RUST serve --offline --port 0` with typed-decisions, sends the
24 requests in bench/workloads/smoke.jsonl, compares the answers with the fp32 CPU reference in
bench/reference/typed-decisions/smoke.jsonl by the rules of bench/harness/compare.py (each
question's answer agrees unless the reference is a near tie, and probabilities are within
MAX_DRIFT), then stops the server with SIGINT and requires exit code 0.

Run `SYS1RUST pull` first. Usage: packaging/smoke.py PATH_TO_SYS1RUST
"""
import http.client
import json
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


def main():
    exe = sys.argv[1]
    reqs = read_jsonl(os.path.join(ROOT, "bench", "workloads", "smoke.jsonl"))
    ref = {r["id"]: r["answers"]
           for r in read_jsonl(os.path.join(ROOT, "bench", "reference", MODEL, "smoke.jsonl"))
           if r.get("type") == "result"}
    assert len(reqs) == REQUESTS and len(ref) == REQUESTS, (len(reqs), len(ref))

    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("LAYA_", "SYS1_")) and k != "HF_HUB_OFFLINE"}
    proc = subprocess.Popen([exe, "serve", "--offline", "--model", MODEL, "--port", "0"],
                            stdout=subprocess.PIPE, text=True, env=env)
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(line) for line in proc.stdout], daemon=True).start()
    failures, questions, worst = [], 0, 0.0
    try:
        try:
            ready = json.loads(lines.get(timeout=READY_TIMEOUT_S))
        except queue.Empty:
            sys.exit(f"no ready line within {READY_TIMEOUT_S} s")
        host, port = ready["addr"].rsplit(":", 1)
        print(f"ready: {ready['model']} @ {ready['revision']}, engine {ready['engine']}, "
              f"load {ready['load_ms']} ms", flush=True)
        conn = http.client.HTTPConnection(host, int(port), timeout=120)
        for r in reqs:
            conn.request("POST", "/v1/systemone", body=json.dumps(dict(r["body"], model=MODEL)),
                         headers={"content-type": "application/json"})
            resp = conn.getresponse()
            data = resp.read()
            if resp.status != 200:
                failures.append(f"{r['id']}: HTTP {resp.status} {data[:200]!r}")
                continue
            answers = json.loads(data)["answers"]
            for q, want in ref[r["id"]].items():
                questions += 1
                got = answers.get(q)
                if got is None:
                    failures.append(f"{r['id']} {q}: missing")
                    continue
                if not agree(got, want) and margin(want) >= NEAR_TIE:
                    failures.append(f"{r['id']} {q}: {got} vs reference {want}")
                d = drift(got, want)
                if d is not None:
                    worst = max(worst, d)
                    if d > MAX_DRIFT:
                        failures.append(f"{r['id']} {q}: drift {d:.4f} > {MAX_DRIFT}")
        conn.close()
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            code = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = "killed after 30 s"
    print(f"{REQUESTS} requests, {questions} questions, largest drift {worst:.4f}, "
          f"{len(failures)} failures, server exit {code}")
    for f in failures:
        print("FAIL", f)
    if failures or code != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
