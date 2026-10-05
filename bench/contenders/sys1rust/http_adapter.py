"""Bench adapter for the sys1d HTTP server (bench/PLAN.md, "Adapter interface").

Invoked through ./run for `http*` variants and for --list-variants (which merges in the in-process
variants from sys1-bench). The client matches the bake-off's http contenders (laya-upstream,
laya-apple): Python http.client, one kept-alive connection per client thread, `latency_ms` is the
client-side round trip. Each result also records the server's own inference time
(`X-Inference-Time-Ms`), so the HTTP and queue overhead is `latency_ms - server_ms`.
"""
import argparse
import atexit
import http.client
import json
import os
import signal
import subprocess
import sys
import threading
import time

T_PROCESS_START = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(os.path.dirname(HERE))
LOCK = json.load(open(os.path.join(BENCH, "models.lock.json")))
TARGET = os.path.join(os.path.abspath(os.environ["CARGO_TARGET_DIR"]), "release")
MODELS = ["typed-decisions", "multilingual"]

VARIANTS = [
    {"variant": "http-fp16-fast", "tuning": None,
     "notes": "sys1d with its default engine settings. Since the speed round these are the mlx-fp16-lean "
              "settings (f16gelu, 512 MiB MLX cache, 2 GiB wired, dense local attention up to 1,024 tokens, "
              "head pruning, unpadding, and since rounds 2 and 3 fuserope, band=512, nax=all, the loading "
              "settings and MLX_MAX_MB_PER_BUFFER=10); runs before that used the mlx-fp16-fast settings. The name is kept "
              "so earlier result paths stay valid. One inference thread, admission limit 16. Supports "
              "--concurrency K (K client threads, one kept-alive connection each)."},
]


def emit(fh, obj):
    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fh.flush()


def schedule(reqs, repeats, duration):
    """Yield (repeat, request) in file order, `repeats` times, or until `duration` seconds pass."""
    if duration:
        t_end = time.time() + duration
        rep = 0
        while True:
            for r in reqs:
                if time.time() >= t_end:
                    return
                yield rep, r
            rep += 1
    else:
        for rep in range(repeats):
            for r in reqs:
                yield rep, r


def start_server(model, variant, log_path):
    """Spawn sys1d on a free port and wait for its ready line. Returns (port, ready_ms, ready, proc, stop)."""
    cmd = [TARGET + "/sys1d", "--model", model, "--revision", LOCK[model]["sha"], "--port", "0"]
    if variant["tuning"] is not None:
        cmd += ["--tuning", variant["tuning"]]
    log = open(log_path, "w")
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True, start_new_session=True)

    def stop():
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=15)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        log.close()

    atexit.register(stop)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: sys.exit(1))
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("sys1d exited with %s before it was ready, see %s" % (proc.wait(), log_path))
    ready = json.loads(line)
    port = int(ready["addr"].rsplit(":", 1)[1])
    return port, (time.perf_counter() - t0) * 1000, ready, proc, stop


def run_http(args, variant, reqs, out):
    log_path = args.out + ".server.log"
    port, ready_ms, ready, proc, stop = start_server(args.model, variant, log_path)
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", "/health")
    health = json.loads(c.getresponse().read())
    c.close()
    emit(out, {"type": "meta", "contender": "sys1rust", "variant": variant["variant"], "model": args.model,
               "model_sha": ready.get("revision"), "code_version": os.environ.get("SYS1_CODE_VERSION", "unknown"),
               "backend": "mlx", "mode": "http", "load_ms": round(ready_ms, 1), "pid": os.getpid(),
               "server_pid": proc.pid, "t_process_start": T_PROCESS_START, "concurrency": args.concurrency,
               "warmup": args.warmup, "repeats": args.repeats, "duration": args.duration,
               "server_log": log_path, "ready": ready, "health": health})

    def post(conn, r):
        body = dict(r["body"], model=args.model)
        conn.request("POST", "/v1/systemone", body=json.dumps(body).encode(),
                     headers={"content-type": "application/json"})
        resp = conn.getresponse()
        data = resp.read()
        if resp.status != 200:
            raise RuntimeError("HTTP %d: %s" % (resp.status, data[:300].decode(errors="replace")))
        res = json.loads(data)
        return res["answers"], res.get("usage", {}).get("input_tokens"), resp.getheader("X-Inference-Time-Ms")

    def new_conn():
        return http.client.HTTPConnection("127.0.0.1", port, timeout=600)

    warm = new_conn()
    for r in reqs[: args.warmup]:
        try:
            post(warm, r)
        except Exception:
            warm.close()
            warm = new_conn()
    warm.close()

    lock = threading.Lock()
    it = iter(schedule(reqs, args.repeats, args.duration))

    def worker():
        conn = new_conn()
        while True:
            with lock:
                try:
                    rep, r = next(it)
                except StopIteration:
                    break
            t_start = time.time()
            t0 = time.perf_counter()
            try:
                answers, tokens, server_ms = post(conn, r)
                rec = {"type": "result", "id": r["id"], "repeat": rep, "t_start": t_start,
                       "latency_ms": round((time.perf_counter() - t0) * 1000, 3), "answers": answers,
                       "input_tokens": tokens, "server_ms": float(server_ms) if server_ms else None}
            except Exception as e:
                rec = {"type": "result", "id": r["id"], "repeat": rep, "t_start": t_start,
                       "error": "%s: %s" % (type(e).__name__, str(e)[:500])}
                conn.close()
                conn = new_conn()
            if args.concurrency > 1:
                rec["worker"] = threading.current_thread().name
            with lock:
                emit(out, rec)
        conn.close()

    threads = [threading.Thread(target=worker, name="c%d" % i) for i in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-variants", action="store_true")
    ap.add_argument("--variant")
    ap.add_argument("--model")
    ap.add_argument("--workload")
    ap.add_argument("--out")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--concurrency", type=int, default=1)
    args = ap.parse_args()

    if args.list_variants:
        inproc = json.loads(subprocess.check_output([TARGET + "/sys1-bench", "--list-variants"], text=True))
        http_v = [{"variant": v["variant"], "models": MODELS, "mode": "http", "max_state_tokens": None,
                   "notes": v["notes"]} for v in VARIANTS]
        print(json.dumps(inproc + http_v, indent=1))
        return
    variant = next((v for v in VARIANTS if v["variant"] == args.variant), None)
    if variant is None:
        ap.error("--variant must be one of %s" % [v["variant"] for v in VARIANTS])
    if args.model not in MODELS:
        ap.error("--model must be one of %s" % MODELS)
    if not (args.workload and args.out):
        ap.error("--workload and --out are required")
    with open(args.workload) as f:
        reqs = [json.loads(l) for l in f if l.strip()]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as out:
        run_http(args, variant, reqs, out)


if __name__ == "__main__":
    main()
