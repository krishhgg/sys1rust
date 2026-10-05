# Results

Measured write-ups, in the order they were done:

1. `REPORT.md`: the base-M5 bake-off of existing Laya runtimes (227 runs). `tables.md` and `summary.json` hold every number.
2. `SYNTHESIS.md`: what the bake-off means and the plan for this runtime.
3. `SPIKE.md`: the Rust MLX spike, the fp16 slowdown fix and the MLX buffer cache cap.
4. `SERVER.md`: the HTTP server, named `sys1d` until it became `sys1rust serve`.
5. `SPEED.md`: the speed settings that make the runtime faster than Python MLX, and their review.

The documents cite raw result files under `raw/`. Those (about 170 MB of JSONL and logs) stay out of the repository.
