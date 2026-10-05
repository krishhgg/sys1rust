#!/bin/bash
# Stage D: the sys1rust server over HTTP against the in-process engine, on AC power.
# Also reruns the in-process engine and the cache-capped Python control on AC (the bar was set on AC).
cd "$(dirname "${BASH_SOURCE[0]}")"
# The recorded run used STAMP=20260929T123000. By default a new run gets a fresh stamp, so it
# cannot overwrite the recorded results.
S=${STAMP:-$(date +%Y%m%dT%H%M%S)}
fail=0
./run_stage.sh $S-D \
  "sys1rust http-fp16-fast typed-decisions correctness 5 1" \
  "sys1rust mlx-fp16-fast typed-decisions timing 12 2" \
  "sys1rust http-fp16-fast typed-decisions timing 12 2" \
  "laya-mlx mlx-fp16-opt-c512 typed-decisions timing 12 2" \
  "sys1rust http-fp16-fast typed-decisions short 5 5" \
  "sys1rust mlx-fp16-fast typed-decisions short 5 5" || fail=1
./run_stage.sh $S-c4 "sys1rust http-fp16-fast typed-decisions timing 12 1 - 4" || fail=1
for i in 1 2 3; do
  ./run_stage.sh $S-cold$i "sys1rust http-fp16-fast typed-decisions cold 0 1" || fail=1
done
./run_stage.sh $S-sus "sys1rust http-fp16-fast typed-decisions timing 12 1 300" || fail=1
exit $fail
