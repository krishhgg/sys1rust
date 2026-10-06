#!/bin/bash
# Tests install.sh without the network, the GPU or launchd: packaging/test_install.sh [SHELL]
# SHELL runs install.sh and defaults to /bin/sh. Fake bundles hold a shell script in place of
# sys1rust. Fake uname, sysctl and sw_vers set the Mac that install.sh sees. A local server
# stands in for GitHub releases. A fake launchctl records its calls and serves /health on
# LAYA_PORT while the service is loaded. Each run of install.sh gets a clean environment with
# HOME in a temp dir, so the test touches nothing outside that dir.
# Most checks are single-quoted strings that check() evals, so their variables expand then.
# shellcheck disable=SC2016,SC2034
set -euo pipefail

SH=${1:-/bin/sh}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d "${TMPDIR:-/tmp}/sys1rust-install-test.XXXXXX")
T=$(cd "$T" && pwd -P)
SERVER_PID=
cleanup() {
  if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" && wait "$SERVER_PID"; fi 2>/dev/null || true
  [ ! -f "$T/launchd/pid" ] || kill "$(cat "$T/launchd/pid")" 2>/dev/null || true
  rm -rf "$T"
}
trap cleanup EXIT

passed=0
failed=0
ok() { passed=$((passed + 1)); }
bad() { failed=$((failed + 1)); echo "FAIL: $*"; printf '%s\n' "$out" | sed 's/^/  | /'; }
# check WHAT CMD...: counts CMD's result.
check() {
  local what=$1
  shift
  if "$@"; then ok; else bad "$what"; fi
}
has() { [[ $out == *"$1"* ]]; }
link_is() { [ "$(readlink "$1" 2>/dev/null)" = "$2" ]; }

free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }

# The fake system tools.
mkdir -p "$T/fake" "$T/launchd/www" "$T/home"
touch "$T/launchd/www/health"
cat >"$T/fake/uname" <<'EOF'
#!/bin/sh
case $1 in
  -s) echo "${FAKE_UNAME_S:-Darwin}" ;;
  -m) echo "${FAKE_UNAME_M:-arm64}" ;;
  *) exec /usr/bin/uname "$@" ;;
esac
EOF
cat >"$T/fake/sysctl" <<'EOF'
#!/bin/sh
if [ "$2" = sysctl.proc_translated ]; then echo "${FAKE_TRANSLATED:-0}"; else exec /usr/sbin/sysctl "$@"; fi
EOF
cat >"$T/fake/sw_vers" <<'EOF'
#!/bin/sh
echo "${FAKE_MACOS:-26.2}"
EOF
cat >"$T/fake/launchctl" <<'EOF'
#!/bin/sh
# Records each call. bootstrap starts a /health server on LAYA_PORT, or with fail-next a
# server that exits 1 at once, kickstart restarts it and bootout stops it. print reports like
# launchd: after kickstart -k, the stopped run's exit code 0 stays in the report.
d=$FAKE_LAUNCHD
echo "$*" >>"$d/calls"
case $1 in
  print)
    [ -f "$d/loaded" ] || exit 113
    if [ -f "$d/failed" ]; then
      printf '\tstate = spawn scheduled\n\tlast exit code = 1\n'
    elif [ -f "$d/kicked" ]; then
      printf '\tstate = running\n\tlast exit code = 0\n'
    else
      printf '\tstate = running\n\tlast exit code = (never exited)\n'
    fi
    ;;
  bootstrap)
    [ ! -f "$d/loaded" ] || exit 37
    cp "$3" "$d/plist"
    if [ -f "$d/fail-next" ]; then
      touch "$d/failed"
    else
      python3 -m http.server --bind 127.0.0.1 --directory "$d/www" "$LAYA_PORT" >/dev/null 2>&1 </dev/null &
      echo $! >"$d/pid"
    fi
    touch "$d/loaded"
    ;;
  kickstart)
    # The new run answers 2 s later, as a real server does after loading the model.
    kill "$(cat "$d/pid")"
    (sleep 2 && exec python3 -m http.server --bind 127.0.0.1 --directory "$d/www" "$LAYA_PORT") \
      >/dev/null 2>&1 </dev/null &
    echo $! >"$d/pid"
    touch "$d/kicked"
    ;;
  bootout)
    [ -f "$d/loaded" ] || exit 113
    [ ! -f "$d/pid" ] || kill "$(cat "$d/pid")"
    rm -f "$d/loaded" "$d/pid" "$d/failed" "$d/kicked"
    ;;
esac
EOF
chmod +x "$T/fake"/*

# make_bundle VERSION FLAVOR DIR: writes a fake bundle tarball into DIR and rewrites DIR/SHA256SUMS.
make_bundle() {
  local name=sys1rust-$1-$2-arm64
  mkdir -p "$T/build/$name/bin" "$T/build/$name/lib" "$3"
  cat >"$T/build/$name/bin/sys1rust" <<EOF
#!/bin/sh
case \$1 in
  --version) echo "sys1rust $1 (MLX 0.32.2, $2 build)" ;;
  models)
    echo "MODEL            REPO                                    REVISION STATUS"
    echo "typed-decisions  convaiinnovations/laya-typed-decisions  1a793eb  \${FAKE_MODEL:-not downloaded, 846 MB}"
    ;;
  *) exit 1 ;;
esac
EOF
  chmod +x "$T/build/$name/bin/sys1rust"
  tar -C "$T/build" -czf "$3/$name.tar.gz" "$name"
  (cd "$3" && shasum -a 256 sys1rust-*.tar.gz >SHA256SUMS)
}

# run [VAR=VALUE ...] [--] ARGS...: runs install.sh in a clean environment and sets out and status.
run() {
  local envs=()
  while [ $# -gt 0 ] && [[ $1 == *=* ]]; do envs+=("$1"); shift; done
  [ "${1:-}" != -- ] || shift
  status=0
  out=$(env -i HOME="$T/home" PATH="$T/fake:/usr/bin:/bin:/usr/sbin:/sbin" TMPDIR="$T" \
    FAKE_LAUNCHD="$T/launchd" LAYA_PORT="$PORT" ${envs[@]+"${envs[@]}"} "$SH" "$ROOT/install.sh" "$@" 2>&1) || status=$?
}

# state DIR...: prints the paths under DIR and the targets of its links, to compare 2 runs.
state() {
  find "$@" | sort | while IFS= read -r f; do
    if [ -L "$f" ]; then echo "$f -> $(readlink "$f")"; else echo "$f"; fi
  done
}

PORT=$(free_port)
P=$T/p
B=$T/b
make_bundle 0.1.0 macos26 "$T/d1"
make_bundle 0.1.0 macos14 "$T/d1"
make_bundle 0.2.0 macos26 "$T/d2"
make_bundle 0.3.0 macos26 "$T/d3"
make_bundle 0.4.0 macos26 "$T/d4"
make_bundle 0.6.0 macos26 "$T/d6"

# Help and options, also piped as `curl ... | sh` runs it.
status=0
out=$(env -i PATH=/usr/bin:/bin HOME="$T/home" "$SH" -s -- --help <"$ROOT/install.sh" 2>&1) || status=$?
check "piped --help" eval '[ $status = 0 ] && has "Usage: install.sh"'
run --bogus
check "unknown option exits 2" eval '[ $status = 2 ] && has "install.sh: unknown option"'
run --version
check "--version without a value" eval '[ $status = 2 ] && has "install.sh: --version needs a value"'
run --version 1.2
check "a bad tag" eval '[ $status = 2 ] && has "install.sh: --version takes a tag"'
run --uninstall --service
check "--uninstall with --service" eval '[ $status = 2 ] && has "--uninstall takes only"'

# Platform checks.
run FAKE_UNAME_S=Linux -- --from "$T/d1" --prefix "$P" --bin-dir "$B"
check "Linux" eval '[ $status = 1 ] && has "install.sh: sys1rust runs only on macOS, and this system is Linux"'
run FAKE_TRANSLATED=1 FAKE_UNAME_M=x86_64 -- --from "$T/d1" --prefix "$P" --bin-dir "$B"
check "Rosetta" eval '[ $status = 1 ] && has "install.sh: this shell runs under Rosetta"'
run FAKE_UNAME_M=x86_64 -- --from "$T/d1" --prefix "$P" --bin-dir "$B"
check "Intel" eval '[ $status = 1 ] && has "install.sh: sys1rust needs an Apple silicon (arm64) Mac, and this one is x86_64"'
run FAKE_MACOS=13.6.1 -- --from "$T/d1" --prefix "$P" --bin-dir "$B"
check "macOS 13" eval '[ $status = 1 ] && has "install.sh: sys1rust needs macOS 14.0 or later, and this Mac runs 13.6.1"'
check "failed checks make no prefix" eval '[ ! -e "$P" ]'

# The macOS version picks the bundle.
for pair in 26.2:macos26 26.10:macos26 26.2.1:macos26 27:macos26 26.1:macos14 26:macos14 \
  15.7.1:macos14 14.0:macos14 14:macos14; do
  rm -rf "$T/f"
  run FAKE_MACOS="${pair%%:*}" -- --from "$T/d1" --prefix "$T/f" --bin-dir "$T/f/bin"
  check "macOS ${pair%%:*} picks ${pair#*:}" link_is "$T/f/current" "0.1.0-${pair#*:}"
done

# Install, rerun, update, prune and a bad checksum, all from local dirs.
run --from "$T/d1" --prefix "$P" --bin-dir "$B"
check "install from d1" eval '[ $status = 0 ] && has "Installed sys1rust 0.1.0 (MLX 0.32.2, macos26 build) in $P/0.1.0-macos26"'
check "version dir" test -x "$P/0.1.0-macos26/bin/sys1rust"
check "current link" link_is "$P/current" 0.1.0-macos26
check "bin link" link_is "$B/sys1rust" "$P/current/bin/sys1rust"
check "--version through the link" eval '[ "$("$B/sys1rust" --version)" = "sys1rust 0.1.0 (MLX 0.32.2, macos26 build)" ]'
check "PATH line for zsh" has "  export PATH=\"$B:\$PATH\""
check "next step and model size" eval 'has "Next, start the server: $B/sys1rust serve" && has "typed-decisions model (846 MB)"'
check "no staging dir left" eval '[ -z "$(find "$P" -maxdepth 1 -name ".install.*")" ]'
before=$(state "$P" "$B")
run --from "$T/d1" --prefix "$P" --bin-dir "$B"
check "rerun succeeds" test "$status" = 0
check "rerun leaves the same files and links" eval '[ "$(state "$P" "$B")" = "$before" ]'
run --from "$T/d2" --prefix "$P" --bin-dir "$B"
check "update to 0.2.0" eval '[ $status = 0 ] && link_is "$P/current" 0.2.0-macos26 && link_is "$P/previous" 0.1.0-macos26'
check "0.1.0 kept as previous" test -d "$P/0.1.0-macos26"
run --from "$T/d3" --prefix "$P" --bin-dir "$B"
check "update to 0.3.0" eval '[ $status = 0 ] && link_is "$P/current" 0.3.0-macos26 && link_is "$P/previous" 0.2.0-macos26'
check "0.1.0 pruned" eval '[ ! -e "$P/0.1.0-macos26" ] && has "Removed the old version 0.1.0-macos26"'
check "2 versions left" eval '[ "$(find "$P" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d " ")" = 2 ]'
# head closes the pipe after 1 line, and each install must still finish, pruning included.
status=0
for d in d3 d2 d1; do
  env -i HOME="$T/home" PATH="$T/fake:/usr/bin:/bin:/usr/sbin:/sbin" TMPDIR="$T" FAKE_LAUNCHD="$T/launchd" \
    "$SH" "$ROOT/install.sh" --from "$T/$d" --prefix "$T/h" --bin-dir "$T/hb" | head -n 1 >/dev/null || status=$?
done
out=$(ls -a "$T/h")
check "output closed early still finishes" eval '[ $status = 0 ] && link_is "$T/h/current" 0.1.0-macos26 && [ ! -e "$T/h/0.3.0-macos26" ]'
printf 'corrupt' | dd of="$T/d4/sys1rust-0.4.0-macos26-arm64.tar.gz" bs=1 seek=100 conv=notrunc 2>/dev/null
run --from "$T/d4" --prefix "$P" --bin-dir "$B"
check "bad checksum fails" eval '[ $status = 1 ] && has "install.sh: sys1rust-0.4.0-macos26-arm64.tar.gz has sha256"'
check "bad checksum changes nothing" eval 'link_is "$P/current" 0.3.0-macos26 && [ ! -e "$P/0.4.0-macos26" ]'
check "bad checksum leaves no staging dir" eval '[ -z "$(find "$P" -maxdepth 1 -name ".install.*")" ]'
mkdir -p "$T/b2" && touch "$T/b2/sys1rust"
run --from "$T/d1" --prefix "$T/p2" --bin-dir "$T/b2"
check "a file at the link path" eval '[ $status = 1 ] && has "install.sh: $T/b2/sys1rust exists and is not a link"'

# Downloads from a local server laid out like GitHub releases.
W=$T/www
for tag in v0.2.0 v0.3.0-rc1; do
  mkdir -p "$W/releases/download/$tag" "$W/releases/tag/$tag"
done
cp "$T/d2"/sys1rust-0.2.0-macos26-arm64.tar.gz "$T/d2/SHA256SUMS" "$W/releases/download/v0.2.0/"
cp "$T/d3"/sys1rust-0.3.0-macos26-arm64.tar.gz "$W/releases/download/v0.3.0-rc1/"
(cd "$W/releases/download/v0.3.0-rc1" && shasum -a 256 sys1rust-*.tar.gz >SHA256SUMS)
mkdir -p "$W/releases/tag/v0.5.0" "$W/releases/download/v0.6.0"
cp "$T/d6"/* "$W/releases/download/v0.6.0/"
printf 'corrupt' | dd of="$W/releases/download/v0.6.0/sys1rust-0.6.0-macos26-arm64.tar.gz" bs=1 seek=100 conv=notrunc 2>/dev/null
echo v0.2.0 >"$W/latest-tag"
cat >"$T/server.py" <<'EOF'
# Serves sys.argv[1] like github.com/<repo>: /releases/latest redirects to the tag in
# latest-tag, or to /releases when that file is missing. Writes its port to sys.argv[2].
import http.server, os, sys
root = sys.argv[1]
class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=root, **k)
    def do_GET(self):
        if self.path == "/releases/latest":
            f = os.path.join(root, "latest-tag")
            to = "/releases/tag/" + open(f).read().strip() if os.path.exists(f) else "/releases"
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:%d%s" % (self.server.server_port, to))
            self.end_headers()
            return
        super().do_GET()
    def log_message(self, *a):
        pass
server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
with open(sys.argv[2], "w") as f:
    f.write(str(server.server_port))
server.serve_forever()
EOF
python3 "$T/server.py" "$W" "$T/server-port" &
SERVER_PID=$!
for _ in $(seq 50); do [ -s "$T/server-port" ] && break; sleep 0.1; done
URL=http://127.0.0.1:$(cat "$T/server-port")/releases
run SYS1RUST_RELEASES_URL="$URL" -- --prefix "$T/u" --bin-dir "$T/u/bin"
check "latest release" eval '[ $status = 0 ] && link_is "$T/u/current" 0.2.0-macos26 && has "Downloading $URL/download/v0.2.0/sys1rust-0.2.0-macos26-arm64.tar.gz"'
run SYS1RUST_RELEASES_URL="$URL" -- --prefix "$T/u" --bin-dir "$T/u/bin"
check "rerun on the latest skips the download" eval '[ $status = 0 ] && has "already in" && ! has Downloading'
run SYS1RUST_RELEASES_URL="$URL" -- --version v0.3.0-rc1 --prefix "$T/u" --bin-dir "$T/u/bin"
check "--version picks a release candidate" eval '[ $status = 0 ] && link_is "$T/u/current" 0.3.0-rc1-macos26'
run SYS1RUST_RELEASES_URL="$URL" -- --version 0.2.0 --prefix "$T/u" --bin-dir "$T/u/bin"
check "--version without v rolls back to previous" eval '[ $status = 0 ] && link_is "$T/u/current" 0.2.0-macos26 && link_is "$T/u/previous" 0.3.0-rc1-macos26'
run SYS1RUST_RELEASES_URL="$URL" -- --version v9.9.9 --prefix "$T/u" --bin-dir "$T/u/bin"
check "missing tag" eval '[ $status = 1 ] && has "install.sh: there is no release v9.9.9"'
run SYS1RUST_RELEASES_URL="$URL" -- --version v9.9.9 --prefix "$T/u3" --bin-dir "$T/u3/bin"
check "a failed first install leaves no prefix" eval '[ $status = 1 ] && [ ! -e "$T/u3" ]'
run SYS1RUST_RELEASES_URL="$URL" -- --version v0.5.0 --prefix "$T/u" --bin-dir "$T/u/bin"
check "release without SHA256SUMS" eval '[ $status = 1 ] && has "install.sh: release v0.5.0 has no SHA256SUMS"'
run SYS1RUST_RELEASES_URL="$URL" FAKE_MACOS=15.5 -- --version v0.2.0 --prefix "$T/u" --bin-dir "$T/u/bin"
check "release without this flavor" eval '[ $status = 1 ] && has "install.sh: release v0.2.0 has no sys1rust-0.2.0-macos14-arm64.tar.gz"'
run SYS1RUST_RELEASES_URL="$URL" -- --version v0.6.0 --prefix "$T/u" --bin-dir "$T/u/bin"
check "a bad download fails" eval '[ $status = 1 ] && has "install.sh: sys1rust-0.6.0-macos26-arm64.tar.gz has sha256" && link_is "$T/u/current" 0.2.0-macos26'
check "a bad download is deleted" eval '[ -z "$(find "$T/u" -name "*.tar.gz")" ] && [ ! -e "$T/u/0.6.0-macos26" ]'
rm "$W/latest-tag"
run SYS1RUST_RELEASES_URL="$URL" -- --prefix "$T/u2" --bin-dir "$T/u2/bin"
check "no release yet" eval '[ $status = 1 ] && has "install.sh: sys1rust has no published release yet"'

# The LaunchAgent, with the fake launchctl.
AGENT=$T/home/Library/LaunchAgents/io.github.krishhgg.sys1rust.plist
LOG=$T/home/Library/Logs/sys1rust.log
S=$T/s
run HF_HUB_CACHE="$T/hub" -- --from "$T/d1" --prefix "$S" --bin-dir "$T/sb" --service
check "--service" eval '[ $status = 0 ] && has "sys1rust serve answers on http://127.0.0.1:$PORT" && has "with HF_HUB_CACHE=$T/hub"'
check "plist is valid" plutil -lint -s "$AGENT"
check "plist runs the link" eval '[ "$(plutil -extract ProgramArguments.0 raw "$AGENT")" = "$T/sb/sys1rust" ] && [ "$(plutil -extract ProgramArguments.1 raw "$AGENT")" = serve ]'
check "plist passes HF_HUB_CACHE and LAYA_PORT" eval '[ "$(plutil -extract EnvironmentVariables.HF_HUB_CACHE raw "$AGENT")" = "$T/hub" ] && [ "$(plutil -extract EnvironmentVariables.LAYA_PORT raw "$AGENT")" = "$PORT" ]'
check "plist restarts on failure only" eval '[ "$(plutil -extract KeepAlive.SuccessfulExit raw "$AGENT")" = false ] && [ "$(plutil -extract RunAtLoad raw "$AGENT")" = true ]'
check "plist logs" eval '[ "$(plutil -extract StandardErrorPath raw "$AGENT")" = "$LOG" ]'
: >"$T/launchd/calls"
run HF_HUB_CACHE="$T/hub" -- --from "$T/d1" --prefix "$S" --bin-dir "$T/sb" --service
check "--service rerun reloads" eval '[ $status = 0 ] && grep -q "^bootout gui/$(id -u)/io.github.krishhgg.sys1rust$" "$T/launchd/calls" && grep -q "^bootstrap gui/$(id -u) $AGENT$" "$T/launchd/calls"'
: >"$T/launchd/calls"
run --from "$T/d2" --prefix "$S" --bin-dir "$T/sb"
check "an update restarts the service" eval '[ $status = 0 ] && grep -q "^kickstart -k gui/$(id -u)/io.github.krishhgg.sys1rust$" "$T/launchd/calls" && has "Restarting the sys1rust service"'
: >"$T/launchd/calls"
run --from "$T/d2" --prefix "$S" --bin-dir "$T/sb"
check "a rerun leaves the service alone" eval '[ $status = 0 ] && ! grep -q kickstart "$T/launchd/calls" && has "The sys1rust service runs this version"'
# An install and uninstall in another prefix leave this service alone.
: >"$T/launchd/calls"
run --from "$T/d3" --prefix "$T/o" --bin-dir "$T/ob"
check "another prefix does not restart the service" eval '[ $status = 0 ] && ! grep -q kickstart "$T/launchd/calls" && has "Next, start the server"'
run --uninstall --prefix "$T/o" --bin-dir "$T/ob"
check "another prefix keeps the LaunchAgent" eval '[ $status = 0 ] && has "Left the LaunchAgent $AGENT and its log" && [ -f "$AGENT" ] && [ -f "$T/launchd/loaded" ] && [ ! -e "$T/o" ]'

# Uninstall.
run --from "$T/d2" --prefix "$S" --bin-dir "$T/sb" --service
mkdir -p "$T/hub/models--convaiinnovations--laya-typed-decisions/blobs"
dd if=/dev/zero of="$T/hub/models--convaiinnovations--laya-typed-decisions/blobs/x" bs=1000000 count=3 2>/dev/null
echo log >"$LOG"
run HF_HUB_CACHE="$T/hub" -- --uninstall --prefix "$S" --bin-dir "$T/sb"
check "--uninstall" eval '[ $status = 0 ] && has "Stopped the sys1rust service" && [ ! -e "$T/launchd/loaded" ]'
check "--uninstall removes everything" eval '[ ! -e "$AGENT" ] && [ ! -e "$S" ] && [ ! -L "$T/sb/sys1rust" ] && [ ! -e "$LOG" ]'
check "--uninstall keeps the models" eval 'has "Kept the Laya models in the model cache $T/hub, 3 MB" && has "rm -rf \"$T/hub\"/models--convaiinnovations--laya*" && [ -d "$T/hub/models--convaiinnovations--laya-typed-decisions" ]'
check "health server stopped" eval '! curl -fs --max-time 2 "http://127.0.0.1:$PORT/health"'
run --from "$T/d1" --prefix "$T/n" --bin-dir "$T/n/bin"
run --uninstall --prefix "$T/n" --bin-dir "$T/n/bin"
check "--uninstall removes a bin dir inside the prefix" eval '[ $status = 0 ] && has "Removed $T/n" && [ ! -e "$T/n" ]'
run --from "$T/d1" --prefix "$P" --bin-dir "$B"
touch "$P/notes.txt"
ln -sf /usr/bin/true "$B/sys1rust"
run --uninstall --prefix "$P" --bin-dir "$B"
check "--uninstall keeps foreign files" eval '[ $status = 0 ] && has "Left $P, which holds files" && [ -f "$P/notes.txt" ] && [ ! -e "$P/current" ]'
check "--uninstall keeps a foreign link" eval 'has "Left $B/sys1rust, which points to /usr/bin/true" && [ -L "$B/sys1rust" ]'
check "--uninstall default cache line" eval 'has "$T/home/.cache/huggingface/hub holds no Laya models"'

# A server that exits at once fails --service without waiting out HEALTH_WAIT.
touch "$T/launchd/fail-next"
SECONDS=0
run --from "$T/d1" --prefix "$T/s2" --bin-dir "$T/sb2" --service
check "a failing service is reported" eval '[ $status = 1 ] && has "install.sh: the service exited before it answered /health" && [ $SECONDS -lt 10 ]'
rm "$T/launchd/fail-next"
run --uninstall --prefix "$T/s2" --bin-dir "$T/sb2"

# Another server on the port stops --service before it loads the LaunchAgent.
OTHER_PORT=$(free_port)
python3 -m http.server --bind 127.0.0.1 --directory "$T/launchd/www" "$OTHER_PORT" >/dev/null 2>&1 &
OTHER=$!
for _ in $(seq 50); do curl -fs -o /dev/null "http://127.0.0.1:$OTHER_PORT/health" && break; sleep 0.1; done
run LAYA_PORT="$OTHER_PORT" -- --from "$T/d1" --prefix "$T/s3" --bin-dir "$T/sb3" --service
kill "$OTHER"
wait "$OTHER" 2>/dev/null || true
check "a busy port stops --service" eval '[ $status = 1 ] && has "install.sh: another server answers on 127.0.0.1:$OTHER_PORT" && [ ! -e "$AGENT" ]'

echo "$SH: $passed passed, $failed failed"
[ "$failed" = 0 ]
