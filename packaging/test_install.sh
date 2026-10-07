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
INSTALL_PID=
cleanup() {
  if [ -n "$INSTALL_PID" ]; then kill "$INSTALL_PID" && wait "$INSTALL_PID"; fi 2>/dev/null || true
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
version_is() { case $(readlink "$1" 2>/dev/null) in "$2".??????) return 0 ;; *) return 1 ;; esac; }

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
# server that exits 1 at once, kickstart restarts it and bootout stops it. After kickstart -k,
# print keeps the stopped run's exit code 0 in its report, as launchd does.
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
    [ ! -f "$d/pid" ] || printf '\tpid = %s\n' "$(cat "$d/pid")"
    ;;
  bootstrap)
    [ ! -f "$d/loaded" ] || exit 37
    cp "$3" "$d/plist"
    if [ -f "$d/fail-next" ]; then
      touch "$d/failed"
    else
      python3 -m http.server --bind 127.0.0.1 --directory "$d/www" "$(plutil -extract EnvironmentVariables.LAYA_PORT raw "$d/plist" 2>/dev/null || echo 8000)" >/dev/null 2>&1 </dev/null &
      echo $! >"$d/pid"
    fi
    touch "$d/loaded"
    ;;
  kickstart)
    # The new run answers 2 s later, as a real server does after loading the model.
    [ ! -f "$d/kick-fail-next" ] || exit 5
    [ ! -f "$d/pid" ] || kill "$(cat "$d/pid")" 2>/dev/null || true
    if [ -f "$d/fail-next" ]; then
      rm -f "$d/pid"
      touch "$d/failed"
      exit 0
    fi
    if [ -f "$d/no-listener-next" ]; then
      # The job has a live PID before it exits, while a different process serves /health.
      (sleep 2; touch "$d/failed"; sleep 30) >/dev/null 2>&1 </dev/null &
      echo $! >"$d/pid"
      exit 0
    fi
    (sleep 2 && exec python3 -m http.server --bind 127.0.0.1 --directory "$d/www" "$(plutil -extract EnvironmentVariables.LAYA_PORT raw "$d/plist" 2>/dev/null || echo 8000)") \
      >/dev/null 2>&1 </dev/null &
    echo $! >"$d/pid"
    rm -f "$d/failed"
    touch "$d/kicked"
    ;;
  bootout)
    [ -f "$d/loaded" ] || exit 113
    [ ! -f "$d/pid" ] || kill "$(cat "$d/pid")"
    rm -f "$d/loaded" "$d/pid" "$d/failed" "$d/kicked"
    ;;
esac
EOF
cat >"$T/fake/mv" <<'EOF'
#!/bin/sh
case $1 in
  -fh)
    if [ "${FAKE_MV_MODE:-}" = after-switch ]; then
      case $3 in
        */current) /bin/mv "$@"; kill -TERM "$PPID"; exit 1 ;;
      esac
    fi
    ;;
  */.install.*/sys1rust-*)
    case ${FAKE_MV_MODE:-} in
      fail) exit 1 ;;
      signal) kill -TERM "$PPID"; exit 1 ;;
      pause)
        touch "$FAKE_MV_READY"
        while [ ! -e "$FAKE_MV_RESUME" ]; do sleep 0.1; done
        ;;
    esac
    ;;
esac
exec /bin/mv "$@"
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
    [ -z "\${FAKE_MODELS_ENV:-}" ] || printf '%s\n' "\${HF_HUB_CACHE:-unset}" >"\$FAKE_MODELS_ENV"
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
  check "macOS ${pair%%:*} picks ${pair#*:}" version_is "$T/f/current" "0.1.0-${pair#*:}"
done

# Install, rerun, update, prune and a bad checksum, all from local dirs.
run --from "$T/d1" --prefix "$P" --bin-dir "$B"
check "install from d1" eval '[ $status = 0 ] && has "Installed sys1rust 0.1.0 (MLX 0.32.2, macos26 build) in $P/0.1.0-macos26"'
first=$(readlink "$P/current")
check "version dir" test -x "$P/$first/bin/sys1rust"
check "current link" version_is "$P/current" 0.1.0-macos26
check "bin link" link_is "$B/sys1rust" "$P/current/bin/sys1rust"
check "--version through the link" eval '[ "$("$B/sys1rust" --version)" = "sys1rust 0.1.0 (MLX 0.32.2, macos26 build)" ]'
check "PATH line for zsh" has "  export PATH='$B':\"\$PATH\""
check "next step" has "Next, start the server: '$B/sys1rust' serve"
check "model size" has "typed-decisions model (846 MB)"
check "no staging dir left" eval '[ -z "$(find "$P" -maxdepth 1 -name ".install.*")" ]'
run --from "$T/d1" --prefix "$P" --bin-dir "$B"
check "rerun succeeds" test "$status" = 0
check "rerun switches immutable directories" eval '[ "$(readlink "$P/current")" != "$first" ] && link_is "$P/previous" "$first" && [ -x "$P/$first/bin/sys1rust" ]'
second=$(readlink "$P/current")
run --from "$T/d2" --prefix "$P" --bin-dir "$B"
check "update to 0.2.0" eval '[ $status = 0 ] && version_is "$P/current" 0.2.0-macos26 && version_is "$P/previous" 0.1.0-macos26'
check "0.1.0 kept as previous" test -d "$P/$second"
run --from "$T/d3" --prefix "$P" --bin-dir "$B"
check "update to 0.3.0" eval '[ $status = 0 ] && version_is "$P/current" 0.3.0-macos26 && version_is "$P/previous" 0.2.0-macos26'
check "0.1.0 pruned" eval '[ ! -e "$P/$second" ] && has "Removed the old version 0.1.0-macos26"'
check "2 versions left" eval '[ "$(find "$P" -mindepth 1 -maxdepth 1 -type d ! -name ".*" | wc -l | tr -d " ")" = 2 ]'
# head closes the pipe after 1 line, and each install must still finish, pruning included.
status=0
for d in d3 d2 d1; do
  env -i HOME="$T/home" PATH="$T/fake:/usr/bin:/bin:/usr/sbin:/sbin" TMPDIR="$T" FAKE_LAUNCHD="$T/launchd" \
    "$SH" "$ROOT/install.sh" --from "$T/$d" --prefix "$T/h" --bin-dir "$T/hb" | head -n 1 >/dev/null || status=$?
done
out=$(ls -a "$T/h")
check "output closed early still finishes" eval '[ $status = 0 ] && version_is "$T/h/current" 0.1.0-macos26 && [ -z "$(find "$T/h" -maxdepth 1 -name "0.3.0-macos26.*")" ] && [ ! -e "$T/h/.lock" ]'
printf 'corrupt' | dd of="$T/d4/sys1rust-0.4.0-macos26-arm64.tar.gz" bs=1 seek=100 conv=notrunc 2>/dev/null
before=$(state "$P" "$B")
run --from "$T/d4" --prefix "$P" --bin-dir "$B"
check "bad checksum fails" eval '[ $status = 1 ] && has "install.sh: sys1rust-0.4.0-macos26-arm64.tar.gz has sha256"'
check "bad checksum changes nothing" eval '[ "$(state "$P" "$B")" = "$before" ]'
check "bad checksum leaves no staging dir" eval '[ -z "$(find "$P" -maxdepth 1 -name ".install.*")" ]'
run --from "$T/d4" --prefix "$T/nested-checksum" --bin-dir "$T/nested-checksum/tools/bin"
check "a bad checksum leaves no nested bin parents" eval '[ $status = 1 ] && has "has sha256" && [ ! -e "$T/nested-checksum" ]'
run --from "$T/d1" --prefix "$T/nested-checksum" --bin-dir "$T/nested-checksum/tools/bin"
check "a nested bin install retries after a bad checksum" eval '[ $status = 0 ] && version_is "$T/nested-checksum/current" 0.1.0-macos26 && "$T/nested-checksum/tools/bin/sys1rust" --version >/dev/null'
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
python3 "$T/server.py" "$W" "$T/server-port" >"$T/server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 600); do
  [ -s "$T/server-port" ] && break
  kill -0 "$SERVER_PID" 2>/dev/null || break
  sleep 0.1
done
if [ ! -s "$T/server-port" ]; then
  echo "The fixture release server did not report its port within 60 s" >&2
  cat "$T/server.log" >&2
  exit 1
fi
URL=http://127.0.0.1:$(cat "$T/server-port")/releases
run SYS1RUST_RELEASES_URL="$URL" -- --prefix "$T/u" --bin-dir "$T/u/bin"
check "latest release" eval '[ $status = 0 ] && version_is "$T/u/current" 0.2.0-macos26 && has "Downloading $URL/download/v0.2.0/sys1rust-0.2.0-macos26-arm64.tar.gz"'
run SYS1RUST_RELEASES_URL="$URL" -- --prefix "$T/u" --bin-dir "$T/u/bin"
check "rerun on the latest skips the download" eval '[ $status = 0 ] && has "already in" && ! has Downloading'
run SYS1RUST_RELEASES_URL="$URL" -- --version v0.3.0-rc1 --prefix "$T/u" --bin-dir "$T/u/bin"
check "--version picks a release candidate" eval '[ $status = 0 ] && version_is "$T/u/current" 0.3.0-rc1-macos26'
run SYS1RUST_RELEASES_URL="$URL" -- --version 0.2.0 --prefix "$T/u" --bin-dir "$T/u/bin"
check "--version without v rolls back to previous" eval '[ $status = 0 ] && version_is "$T/u/current" 0.2.0-macos26 && version_is "$T/u/previous" 0.3.0-rc1-macos26'
run SYS1RUST_RELEASES_URL="$URL" -- --version v9.9.9 --prefix "$T/u" --bin-dir "$T/u/bin"
check "missing tag" eval '[ $status = 1 ] && has "install.sh: there is no release v9.9.9"'
run SYS1RUST_RELEASES_URL="$URL" -- --version v9.9.9 --prefix "$T/u3" --bin-dir "$T/u3/bin"
check "a failed first install leaves no prefix" eval '[ $status = 1 ] && [ ! -e "$T/u3" ]'
run SYS1RUST_RELEASES_URL="$URL" -- --version v9.9.9 --prefix "$T/nested-missing" --bin-dir "$T/nested-missing/tools/bin"
check "a missing release leaves no nested bin parents" eval '[ $status = 1 ] && has "there is no release v9.9.9" && [ ! -e "$T/nested-missing" ]'
run SYS1RUST_RELEASES_URL="$URL" -- --version v0.2.0 --prefix "$T/nested-missing" --bin-dir "$T/nested-missing/tools/bin"
check "a nested bin install retries after a missing release" eval '[ $status = 0 ] && version_is "$T/nested-missing/current" 0.2.0-macos26 && "$T/nested-missing/tools/bin/sys1rust" --version >/dev/null'
run SYS1RUST_RELEASES_URL="$URL" -- --version v0.5.0 --prefix "$T/u" --bin-dir "$T/u/bin"
check "release without SHA256SUMS" eval '[ $status = 1 ] && has "install.sh: release v0.5.0 has no SHA256SUMS"'
run SYS1RUST_RELEASES_URL="$URL" FAKE_MACOS=15.5 -- --version v0.2.0 --prefix "$T/u" --bin-dir "$T/u/bin"
check "release without this flavor" eval '[ $status = 1 ] && has "install.sh: release v0.2.0 has no sys1rust-0.2.0-macos14-arm64.tar.gz"'
run SYS1RUST_RELEASES_URL="$URL" -- --version v0.6.0 --prefix "$T/u" --bin-dir "$T/u/bin"
check "a bad download fails" eval '[ $status = 1 ] && has "install.sh: sys1rust-0.6.0-macos26-arm64.tar.gz has sha256" && version_is "$T/u/current" 0.2.0-macos26'
check "a bad download is deleted" eval '[ -z "$(find "$T/u" -name "*.tar.gz")" ] && [ ! -e "$T/u/0.6.0-macos26" ]'
rm "$W/latest-tag"
run SYS1RUST_RELEASES_URL="$URL" -- --prefix "$T/u2" --bin-dir "$T/u2/bin"
check "no release yet" eval '[ $status = 1 ] && has "install.sh: sys1rust has no published release yet"'

# The LaunchAgent, with the fake launchctl.
AGENT=$T/home/Library/LaunchAgents/io.github.krishhgg.sys1rust.plist
LOG=$T/home/Library/Logs/sys1rust.log
S=$T/s
mkdir -p "$T/home/Library/Logs"
echo foreign >"$LOG"
run --from "$T/d1" --prefix "$T/foreign-log" --bin-dir "$T/foreign-log-bin" --service
check "service refuses an unrelated existing log" eval '[ $status = 1 ] && has "already exists without an installer LaunchAgent" && [ "$(cat "$LOG")" = foreign ]'
run --uninstall --prefix "$T/foreign-log" --bin-dir "$T/foreign-log-bin"
rm "$LOG"
run HF_HUB_CACHE="$T/hub" -- --from "$T/d1" --prefix "$S" --bin-dir "$T/sb" --service
check "--service" eval '[ $status = 0 ] && has "sys1rust serve answers on http://127.0.0.1:$PORT" && has "with HF_HUB_CACHE=$T/hub"'
check "plist is valid" plutil -lint -s "$AGENT"
check "plist runs the link" eval '[ "$(plutil -extract ProgramArguments.0 raw "$AGENT")" = "$T/sb/sys1rust" ] && [ "$(plutil -extract ProgramArguments.1 raw "$AGENT")" = serve ]'
check "plist passes HF_HUB_CACHE and LAYA_PORT" eval '[ "$(plutil -extract EnvironmentVariables.HF_HUB_CACHE raw "$AGENT")" = "$T/hub" ] && [ "$(plutil -extract EnvironmentVariables.LAYA_PORT raw "$AGENT")" = "$PORT" ]'
check "plist restarts on failure only" eval '[ "$(plutil -extract KeepAlive.SuccessfulExit raw "$AGENT")" = false ] && [ "$(plutil -extract RunAtLoad raw "$AGENT")" = true ]'
check "plist logs" eval '[ "$(plutil -extract StandardErrorPath raw "$AGENT")" = "$LOG" ]'
: >"$T/launchd/calls"
run HF_HUB_CACHE=./cache HF_HOME=./hf XDG_CACHE_HOME=./xdg -- --from "$T/d1" --prefix "$S" --bin-dir "$T/sb" --service
check "relative service caches become absolute" eval '[ $status = 0 ] && [ "$(plutil -extract EnvironmentVariables.HF_HUB_CACHE raw "$AGENT")" = "$PWD/./cache" ] && [ "$(plutil -extract EnvironmentVariables.HF_HOME raw "$AGENT")" = "$PWD/./hf" ] && [ "$(plutil -extract EnvironmentVariables.XDG_CACHE_HOME raw "$AGENT")" = "$PWD/./xdg" ]'
run HF_HUB_CACHE="$T/hub" -- --from "$T/d1" --prefix "$S" --bin-dir "$T/sb" --service
check "--service rerun reloads" eval '[ $status = 0 ] && grep -q "^bootout gui/$(id -u)/io.github.krishhgg.sys1rust$" "$T/launchd/calls" && grep -q "^bootstrap gui/$(id -u) $AGENT$" "$T/launchd/calls"'
: >"$T/launchd/calls"
run LAYA_PORT="$(free_port)" HF_HUB_CACHE="$T/wrong-cache" FAKE_MODELS_ENV="$T/model-cache" -- --from "$T/d2" --prefix "$S" --bin-dir "$T/sb"
check "an update restarts the service" eval '[ $status = 0 ] && grep -q "^kickstart -k gui/$(id -u)/io.github.krishhgg.sys1rust$" "$T/launchd/calls" && has "Restarting the sys1rust service"'
check "service update probes its recorded cache" eval '[ "$(cat "$T/model-cache")" = "$T/hub" ]'
: >"$T/launchd/calls"
run SYS1RUST_RELEASES_URL="$URL" -- --version v0.2.0 --prefix "$S" --bin-dir "$T/sb"
check "a rerun leaves the service alone" eval '[ $status = 0 ] && ! grep -q kickstart "$T/launchd/calls" && has "The sys1rust service runs this version"'

service_before=$(cat "$T/launchd/pid")
run FAKE_MV_MODE=after-switch -- --from "$T/d3" --prefix "$S" --bin-dir "$T/sb"
check "an interrupted service update keeps a pending restart" eval '[ $status != 0 ] && version_is "$S/current" 0.3.0-macos26 && [ "$(cat "$T/launchd/pid")" = "$service_before" ] && [ -f "$S/.sys1rust-install/service-restart" ]'
run SYS1RUST_RELEASES_URL="$URL" -- --version v0.3.0 --prefix "$S" --bin-dir "$T/sb"
check "retry restarts after current already switched" eval '[ $status = 0 ] && [ "$(cat "$T/launchd/pid")" != "$service_before" ] && has "Restarting the sys1rust service" && [ ! -e "$S/.sys1rust-install/service-restart" ]'
touch "$T/launchd/kick-fail-next"
run --from "$T/d2" --prefix "$S" --bin-dir "$T/sb"
check "a failed restart keeps its pending marker" eval '[ $status = 1 ] && has "kickstart could not restart" && [ -f "$S/.sys1rust-install/service-restart" ]'
rm "$T/launchd/kick-fail-next"
run SYS1RUST_RELEASES_URL="$URL" -- --version v0.2.0 --prefix "$S" --bin-dir "$T/sb"
check "retry recovers a failed restart" eval '[ $status = 0 ] && has "Restarting the sys1rust service" && [ ! -e "$S/.sys1rust-install/service-restart" ]'

# A healthy foreground server must not hide a failed or non-listening LaunchAgent.
for mode in fail-next no-listener-next; do
  kill "$(cat "$T/launchd/pid")"
  rm "$T/launchd/pid"
  python3 -m http.server --bind 127.0.0.1 --directory "$T/launchd/www" "$PORT" >/dev/null 2>&1 &
  OTHER=$!
  for _ in $(seq 600); do
    curl -fs -o /dev/null "http://127.0.0.1:$PORT/health" 2>/dev/null && break
    kill -0 "$OTHER" 2>/dev/null || break
    sleep 0.1
  done
  check "the foreground server answers before $mode" curl -fs -o /dev/null "http://127.0.0.1:$PORT/health"
  touch "$T/launchd/$mode"
  SECONDS=0
  run --from "$T/d2" --prefix "$S" --bin-dir "$T/sb"
  check "a foreground listener cannot hide $mode" eval '[ $status = 1 ] && has "the service exited before it answered /health" && [ $SECONDS -lt 10 ] && [ -f "$S/.sys1rust-install/service-restart" ]'
  kill "$OTHER"
  wait "$OTHER" 2>/dev/null || true
  rm "$T/launchd/$mode"
  run SYS1RUST_RELEASES_URL="$URL" -- --version v0.2.0 --prefix "$S" --bin-dir "$T/sb"
  check "service recovers after $mode" eval '[ $status = 0 ] && [ -s "$T/launchd/pid" ] && [ ! -e "$S/.sys1rust-install/service-restart" ]'
done
# An install and uninstall in another prefix leave this service alone.
: >"$T/launchd/calls"
run --from "$T/d3" --prefix "$T/o" --bin-dir "$T/ob"
check "another prefix does not restart the service" eval '[ $status = 0 ] && ! grep -q kickstart "$T/launchd/calls" && has "Next, start the server"'
run --uninstall --prefix "$T/o" --bin-dir "$T/ob"
check "another prefix keeps the LaunchAgent" eval '[ $status = 0 ] && has "Left the LaunchAgent $AGENT and its log" && [ -f "$AGENT" ] && [ -f "$T/launchd/loaded" ] && [ ! -e "$T/o" ]'

# Uninstall.
run HF_HUB_CACHE="$T/hub" -- --from "$T/d2" --prefix "$S" --bin-dir "$T/sb" --service
mkdir -p "$T/hub/models--convaiinnovations--laya-typed-decisions/blobs"
dd if=/dev/zero of="$T/hub/models--convaiinnovations--laya-typed-decisions/blobs/x" bs=1000000 count=3 2>/dev/null
echo log >"$LOG"
run HF_HUB_CACHE="$T/wrong-cache" LAYA_PORT=invalid -- --uninstall --prefix "$S" --bin-dir "$T/sb"
check "--uninstall" eval '[ $status = 0 ] && has "Stopped the sys1rust service" && [ ! -e "$T/launchd/loaded" ]'
check "--uninstall removes everything" eval '[ ! -e "$AGENT" ] && [ ! -e "$S" ] && [ ! -L "$T/sb/sys1rust" ] && [ ! -e "$LOG" ]'
check "--uninstall keeps the models" eval 'has "Kept the Laya models in the model cache $T/hub, 3 MB" && [ -d "$T/hub/models--convaiinnovations--laya-typed-decisions" ]'
check "health server stopped" eval '! curl -fs --max-time 2 "http://127.0.0.1:$PORT/health"'
run --from "$T/d1" --prefix "$T/n" --bin-dir "$T/n/bin"
run --uninstall --prefix "$T/n" --bin-dir "$T/n/bin"
check "--uninstall removes a bin dir inside the prefix" eval '[ $status = 0 ] && has "Removed the registered install from $T/n" && [ ! -e "$T/n" ]'
run --from "$T/d1" --prefix "$P" --bin-dir "$B"
touch "$P/notes.txt"
ln -sf /usr/bin/true "$B/sys1rust"
run --uninstall --prefix "$P" --bin-dir "$B"
check "--uninstall keeps foreign files" eval '[ $status = 0 ] && has "Any unrelated files stay there" && [ -f "$P/notes.txt" ] && [ ! -e "$P/current" ]'
check "--uninstall keeps a foreign link" eval 'has "Left $B/sys1rust, which no longer matches the installer registry" && [ -L "$B/sys1rust" ]'
check "--uninstall default cache line" eval 'has "$T/home/.cache/huggingface/hub holds no Laya models"'

# A server that exits at once fails --service without waiting out HEALTH_WAIT.
touch "$T/launchd/fail-next"
SECONDS=0
run --from "$T/d1" --prefix "$T/s2" --bin-dir "$T/sb2" --service
check "a failing service is reported" eval '[ $status = 1 ] && has "install.sh: the service exited before it answered /health" && [ $SECONDS -lt 10 ]'
rm "$T/launchd/fail-next"
run --uninstall --prefix "$T/s2" --bin-dir "$T/sb2"

run --from "$T/d1" --prefix "$T/replaced-log" --bin-dir "$T/replaced-log-bin" --service
mv "$LOG" "$LOG.original"
echo replacement >"$LOG"
run --uninstall --prefix "$T/replaced-log" --bin-dir "$T/replaced-log-bin"
check "uninstall preserves a replaced log" eval '[ $status = 0 ] && [ "$(cat "$LOG")" = replacement ]'
rm "$LOG" "$LOG.original"

# Another server on the port stops --service before it loads the LaunchAgent.
OTHER_PORT=$(free_port)
python3 -m http.server --bind 127.0.0.1 --directory "$T/launchd/www" "$OTHER_PORT" >/dev/null 2>&1 &
OTHER=$!
for _ in $(seq 600); do
  curl -fs -o /dev/null "http://127.0.0.1:$OTHER_PORT/health" 2>/dev/null && break
  kill -0 "$OTHER" 2>/dev/null || break
  sleep 0.1
done
run LAYA_PORT="$OTHER_PORT" -- --from "$T/d1" --prefix "$T/s3" --bin-dir "$T/sb3" --service
kill "$OTHER"
wait "$OTHER" 2>/dev/null || true
check "a busy port stops --service" eval '[ $status = 1 ] && has "install.sh: another server answers on 127.0.0.1:$OTHER_PORT" && [ ! -e "$AGENT" ]'


# Wrong prefixes and familiar-looking foreign names never establish ownership.
FOREIGN=$T/foreign
mkdir -p "$FOREIGN/backup-macos14" "$FOREIGN/0.0.1-macos26"
touch "$FOREIGN/current" "$FOREIGN/previous" "$FOREIGN/backup-macos14/keep"
foreign_before=$(state "$FOREIGN")
run --uninstall --prefix "$FOREIGN" --bin-dir "$T/fb"
check "uninstall refuses an unregistered prefix" eval '[ $status = 1 ] && [ "$(state "$FOREIGN")" = "$foreign_before" ]'
run --from "$T/d1" --prefix "$FOREIGN" --bin-dir "$T/fb"
check "install refuses foreign prefix contents" eval '[ $status = 1 ] && has "has no installer registry" && [ "$(state "$FOREIGN")" = "$foreign_before" ]'

run --from "$T/d1" --prefix "$T/owned" --bin-dir "$T/owned-bin"
mkdir -p "$T/owned/backup-macos14" "$T/owned/0.0.1-macos26"
touch "$T/owned/backup-macos14/keep"
run --from "$T/d2" --prefix "$T/owned" --bin-dir "$T/owned-bin"
run --from "$T/d3" --prefix "$T/owned" --bin-dir "$T/owned-bin"
check "pruning preserves foreign version names" eval '[ $status = 0 ] && [ -f "$T/owned/backup-macos14/keep" ] && [ -d "$T/owned/0.0.1-macos26" ]'
rm "$T/owned/current" "$T/owned/previous"
touch "$T/owned/current" "$T/owned/previous"
run --uninstall --prefix "$T/owned" --bin-dir "$T/owned-bin"
check "uninstall preserves replaced links and foreign names" eval '[ $status = 0 ] && [ -f "$T/owned/current" ] && [ -f "$T/owned/previous" ] && [ -f "$T/owned/backup-macos14/keep" ] && [ -d "$T/owned/0.0.1-macos26" ]'
for target in "$T/flink/other" "$T/flink/../outside/bin/sys1rust"; do
  rm -rf "$T/flink" "$T/flink-bin"
  run --from "$T/d1" --prefix "$T/flink" --bin-dir "$T/flink-bin"
  rm "$T/flink-bin/sys1rust"
  ln -s "$target" "$T/flink-bin/sys1rust"
  run --from "$T/d2" --prefix "$T/flink" --bin-dir "$T/flink-bin"
  check "install refuses a foreign link $target" eval '[ $status = 1 ] && link_is "$T/flink-bin/sys1rust" "$target" && version_is "$T/flink/current" 0.1.0-macos26'
  run --uninstall --prefix "$T/flink" --bin-dir "$T/flink-bin"
  check "uninstall preserves foreign link $target" link_is "$T/flink-bin/sys1rust" "$target"
done

# Failure at the directory move leaves the old directory and current usable.
run --from "$T/d1" --prefix "$T/atomic" --bin-dir "$T/atomic-bin"
active=$(readlink "$T/atomic/current")
for mode in fail signal; do
  run FAKE_MV_MODE="$mode" -- --from "$T/d1" --prefix "$T/atomic" --bin-dir "$T/atomic-bin"
  check "a $mode at the directory move keeps current" eval '[ $status != 0 ] && link_is "$T/atomic/current" "$active" && [ -x "$T/atomic/$active/bin/sys1rust" ] && "$T/atomic-bin/sys1rust" --version >/dev/null && [ ! -e "$T/atomic/.lock" ]'
done

# Hold one install at the move, then try both an update and an uninstall.
env -i HOME="$T/home" PATH="$T/fake:/usr/bin:/bin:/usr/sbin:/sbin" TMPDIR="$T" \
  FAKE_LAUNCHD="$T/launchd" FAKE_MV_MODE=pause FAKE_MV_READY="$T/ready" FAKE_MV_RESUME="$T/resume" \
  "$SH" "$ROOT/install.sh" --from "$T/d2" --prefix "$T/atomic" --bin-dir "$T/atomic-bin" >"$T/overlap.log" 2>&1 &
INSTALL_PID=$!
for _ in $(seq 600); do
  [ -e "$T/ready" ] && break
  kill -0 "$INSTALL_PID" 2>/dev/null || break
  sleep 0.1
done
check "the first install holds the lock" test -f "$T/ready"
run --from "$T/d3" --prefix "$T/atomic" --bin-dir "$T/atomic-bin"
check "an overlapping update refuses the lock" eval '[ $status = 1 ] && has "another installer is running" && link_is "$T/atomic/current" "$active"'
run --uninstall --prefix "$T/atomic" --bin-dir "$T/atomic-bin"
check "an overlapping uninstall refuses the lock" eval '[ $status = 1 ] && has "another installer is running" && link_is "$T/atomic/current" "$active"'
touch "$T/resume"
status=0
wait "$INSTALL_PID" || status=$?
INSTALL_PID=
check "the serialized update leaves both versions usable" eval '[ $status = 0 ] && version_is "$T/atomic/current" 0.2.0-macos26 && link_is "$T/atomic/previous" "$active" && "$T/atomic-bin/sys1rust" --version >/dev/null'
mkdir "$T/atomic/.lock"
echo 99999999 >"$T/atomic/.lock/pid"
run --from "$T/d3" --prefix "$T/atomic" --bin-dir "$T/atomic-bin"
check "a stale lock has a recovery command" eval '[ $status = 1 ] && has "stale installer lock" && has "rmdir" && version_is "$T/atomic/current" 0.2.0-macos26'
rm "$T/atomic/.lock/pid"
rmdir "$T/atomic/.lock"
run --from "$T/d3" --prefix "$T/atomic" --bin-dir "$T/atomic-bin"
check "install resumes after stale lock recovery" eval '[ $status = 0 ] && version_is "$T/atomic/current" 0.3.0-macos26'

run FAKE_MV_MODE=after-switch -- --from "$T/d2" --prefix "$T/atomic" --bin-dir "$T/atomic-bin"
switched=$(readlink "$T/atomic/current")
check "an interruption after current switches leaves a working binary" eval '[ $status != 0 ] && version_is "$T/atomic/current" 0.2.0-macos26 && "$T/atomic-bin/sys1rust" --version >/dev/null && [ ! -e "$T/atomic/.lock" ]'
run SYS1RUST_RELEASES_URL="$URL" -- --version v0.2.0 --prefix "$T/atomic" --bin-dir "$T/atomic-bin"
check "a rerun recognizes an interrupted link registry update" eval '[ $status = 0 ] && link_is "$T/atomic/current" "$switched"'

# Two first installs have different prefixes but share the absent default bin link.
rm "$T/ready" "$T/resume"
env -i HOME="$T/home" PATH="$T/fake:/usr/bin:/bin:/usr/sbin:/sbin" TMPDIR="$T" LAYA_PORT="$PORT" \
  FAKE_LAUNCHD="$T/launchd" FAKE_MV_MODE=pause FAKE_MV_READY="$T/ready" FAKE_MV_RESUME="$T/resume" \
  "$SH" "$ROOT/install.sh" --from "$T/d1" --prefix "$T/shared-a" --service >"$T/overlap.log" 2>&1 &
INSTALL_PID=$!
for _ in $(seq 600); do
  [ -e "$T/ready" ] && break
  kill -0 "$INSTALL_PID" 2>/dev/null || break
  sleep 0.1
done
check "a first install holds shared locks" test -f "$T/ready"
run --from "$T/d2" --prefix "$T/shared-b"
check "different prefixes cannot race an absent bin link" eval '[ $status = 1 ] && has "another installer is running" && [ ! -e "$T/home/.local/bin/sys1rust" ] && [ ! -e "$T/shared-b" ]'
run --from "$T/d2" --prefix "$T/shared-service" --bin-dir "$T/separate-bin" --service
check "different bin directories cannot race the service" eval '[ $status = 1 ] && has "another installer is running" && [ ! -e "$AGENT" ] && [ ! -e "$T/shared-service" ]'
run HOME="$T/another-home" -- --from "$T/d2" --prefix "$T/shared-home" --bin-dir "$T/home/.local/bin"
check "the bin lock also protects callers with another HOME" eval '[ $status = 1 ] && has "another installer is running" && [ ! -e "$T/home/.local/bin/sys1rust" ]'
touch "$T/resume"
status=0
wait "$INSTALL_PID" || status=$?
INSTALL_PID=
check "the first shared install finishes without replacement" eval '[ $status = 0 ] && link_is "$T/home/.local/bin/sys1rust" "$T/shared-a/current/bin/sys1rust" && [ "$(plutil -extract Sys1rustInstallPrefix raw "$AGENT")" = "$T/shared-a" ]'
run --from "$T/d2" --prefix "$T/shared-b"
check "a later install refuses the other prefix bin link" eval '[ $status = 1 ] && has "not this installer" && link_is "$T/home/.local/bin/sys1rust" "$T/shared-a/current/bin/sys1rust"'
run --uninstall --prefix "$T/shared-a"
check "shared locks are removed after completion" eval '[ $status = 0 ] && [ ! -e "$T/home/.local/bin/.sys1rust-install.lock" ] && [ ! -e "$T/home/Library/LaunchAgents/.io.github.krishhgg.sys1rust.install.lock" ]'

# Evaluate the printed PATH assignment in a fresh shell. Directory text stays literal.
WEIRD="$T/bin space' dollar\$ backtick\` quote\" backslash\\"
run --from "$T/d1" --prefix "$T/quoted" --bin-dir "$WEIRD"
path_line=$(printf '%s\n' "$out" | sed -n 's/^  export PATH=/export PATH=/p')
quoted_path=$(env -i PATH=/usr/bin:/bin /bin/sh -c "$path_line"'; printf "%s" "$PATH"')
check "printed PATH preserves shell metacharacters" test "$quoted_path" = "$WEIRD:/usr/bin:/bin"
serve_line=$(printf '%s\n' "$out" | sed -n 's/^Next, start the server: //p')
version_line=${serve_line% serve}' --version'
quoted_version=$(env -i PATH=/usr/bin:/bin /bin/sh -c "$version_line")
check "printed binary command preserves shell metacharacters" test "$quoted_version" = 'sys1rust 0.1.0 (MLX 0.32.2, macos26 build)'

echo "$SH: $passed passed, $failed failed"
[ "$failed" = 0 ]
