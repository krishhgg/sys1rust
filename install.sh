#!/bin/sh
# Installs, updates or removes sys1rust on an Apple silicon Mac, without sudo. It also runs
# piped, as `curl -fsSL https://raw.githubusercontent.com/krishhgg/sys1rust/main/install.sh | sh`.
# It downloads the release bundle for this macOS version, checks it against the release's
# SHA256SUMS, unpacks it into a unique PREFIX/<version>-<flavor>.<id>/ and points PREFIX/current and
# BIN_DIR/sys1rust at it. A rerun updates to the latest release, keeps the version before it as
# PREFIX/previous and deletes older ones. --service runs `sys1rust serve` as a LaunchAgent.
# install.sh never downloads a model. `sys1rust pull` and the first `sys1rust serve` do that.
#
# curl sets no com.apple.quarantine attribute on its downloads (xattr -l on a file that curl
# downloaded on macOS 26.2 lists only com.apple.provenance), so Gatekeeper never checks the
# unpacked binary and install.sh needs no xattr step.
#
# The whole script is functions until the last line, so a piped copy that arrives cut short
# installs nothing.
set -eu

REPO_URL=https://github.com/krishhgg/sys1rust
# Tests point this at a local server.
RELEASES_URL=${SYS1RUST_RELEASES_URL:-$REPO_URL/releases}
LABEL=io.github.krishhgg.sys1rust
# How long --service and a service restart wait for /health.
HEALTH_WAIT=60

usage() {
  cat <<'EOF'
Usage: install.sh [options]

Installs sys1rust from its GitHub releases, or updates it to the latest release.

  --version vX.Y.Z[-rcN]  install this release instead of the latest one
  --from DIR              install the tarball and SHA256SUMS in DIR instead of downloading
  --prefix DIR            where versions go (default ~/.local/share/sys1rust)
  --bin-dir DIR           where the sys1rust link goes (default ~/.local/bin)
  --service               also run `sys1rust serve` at login, as a LaunchAgent
  --uninstall             remove sys1rust, its link, its LaunchAgent and its log
  --help                  print this help
EOF
}

# A reader that closes the output early, such as head, must not stop an install halfway, so
# main ignores SIGPIPE and say ignores the write error. External printf keeps Bash 3.2's
# failed output buffer out of later command substitutions.
say() { /usr/bin/printf '%s\n' "$*" 2>/dev/null || true; }
die() { printf 'install.sh: %s\n' "$*" >&2; exit 1; }
usage_error() { printf 'install.sh: %s. install.sh --help lists the options.\n' "$*" >&2; exit 2; }

# Exits 0 when dotted version $1 is at least $2. Compares 3 numeric parts, so 26.10 is above
# 26.2 and 26 means 26.0.0.
ver_ge() {
  printf '%s\n%s\n' "$1" "$2" | awk -F. '
    NR == 1 { for (i = 1; i <= 3; i++) a[i] = $i + 0 }
    NR == 2 { for (i = 1; i <= 3; i++) b[i] = $i + 0 }
    END {
      for (i = 1; i <= 3; i++) { if (a[i] > b[i]) exit 0; if (a[i] < b[i]) exit 1 }
      exit 0
    }'
}

valid_tag() { printf '%s\n' "$1" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+(-rc[0-9]+)?$'; }

# Prints $1 as an absolute path without trailing slashes. It resolves no symlinks, so install
# and uninstall compute the same path even after the directory is gone.
abs_path() {
  case $1 in
    /*) p=$1 ;;
    *) p=$PWD/$1 ;;
  esac
  while [ "$p" != / ] && [ "${p%/}" != "$p" ]; do p=${p%/}; done
  printf '%s\n' "$p"
}

# The Hugging Face hub cache, found as sys1rust finds it.
hub_dir() {
  if [ -n "${HF_HUB_CACHE:-}" ]; then
    say "$HF_HUB_CACHE"
  elif [ -n "${HF_HOME:-}" ]; then
    say "$HF_HOME/hub"
  elif [ -n "${XDG_CACHE_HOME:-}" ]; then
    say "$XDG_CACHE_HOME/huggingface/hub"
  else
    say "$HOME/.cache/huggingface/hub"
  fi
}

xml() { printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'; }

shell_quote() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }

# The registry records the paths this installer created. Never infer ownership from a
# filename. A marker inside each version also prevents deletion of a replacement directory.
check_prefix() {
  [ ! -L "$prefix" ] || die "$prefix is a symlink. Pick a directory for --prefix"
  if [ -e "$owner" ] || [ -L "$owner" ]; then
    if [ ! -d "$owner" ] || [ -L "$owner" ] ||
      [ "$(cat "$owner/format" 2>/dev/null)" != 'sys1rust installer 1' ]; then
      die "$prefix has an invalid installer registry"
    fi
    if [ ! -d "$owner/versions" ] || [ -L "$owner/versions" ]; then die "$owner has no version registry"; fi
    [ "$(cat "$owner/bin-link" 2>/dev/null)" = "$link" ] ||
      die "$prefix belongs to another --bin-dir. Use the bin directory recorded in $owner/bin-link"
  elif [ -d "$prefix" ]; then
    for entry in "$prefix"/* "$prefix"/.[!.]* "$prefix"/..?*; do
      [ -e "$entry" ] || [ -L "$entry" ] || continue
      [ "$entry" = "$lock" ] && [ "$locked" = 1 ] && continue
      die "$prefix is not empty and has no installer registry. Pick an empty --prefix"
    done
  elif [ -e "$prefix" ]; then
    die "$prefix is not a directory"
  fi
}

acquire_lock() {
  if ! mkdir "$1" 2>/dev/null; then
    lock_pid=$(cat "$1/pid" 2>/dev/null || true)
    case $lock_pid in
      '' | *[!0-9]*) die "another installer is acquiring $1. Retry when it finishes" ;;
    esac
    if kill -0 "$lock_pid" 2>/dev/null; then
      die "another installer is running at $1 (PID $lock_pid)"
    fi
    # Automatic removal could race another process that has already replaced this lock.
    die "a stale installer lock remains at $1 (PID $lock_pid). After checking no installer runs, remove it with: rm -f $(shell_quote "$1/pid") && rmdir $(shell_quote "$1")"
  fi
  case $1 in
    "$lock") locked=1 ;;
    "$shared_lock") shared_locked=1 ;;
    "$bin_lock") bin_locked=1 ;;
  esac
  printf '%s\n' "$$" >"$1/pid"
}

release_lock() {
  if [ "$2" = 1 ] && [ "$(cat "$1/pid" 2>/dev/null)" = "$$" ]; then
    rm -f "$1/pid"
    rmdir "$1" 2>/dev/null || true
  fi
}

create_owner() {
  [ ! -d "$owner" ] || return 0
  registry=$stage/registry
  mkdir "$registry" "$registry/versions"
  printf '%s\n' 'sys1rust installer 1' >"$registry/format"
  printf '%s\n' "$link" >"$registry/bin-link"
  mv "$registry" "$owner"
}

owned_version() {
  printf '%s\n' "$1" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+(-rc[0-9]+)?-macos(14|26)\.[[:alnum:]]{6}$' &&
    [ -f "$owner/versions/$1" ] && [ ! -L "$prefix/$1" ] &&
    [ "$(cat "$prefix/$1/.sys1rust-owned" 2>/dev/null)" = "$1" ]
}

check_owned_link() {
  if [ -e "$1" ] || [ -L "$1" ]; then
    link_matches "$1" "$2" || die "$1 is not this installer's link. Move it away before installing"
  fi
}

check_bin_link() {
  if [ -e "$link" ] && [ ! -L "$link" ]; then
    die "$link exists and is not a link. Move it away or pick another --bin-dir"
  fi
  check_owned_link "$link" "$owner/bin-target"
}

link_matches() {
  [ -L "$1" ] && {
    [ "$(readlink "$1")" = "$(cat "$2" 2>/dev/null)" ] ||
    [ "$(readlink "$1")" = "$(cat "$2.next" 2>/dev/null)" ]
  }
}

record_link() {
  # Keep both possible targets registered until the atomic switch finishes. A signal
  # between the switch and the registry update still leaves a recognized working link.
  printf '%s\n' "$1" >"$3.next"
  swap_link "$1" "$2"
  mv -f "$3.next" "$3"
}

remove_owned_link() {
  if link_matches "$1" "$2"; then
    rm -f "$1"
    say "Removed $1"
  elif [ -e "$1" ] || [ -L "$1" ]; then
    say "Left $1, which no longer matches the installer registry"
  fi
}

# Points symlink $2 at $1 in 1 rename, so readers see the old target or the new one and never
# a missing link. mv -h renames over a link to a directory instead of moving into it.
swap_link() {
  next_link=$2.new.$$
  if [ -e "$next_link" ] || [ -L "$next_link" ]; then die "$next_link already exists"; fi
  pending_link=$next_link
  ln -s "$1" "$pending_link"
  mv -fh "$pending_link" "$2"
  pending_link=
}

# fetch URL FILE: downloads URL to FILE and sets http_code. curl retries transient failures
# twice and fails on HTTP errors.
fetch() {
  http_code=000
  http_code=$(curl -fsL --retry 2 --connect-timeout 20 -o "$2" -w '%{http_code}' "$1")
}

check_mac() {
  os_name=$(uname -s)
  [ "$os_name" = Darwin ] || die "sys1rust runs only on macOS, and this system is $os_name"
  # sysctl.proc_translated is 1 under Rosetta, where uname -m says x86_64 on an arm64 Mac.
  if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || true)" = 1 ]; then
    die "this shell runs under Rosetta. Run install.sh from a native arm64 terminal"
  fi
  arch=$(uname -m)
  [ "$arch" = arm64 ] || die "sys1rust needs an Apple silicon (arm64) Mac, and this one is $arch"
  macos=$(sw_vers -productVersion)
  case $macos in
    '' | *[!0-9.]*) die "could not read the macOS version, sw_vers printed '$macos'" ;;
  esac
  ver_ge "$macos" 14.0 || die "sys1rust needs macOS 14.0 or later, and this Mac runs $macos"
  # MLX's build for macOS 26.2 and later is about 3x faster on the M5 than its macOS 14 build.
  if ver_ge "$macos" 26.2; then flavor=macos26; else flavor=macos14; fi
}

# Sets tag to the latest published release that is not a prerelease. GitHub redirects
# /releases/latest to /releases/tag/<tag>, or to /releases when there is none. That page is
# not the API, so its rate limit does not apply.
latest_tag() {
  rc=0
  out=$(curl -s --retry 2 --connect-timeout 20 -o /dev/null -w '%{http_code} %{redirect_url}' \
    "$RELEASES_URL/latest") || rc=$?
  [ "$rc" = 0 ] || die "could not reach $RELEASES_URL/latest (curl exit $rc)"
  code=${out%% *}
  location=${out#* }
  case $location in
    */releases/tag/?*) tag=${location##*/releases/tag/} ;;
    */releases | */releases/) die "sys1rust has no published release yet. See $REPO_URL for a source build" ;;
    *) die "could not find the latest release, $RELEASES_URL/latest answered HTTP $code" ;;
  esac
  valid_tag "$tag" || die "the latest release has tag '$tag', which is not vX.Y.Z"
}

# Downloads or copies the bundle for $tag into $stage, checks it and unpacks it. Sets new to
# the unpacked folder.
get_bundle() {
  base=${tag#v}
  base=${base%%-rc*}
  tarball=sys1rust-$base-$flavor-arm64.tar.gz
  if [ -n "$from" ]; then
    sums=$from/SHA256SUMS
    src=$from/$tarball
    [ -f "$sums" ] || die "$from has no SHA256SUMS"
  else
    url=$RELEASES_URL/download/$tag
    sums=$stage/SHA256SUMS
    src=$stage/$tarball
    if ! fetch "$url/SHA256SUMS" "$sums"; then
      [ "$http_code" = 404 ] || die "could not download $url/SHA256SUMS (HTTP $http_code)"
      if curl -fsL --retry 2 -o /dev/null "$RELEASES_URL/tag/$tag"; then
        die "release $tag has no SHA256SUMS"
      fi
      die "there is no release $tag. The releases are at $REPO_URL/releases"
    fi
    say "Downloading $url/$tarball"
    if ! fetch "$url/$tarball" "$src"; then
      [ "$http_code" = 404 ] && die "release $tag has no $tarball"
      die "could not download $url/$tarball (HTTP $http_code)"
    fi
  fi
  # sha256sum writes "<hash>  <name>", or "<hash> *<name>" in binary mode.
  want=$(awk -v f="$tarball" '{ n = $2; sub(/^\*/, "", n) } n == f { print $1; exit }' "$sums")
  printf '%s\n' "$want" | grep -Eq '^[0-9a-f]{64}$' || die "SHA256SUMS has no sha256 for $tarball"
  got=$(shasum -a 256 "$src" | awk '{ print $1 }')
  if [ "$got" != "$want" ]; then
    [ -n "$from" ] || rm -f "$src"
    die "$tarball has sha256 $got, and SHA256SUMS says $want. Nothing changed"
  fi
  tar -xzf "$src" -C "$stage" || die "could not unpack $tarball"
  [ -n "$from" ] || rm -f "$src"
  new=$stage/sys1rust-$base-$flavor-arm64
  [ -x "$new/bin/sys1rust" ] || die "$tarball has no sys1rust-$base-$flavor-arm64/bin/sys1rust"
  # Runs the new binary before anything points at it.
  out=$("$new/bin/sys1rust" --version </dev/null 2>&1) || die "the new sys1rust does not run: $out"
  case $out in
    "sys1rust $base ("*) ;;
    *) die "the binary in $tarball says '$out', not sys1rust $base" ;;
  esac
}

# Removes the staging dir, and the prefix too when a failed first install leaves it empty.
cleanup() {
  if [ -n "$stage" ]; then rm -rf "$stage"; fi
  if [ -n "$pending_link" ] && [ -L "$pending_link" ]; then rm -f "$pending_link"; fi
  if [ -n "$plist_stage" ]; then rm -f "$plist_stage"; fi
  release_lock "$bin_lock" "$bin_locked"
  release_lock "$shared_lock" "$shared_locked"
  release_lock "$lock" "$locked"
  case $bin_dir in "$prefix"/*) rmdir "$bin_dir" 2>/dev/null || true ;; esac
  rmdir "$prefix" 2>/dev/null || true
}

service_loaded() { launchctl print "gui/$uid/$LABEL" >/dev/null 2>&1; }

# Exits 0 when the LaunchAgent runs this install's link. The label is the same for every
# prefix, so an install or uninstall in a scratch prefix leaves another install's service be.
plist_value() { /usr/libexec/PlistBuddy -c "Print :$1" "$plist" 2>/dev/null; }
plist_runs_link() {
  [ -f "$owner/service" ] && [ -f "$plist" ] && [ ! -L "$plist" ] &&
    [ "$(cat "$owner/service")" = "$plist" ] && [ "$(plist_value Label)" = "$LABEL" ] &&
    [ "$(plist_value ProgramArguments:0)" = "$link" ] &&
    [ "$(plist_value Sys1rustInstallPrefix)" = "$prefix" ]
}

read_service_env() {
  for name in HF_HUB_CACHE HF_HOME XDG_CACHE_HOME LAYA_PORT; do
    unset "$name"
    value=$(plist_value "EnvironmentVariables:$name" || true)
    [ -z "$value" ] || export "$name=$value"
  done
  port=${LAYA_PORT:-8000}
}

health_ok() { curl -fs --max-time 2 -o /dev/null "http://127.0.0.1:$port/health"; }

service_pid() {
  launchctl print "gui/$uid/$LABEL" 2>/dev/null |
    awk '$1 == "pid" && $2 == "=" && $3 ~ /^[0-9]+$/ { print $3; exit }'
}

# A foreground server can answer /health while this job fails to bind. Require launchd's
# PID to own the listening socket, and check the PID again after the HTTP request.
service_health_ok() {
  health_pid=$(service_pid)
  [ -n "$health_pid" ] || return 1
  [ "$(lsof -nP -a -p "$health_pid" -iTCP@127.0.0.1:"$port" -sTCP:LISTEN -t 2>/dev/null)" = "$health_pid" ] || return 1
  health_ok && [ "$(service_pid)" = "$health_pid" ]
}

# Exits 0 when the service is not running and its last run failed. A run that kickstart -k
# stopped exits 0 and stays in launchd's report, so a clean exit does not count. launchd's
# report is not a stable format, so this only speeds up a failure that the wait would
# otherwise find at its end.
service_exited() {
  launchctl print "gui/$uid/$LABEL" 2>/dev/null | awk '
    $1 == "state" && $3 == "running" { running = 1 }
    /^[[:space:]]*last exit code = -?[1-9]/ || /^[[:space:]]*last terminating signal/ { failed = 1 }
    END { exit !(failed && !running) }'
}

# Waits up to HEALTH_WAIT seconds for the job's /health. Returns 0 when it answers, 2 when the server
# exited first and 1 on timeout.
wait_health() {
  t=0
  while [ "$t" -lt "$HEALTH_WAIT" ]; do
    service_exited && return 2
    service_health_ok && return 0
    sleep 1
    t=$((t + 1))
  done
  return 1
}

# Exits 0 when the model `sys1rust serve` loads by default is in the cache. Also sets
# model_mb to its size from `sys1rust models`.
model_ready() {
  row=$("$link" models </dev/null 2>/dev/null | awk '$1 == "typed-decisions"') || true
  model_mb=$(printf '%s\n' "$row" | awk '{ print $(NF - 1) }')
  case $model_mb in '' | *[!0-9]*) model_mb=846 ;; esac
  [ "$(printf '%s\n' "$row" | awk '{ print $4 }')" = "downloaded," ]
}

# Waits for the service that launchd just started and reports it. A first start that has to
# download the model may take longer than the wait, and that is no error.
check_service() {
  downloading=0
  if ! model_ready; then
    downloading=1
    say "The first start downloads the typed-decisions model ($model_mb MB) before it answers."
  fi
  rc=0
  wait_health || rc=$?
  if [ "$rc" = 0 ]; then
    rm -f "$owner/service-restart"
    say "sys1rust serve answers on http://127.0.0.1:$port. Its log is $log"
    return 0
  fi
  if [ "$rc" = 1 ] && [ "$downloading" = 1 ]; then
    say "The service is still starting. Follow the download with: tail -f $(shell_quote "$log")"
    say "It answers on http://127.0.0.1:$port/health once the download and load finish."
    return 0
  fi
  [ -f "$log" ] && tail -n 5 "$log" >&2
  if [ "$rc" = 2 ]; then
    die "the service exited before it answered /health. Read $log, and install.sh --uninstall removes the service"
  fi
  die "the service did not answer http://127.0.0.1:$port/health in $HEALTH_WAIT s. Read $log"
}

write_plist() {
  cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>Sys1rustInstallPrefix</key>
  <string>$(xml "$prefix")</string>
  <key>ProgramArguments</key>
  <array>
    <string>$(xml "$link")</string>
    <string>serve</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key>
    <false/>
  </dict>
  <key>ThrottleInterval</key>
  <integer>30</integer>
  <key>ProcessType</key>
  <string>Interactive</string>
  <key>StandardOutPath</key>
  <string>$(xml "$log")</string>
  <key>StandardErrorPath</key>
  <string>$(xml "$log")</string>
EOF
  if [ -n "$service_env" ]; then
    say "  <key>EnvironmentVariables</key>"
    say "  <dict>"
    for name in $service_env; do
      say "    <key>$name</key>"
      say "    <string>$(xml "$(printenv "$name")")</string>"
    done
    say "  </dict>"
  fi
  say "</dict>"
  say "</plist>"
}

install_service() {
  if [ -e "$plist" ] || [ -L "$plist" ]; then
    if [ -L "$plist" ] || [ -z "$(plist_value Sys1rustInstallPrefix || true)" ]; then
      die "$plist is not an installer LaunchAgent. Move it away before using --service"
    fi
  elif service_loaded; then
    die "the LaunchAgent label $LABEL is already loaded without an installer plist"
  elif [ -e "$log" ] || [ -L "$log" ]; then
    die "$log already exists without an installer LaunchAgent. Move it away before using --service"
  fi
  if [ -L "$log" ] || { [ -e "$log" ] && [ ! -f "$log" ]; }; then die "$log is not a regular log file"; fi
  mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
  if service_loaded; then
    launchctl bootout "gui/$uid/$LABEL" 2>/dev/null || true
    # bootout sends SIGTERM, and the server finishes its requests before it exits.
    t=0
    while service_loaded && [ "$t" -lt 20 ]; do
      sleep 1
      t=$((t + 1))
    done
    t=0
    while health_ok && [ "$t" -lt 5 ]; do
      sleep 1
      t=$((t + 1))
    done
  fi
  if health_ok; then
    die "another server answers on 127.0.0.1:$port, maybe a sys1rust serve in a terminal. Stop it and rerun with --service"
  fi
  # launchd starts the service without this shell's environment. These variables choose the
  # model cache and the port, so the service gets them when they are set here. Then it uses
  # the cache that `sys1rust pull` in this shell fills.
  service_env=
  for name in HF_HUB_CACHE HF_HOME XDG_CACHE_HOME LAYA_PORT; do
    value=$(printenv "$name" || true)
    if [ -n "$value" ]; then
      case $name in HF_HUB_CACHE | HF_HOME | XDG_CACHE_HOME) value=$(abs_path "$value") ;; esac
      export "$name=$value"
      service_env="$service_env $name"
    fi
  done
  # KeepAlive restarts the server when it exits with an error or crashes, but not after a
  # clean stop. ThrottleInterval keeps a server that fails at once to 1 start per 30 s.
  # ProcessType Interactive takes away the CPU and I/O throttling that launchd puts on
  # background jobs, which would slow every request.
  touch "$log"
  stat -f '%d:%i' "$log" >"$owner/log-id"
  plist_stage=$(mktemp "$plist.XXXXXX")
  write_plist >"$plist_stage"
  plutil -lint -s "$plist_stage" || die "the LaunchAgent plist is not valid"
  mv -f "$plist_stage" "$plist"
  plist_stage=
  printf '%s\n' "$plist" >"$owner/service"
  launchctl bootstrap "gui/$uid" "$plist" ||
    die "launchctl could not load $plist. A LaunchAgent needs a user logged in to this Mac's desktop"
  say "Installed the LaunchAgent $plist"
  for name in $service_env; do say "  with $name=$(printenv "$name")"; done
  check_service
}

uninstall() {
  [ "$(uname -s)" = Darwin ] || die "sys1rust runs only on macOS"
  [ -d "$owner" ] || die "$prefix has no installer registry. Nothing removed"
  if ! plist_runs_link; then
    say "Left the LaunchAgent $plist and its log, since it runs another sys1rust than $link"
  else
    read_service_env
    if service_loaded; then
      launchctl bootout "gui/$uid/$LABEL" 2>/dev/null || true
      t=0
      while service_loaded && [ "$t" -lt 20 ]; do
        sleep 1
        t=$((t + 1))
      done
      service_loaded && die "launchctl bootout gui/$uid/$LABEL did not stop the service"
      say "Stopped the sys1rust service"
    fi
    if [ -e "$plist" ]; then
      rm -f "$plist"
      say "Removed $plist"
    fi
    if [ -f "$log" ] && [ ! -L "$log" ] && [ "$(stat -f '%d:%i' "$log")" = "$(cat "$owner/log-id" 2>/dev/null)" ]; then
      rm -f "$log"
      say "Removed $log"
    fi
  fi
  remove_owned_link "$link" "$owner/bin-target"
  # A --bin-dir inside the prefix goes with it once it is empty.
  case $bin_dir in
    "$prefix"/*) rmdir "$bin_dir" 2>/dev/null || true ;;
  esac
  if [ -d "$prefix" ]; then
    remove_owned_link "$prefix/current" "$owner/current"
    remove_owned_link "$prefix/previous" "$owner/previous"
    for entry in "$owner/versions"/*; do
      [ -f "$entry" ] || continue
      name=${entry##*/}
      if owned_version "$name"; then rm -rf "${prefix:?}/$name"; fi
      rm -f "$entry"
    done
    rm -f "$owner/format" "$owner/bin-link" "$owner/bin-target" "$owner/bin-target.next" \
      "$owner/current" "$owner/current.next" "$owner/previous" "$owner/previous.next" "$owner/service" "$owner/log-id" "$owner/service-restart"
    rmdir "$owner/versions" "$owner" 2>/dev/null || true
    # cleanup releases the lock and removes the prefix when it has no foreign files.
    say "Removed the registered install from $prefix. Any unrelated files stay there"
  fi
  hub=$(hub_dir)
  set -- "$hub"/models--convaiinnovations--laya*
  if [ -d "$1" ]; then
    # Counts each file once, by inode, since a cache's snapshots link to its blobs and a
    # cache can also link blobs to files that several repos share.
    mb=$(find -L "$@" -type f -exec stat -L -f '%d:%i %z' {} + 2>/dev/null |
      awk '!seen[$1]++ { s += $2 } END { printf "%d", (s + 500000) / 1000000 }')
    say "Kept the Laya models in the model cache $hub, $mb MB. To delete them, run:"
    say "  rm -rf $(shell_quote "$hub")/models--convaiinnovations--laya*"
  else
    say "The model cache $hub holds no Laya models"
  fi
}

main() {
  trap '' PIPE
  version=
  from=
  prefix=
  bin_dir=
  service=0
  remove=0
  while [ $# -gt 0 ]; do
    case $1 in
      --version | --from | --prefix | --bin-dir)
        if [ $# -lt 2 ] || [ -z "$2" ]; then usage_error "$1 needs a value"; fi
        opt=$1
        val=$2
        shift 2
        ;;
      --version=* | --from=* | --prefix=* | --bin-dir=*)
        opt=${1%%=*}
        val=${1#*=}
        shift
        [ -n "$val" ] || usage_error "$opt needs a value"
        ;;
      --service)
        service=1
        shift
        continue
        ;;
      --uninstall)
        remove=1
        shift
        continue
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *) usage_error "unknown option '$1'" ;;
    esac
    case $opt in
      --version) version=$val ;;
      --from) from=$val ;;
      --prefix) prefix=$val ;;
      --bin-dir) bin_dir=$val ;;
    esac
  done
  if [ "$remove" = 1 ] && { [ -n "$version" ] || [ -n "$from" ] || [ "$service" = 1 ]; }; then
    usage_error "--uninstall takes only --prefix and --bin-dir"
  fi
  [ -z "$version" ] || [ -z "$from" ] || usage_error "--version and --from do not go together"
  if [ -n "$version" ]; then
    case $version in v*) ;; *) version=v$version ;; esac
    valid_tag "$version" || usage_error "--version takes a tag such as v0.1.0 or v0.1.0-rc1"
  fi

  prefix=$(abs_path "${prefix:-$HOME/.local/share/sys1rust}")
  bin_dir=$(abs_path "${bin_dir:-$HOME/.local/bin}")
  [ "$prefix" != / ] || die "--prefix cannot be /"
  link=$bin_dir/sys1rust
  uid=$(id -u)
  plist=$HOME/Library/LaunchAgents/$LABEL.plist
  log=$HOME/Library/Logs/sys1rust.log
  port=${LAYA_PORT:-8000}
  if [ "$remove" != 1 ]; then
    case $port in '' | *[!0-9]*) die "LAYA_PORT is '$port', not a port number" ;; esac
    if [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then die "LAYA_PORT must be between 1 and 65535"; fi
  fi

  stage=
  pending_link=
  plist_stage=
  locked=0
  shared_locked=0
  bin_locked=0
  lock=$prefix/.lock
  shared_lock=$HOME/Library/LaunchAgents/.$LABEL.install.lock
  bin_lock=$bin_dir/.sys1rust-install.lock
  owner=$prefix/.sys1rust-install
  check_prefix
  if [ "$remove" = 1 ]; then
    [ -d "$owner" ] || die "$prefix has no installer registry. Nothing removed"
  fi
  mkdir -p "$prefix"
  # Canonicalize the prefix so two paths through parent symlinks use the same lock.
  prefix=$(cd "$prefix" && pwd -P)
  lock=$prefix/.lock
  owner=$prefix/.sys1rust-install
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  acquire_lock "$lock"
  check_prefix
  # Every service uses the same label and plist. Installs take the bin lock after bundle
  # validation below, so a failed first download never creates bin parents inside an
  # unregistered prefix.
  mkdir -p "${shared_lock%/*}"
  acquire_lock "$shared_lock"

  if [ "$remove" = 1 ]; then
    mkdir -p "$bin_dir"
    acquire_lock "$bin_lock"
    uninstall
    return 0
  fi

  check_mac
  # Refuse a foreign link before downloading, then check it again under the bin lock.
  check_bin_link
  check_owned_link "$prefix/current" "$owner/current"
  check_owned_link "$prefix/previous" "$owner/previous"
  if [ -n "$from" ]; then
    [ -d "$from" ] || die "--from $from is not a directory"
    from=$(abs_path "$from")
    set -- "$from"/sys1rust-*-"$flavor"-arm64.tar.gz
    [ -f "$1" ] || die "$from has no sys1rust-<version>-$flavor-arm64.tar.gz"
    [ $# = 1 ] || die "$from has $# $flavor tarballs. Keep 1"
    name=${1##*/}
    name=${name#sys1rust-}
    tag=v${name%-"$flavor"-arm64.tar.gz}
    valid_tag "$tag" || die "cannot read a version from $1"
  elif [ -n "$version" ]; then
    tag=$version
  else
    latest_tag
  fi

  dir=
  if [ -z "$from" ]; then
    for entry in "$(readlink "$prefix/current" 2>/dev/null || true)" \
      "$(readlink "$prefix/previous" 2>/dev/null || true)" "$owner/versions"/*; do
      name=${entry##*/}
      if owned_version "$name" && [ "$(cat "$prefix/$name/.sys1rust-release")" = "$tag-$flavor" ] &&
        "$prefix/$name/bin/sys1rust" --version </dev/null >/dev/null 2>&1; then
        dir=$name
        break
      fi
    done
  fi
  if [ -n "$dir" ]; then
    target=$prefix/$dir
    say "sys1rust $tag for $flavor is already in $target"
  else
    stage=$(mktemp -d "$prefix/.install.XXXXXX")
    get_bundle
    create_owner
    dir=${tag#v}-$flavor.${stage##*.}
    target=$prefix/$dir
    if [ -e "$target" ] || [ -L "$target" ]; then die "$target already exists"; fi
    printf '%s\n' "$dir" >"$new/.sys1rust-owned"
    printf '%s\n' "$tag-$flavor" >"$new/.sys1rust-release"
    : >"$owner/versions/$dir"
  fi

  # The bundle is ready and the prefix has its registry before any bin parents appear.
  # Different HOME values can still share this bin directory, so recheck its link after
  # acquiring the lock and before switching current or changing the bin link.
  mkdir -p "$bin_dir"
  acquire_lock "$bin_lock"
  check_bin_link
  if [ -n "$stage" ]; then
    # The working version stays in place. Only current changes after this move succeeds.
    mv "$new" "$target"
  fi

  old=$(readlink "$prefix/current" 2>/dev/null || true)
  if [ "$old" != "$dir" ]; then
    # Persist the restart before switching current. A retry must restart an old process
    # even when an interrupted invocation already switched the link to this directory.
    if plist_runs_link && service_loaded; then : >"$owner/service-restart"; fi
    # Save the previous link first so interruption after the current switch keeps both.
    if [ -n "$old" ] && owned_version "$old"; then record_link "$old" "$prefix/previous" "$owner/previous"; fi
    record_link "$dir" "$prefix/current" "$owner/current"
  fi
  if [ "$(readlink "$link" 2>/dev/null || true)" != "$prefix/current/bin/sys1rust" ]; then
    record_link "$prefix/current/bin/sys1rust" "$link" "$owner/bin-target"
  fi
  installed=$("$link" --version </dev/null 2>&1) || die "$link --version failed: $installed"
  say "Installed $installed in $target"
  say "Linked $link"

  # Keeps current and previous and deletes older versions.
  previous=$(readlink "$prefix/previous" 2>/dev/null || true)
  for entry in "$owner/versions"/*; do
    [ -f "$entry" ] || continue
    name=${entry##*/}
    if [ "$name" = "$dir" ] || [ "$name" = "$previous" ]; then continue; fi
    if owned_version "$name"; then
      rm -rf "${prefix:?}/$name"
      say "Removed the old version $name"
    fi
    rm -f "$entry"
  done

  run=$(shell_quote "$link")
  case ":$PATH:" in
    *":$bin_dir:"*)
      found=$(command -v sys1rust || true)
      if [ "$found" != "$link" ]; then say "$found comes before $link on PATH, so sys1rust runs that one"; fi
      ;;
    *)
      say "$bin_dir is not on PATH. For zsh, add this line to ~/.zshrc and open a new terminal:"
      say "  export PATH=$(shell_quote "$bin_dir"):\"\$PATH\""
      ;;
  esac

  if [ "$service" = 1 ]; then
    install_service
  elif plist_runs_link && service_loaded; then
    read_service_env
    if [ -f "$owner/service-restart" ]; then
      # launchd starts a job at most once per ThrottleInterval, so this waits up to 30 s
      # when the service started less than 30 s ago.
      say "Restarting the sys1rust service on the new version"
      launchctl kickstart -k "gui/$uid/$LABEL" || die "launchctl kickstart could not restart the service"
      check_service
    else
      check_service
      say "The sys1rust service runs this version. Its log is $log"
    fi
  else
    say "Next, start the server: $run serve"
    if ! model_ready; then
      say "Its first start downloads the typed-decisions model ($model_mb MB). To download it first, run: $run pull typed-decisions"
    fi
  fi
}

main "$@"
