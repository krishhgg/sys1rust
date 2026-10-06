#!/bin/sh
# Installs, updates or removes sys1rust on an Apple silicon Mac, without sudo. It also runs
# piped, as `curl -fsSL https://raw.githubusercontent.com/krishhgg/sys1rust/main/install.sh | sh`.
# It downloads the release bundle for this macOS version, checks it against the release's
# SHA256SUMS, unpacks it into PREFIX/<version>-<flavor>/ and points PREFIX/current and
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
# main ignores SIGPIPE and say ignores the write error.
say() { printf '%s\n' "$*" 2>/dev/null || true; }
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

# Points symlink $2 at $1 in 1 rename, so readers see the old target or the new one and never
# a missing link. mv -h renames over a link to a directory instead of moving into it.
swap_link() {
  rm -f "$2.new.$$"
  ln -s "$1" "$2.new.$$"
  mv -fh "$2.new.$$" "$2"
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
  rmdir "$prefix" 2>/dev/null || true
}

service_loaded() { launchctl print "gui/$uid/$LABEL" >/dev/null 2>&1; }

# Exits 0 when the LaunchAgent runs this install's link. The label is the same for every
# prefix, so an install or uninstall in a scratch prefix leaves another install's service be.
plist_runs_link() { [ -f "$plist" ] && grep -qF "<string>$(xml "$link")</string>" "$plist"; }

health_ok() { curl -fs --max-time 2 -o /dev/null "http://127.0.0.1:$port/health"; }

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

# Waits up to HEALTH_WAIT seconds for /health. Returns 0 when it answers, 2 when the server
# exited first and 1 on timeout.
wait_health() {
  t=0
  while [ "$t" -lt "$HEALTH_WAIT" ]; do
    health_ok && return 0
    service_exited && return 2
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
    say "sys1rust serve answers on http://127.0.0.1:$port. Its log is $log"
    return 0
  fi
  if [ "$rc" = 1 ] && [ "$downloading" = 1 ]; then
    say "The service is still starting. Follow the download with: tail -f $log"
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
    if [ -n "$(printenv "$name" || true)" ]; then service_env="$service_env $name"; fi
  done
  # KeepAlive restarts the server when it exits with an error or crashes, but not after a
  # clean stop. ThrottleInterval keeps a server that fails at once to 1 start per 30 s.
  # ProcessType Interactive takes away the CPU and I/O throttling that launchd puts on
  # background jobs, which would slow every request.
  write_plist >"$plist.new.$$"
  plutil -lint -s "$plist.new.$$" || die "the LaunchAgent plist is not valid"
  mv -f "$plist.new.$$" "$plist"
  launchctl bootstrap "gui/$uid" "$plist" ||
    die "launchctl could not load $plist. A LaunchAgent needs a user logged in to this Mac's desktop"
  say "Installed the LaunchAgent $plist"
  for name in $service_env; do say "  with $name=$(printenv "$name")"; done
  check_service
}

uninstall() {
  [ "$(uname -s)" = Darwin ] || die "sys1rust runs only on macOS"
  if [ -f "$plist" ] && ! plist_runs_link; then
    say "Left the LaunchAgent $plist and its log, since it runs another sys1rust than $link"
  else
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
    if [ -e "$log" ]; then
      rm -f "$log"
      say "Removed $log"
    fi
  fi
  if [ -L "$link" ]; then
    target=$(readlink "$link")
    case $target in
      "$prefix"/*)
        rm -f "$link"
        say "Removed $link"
        ;;
      *) say "Left $link, which points to $target" ;;
    esac
  fi
  # A --bin-dir inside the prefix goes with it once it is empty.
  case $bin_dir in
    "$prefix"/*) rmdir "$bin_dir" 2>/dev/null || true ;;
  esac
  if [ -d "$prefix" ]; then
    # Deletes only what install.sh makes, so a wrong --prefix cannot take other files.
    rm -f "$prefix/current" "$prefix/previous" "$prefix"/current.new.* "$prefix"/previous.new.*
    for d in "$prefix"/*-macos14 "$prefix"/*-macos26 "$prefix"/.install.*; do
      if [ -d "$d" ] && [ ! -L "$d" ]; then rm -rf "$d"; fi
    done
    if rmdir "$prefix" 2>/dev/null; then
      say "Removed $prefix"
    else
      say "Left $prefix, which holds files that install.sh did not make"
    fi
  fi
  hub=$(hub_dir)
  set -- "$hub"/models--convaiinnovations--laya*
  if [ -d "$1" ]; then
    # Counts each file once, by inode, since a cache's snapshots link to its blobs and a
    # cache can also link blobs to files that several repos share.
    mb=$(find -L "$@" -type f -exec stat -L -f '%d:%i %z' {} + 2>/dev/null |
      awk '!seen[$1]++ { s += $2 } END { printf "%d", (s + 500000) / 1000000 }')
    say "Kept the Laya models in the model cache $hub, $mb MB. To delete them, run:"
    say "  rm -rf \"$hub\"/models--convaiinnovations--laya*"
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
  case $port in '' | *[!0-9]*) die "LAYA_PORT is '$port', not a port number" ;; esac

  if [ "$remove" = 1 ]; then
    uninstall
    return 0
  fi

  check_mac
  if [ -e "$link" ] && [ ! -L "$link" ]; then
    die "$link exists and is not a link. Move it away or pick another --bin-dir"
  fi
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

  dir=${tag#v}-$flavor
  target=$prefix/$dir
  mkdir -p "$prefix"
  stage=
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  if [ -z "$from" ] && [ -x "$target/bin/sys1rust" ] &&
    "$target/bin/sys1rust" --version </dev/null >/dev/null 2>&1; then
    say "sys1rust $tag for $flavor is already in $target"
  else
    stage=$(mktemp -d "$prefix/.install.XXXXXX")
    get_bundle
    # The new folder replaces one of the same version. current points at it by name, so it
    # sees the new folder once the second mv is done.
    if [ -e "$target" ] || [ -L "$target" ]; then mv "$target" "$stage/old"; fi
    mv "$new" "$target"
  fi

  old=$(readlink "$prefix/current" 2>/dev/null || true)
  changed=0
  if [ "$old" != "$dir" ]; then
    swap_link "$dir" "$prefix/current"
    if [ -n "$old" ] && [ -d "$prefix/$old" ]; then swap_link "$old" "$prefix/previous"; fi
    changed=1
  fi
  mkdir -p "$bin_dir"
  if [ "$(readlink "$link" 2>/dev/null || true)" != "$prefix/current/bin/sys1rust" ]; then
    swap_link "$prefix/current/bin/sys1rust" "$link"
  fi
  installed=$("$link" --version </dev/null 2>&1) || die "$link --version failed: $installed"
  say "Installed $installed in $target"
  say "Linked $link"

  # Keeps current and previous and deletes older versions.
  previous=$(readlink "$prefix/previous" 2>/dev/null || true)
  for d in "$prefix"/*-macos14 "$prefix"/*-macos26; do
    if [ ! -d "$d" ] || [ -L "$d" ]; then continue; fi
    name=${d##*/}
    if [ "$name" = "$dir" ] || [ "$name" = "$previous" ]; then continue; fi
    rm -rf "$d"
    say "Removed the old version $name"
  done

  run=sys1rust
  case ":$PATH:" in
    *":$bin_dir:"*)
      found=$(command -v sys1rust || true)
      if [ "$found" != "$link" ]; then say "$found comes before $link on PATH, so sys1rust runs that one"; fi
      ;;
    *)
      run=$link
      case $bin_dir in
        "$HOME"/*) path_entry="\$HOME/${bin_dir#"$HOME"/}" ;;
        *) path_entry=$bin_dir ;;
      esac
      say "$bin_dir is not on PATH. For zsh, add this line to ~/.zshrc and open a new terminal:"
      say "  export PATH=\"$path_entry:\$PATH\""
      ;;
  esac

  if [ "$service" = 1 ]; then
    install_service
  elif plist_runs_link && service_loaded; then
    if [ "$changed" = 1 ]; then
      # launchd starts a job at most once per ThrottleInterval, so this waits up to 30 s
      # when the service started less than 30 s ago.
      say "Restarting the sys1rust service on the new version"
      launchctl kickstart -k "gui/$uid/$LABEL" || die "launchctl kickstart could not restart the service"
      check_service
    else
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
