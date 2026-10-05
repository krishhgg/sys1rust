#!/bin/bash
# Print the Homebrew formula for a release:
#   packaging/render_formula.sh TAG SHA256_MACOS26 SHA256_MACOS14 [BASE_URL]
# BASE_URL defaults to the GitHub release of TAG. A file:// URL tests a local build.
set -euo pipefail
TAG=$1
S26=$2
S14=$3
BASE=${4:-https://github.com/krishhgg/sys1rust/releases/download/$TAG}
VERSION=${TAG#v}
VERSION=${VERSION%%-rc*}
for s in "$S26" "$S14"; do
  [[ $s =~ ^[0-9a-f]{64}$ ]] || { echo "render_formula.sh: not a sha256: '$s'" >&2; exit 1; }
done
sed -e "s|@VERSION@|$VERSION|g" -e "s|@BASE_URL@|$BASE|g" \
    -e "s|@SHA256_MACOS26@|$S26|g" -e "s|@SHA256_MACOS14@|$S14|g" \
    "$(dirname "$0")/sys1rust.rb.in"
