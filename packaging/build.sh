#!/bin/bash
# Build one release bundle: packaging/build.sh macos14|macos26 OUT_DIR
# Downloads the pinned MLX wheel (packaging/mlx-wheels.lock), builds sys1rust against it with the
# pinned Rust (packaging/rust-toolchain-version) for the flavor's macOS target, makes the binary
# find lib/ next to bin/ (@executable_path/../lib), signs it ad hoc, adds the license files,
# checks the result and writes OUT_DIR/sys1rust-<version>-<flavor>-arm64.tar.gz.
set -euo pipefail

fail() { echo "build.sh: $*" >&2; exit 1; }

# Prints each LC_RPATH of a Mach-O file on its own line. otool prints "path <dir> (offset N)".
rpaths_of() {
  otool -l "$1" | awk '/cmd LC_RPATH/ {getline; getline; sub(/^ *path /, ""); sub(/ \(offset [0-9]+\)$/, ""); print}'
}

# Third-party code that MLX 0.32.2 compiles into lib/ and whose license asks for its notice in
# binary copies. packaging/licenses/NOTICE describes each one. fmt 12.1.0 (MIT with an exception
# for object code) and ThreadPool (zlib, notice for source copies only) need none.
MLX_THIRD_PARTY=(nlohmann-json-3.11.3.txt gguf-tools-8fa6eb6.txt fp16-ba1d31f.txt metal-cpp-26.txt
  pocketfft-mlx-0.32.2.txt v8-small-vector-mlx-0.32.2.txt expm1f-mlx-0.32.2.txt
  thrust-cexpf-mlx-0.32.2.txt)

FLAVOR=${1:-}
OUT_ARG=${2:-}
case "$FLAVOR" in
  macos14) TARGET=14.0 ;;
  macos26) TARGET=26.2 ;;
  *) echo "usage: $0 macos14|macos26 OUT_DIR" >&2; exit 2 ;;
esac
[ -n "$OUT_ARG" ] || { echo "usage: $0 macos14|macos26 OUT_DIR" >&2; exit 2; }
mkdir -p "$OUT_ARG"
OUT=$(cd "$OUT_ARG" && pwd)
ROOT=$(cd "$(dirname "$0")/.." && pwd)

# read exits 1 when the lock has no line for the flavor, so let the check below report it.
read -r URL SHA < <(awk -v f="$FLAVOR" '$1 == f {print $2, $3}' "$ROOT/packaging/mlx-wheels.lock") || true
[ -n "${URL:-}" ] || fail "no wheel for $FLAVOR in packaging/mlx-wheels.lock"
VERSION=$(sed -n 's/^version = "\(.*\)"$/\1/p' "$ROOT/runtime/Cargo.toml")
[ -n "$VERSION" ] || fail "no workspace version in runtime/Cargo.toml"
NAME="sys1rust-$VERSION-$FLAVOR-arm64"
WORK="$ROOT/packaging/.work/$FLAVOR"
mkdir -p "$WORK"

# The pinned Rust. The release workflow installs the version in the same file.
RUST=$(cat "$ROOT/packaging/rust-toolchain-version")
for tool in rustc cargo; do
  got=$("$tool" --version | awk '{print $2}')
  [ "$got" = "$RUST" ] || fail "$tool is $got, but the release build needs $RUST" \
    "(packaging/rust-toolchain-version). Run 'rustup toolchain install $RUST' and build with RUSTUP_TOOLCHAIN=$RUST."
done

# 1. The pinned wheel.
WHEEL="$WORK/$(basename "$URL")"
if [ ! -f "$WHEEL" ]; then
  curl -fsSL --retry 3 -o "$WHEEL.part" "$URL"
  mv "$WHEEL.part" "$WHEEL"
fi
echo "$SHA  $WHEEL" | shasum -a 256 -c - >/dev/null || fail "sha256 mismatch for $WHEEL"
rm -rf "$WORK/wheel"
unzip -q "$WHEEL" -d "$WORK/wheel"
export MLX_SYS_PREBUILT_DIR="$WORK/wheel/mlx"

# 2. The binary, for the flavor's oldest macOS.
export MACOSX_DEPLOYMENT_TARGET="$TARGET"
export SYS1_BUILD_FLAVOR="$FLAVOR"
export CARGO_TARGET_DIR="$WORK/target"
cargo build --release --locked --manifest-path "$ROOT/runtime/Cargo.toml" -p sys1rust

# 3. The bundle.
STAGE="$WORK/$NAME"
rm -rf "$STAGE"
mkdir -p "$STAGE/bin" "$STAGE/lib" "$STAGE/licenses"
BIN="$STAGE/bin/sys1rust"
cp "$CARGO_TARGET_DIR/release/sys1rust" "$BIN"
cp "$MLX_SYS_PREBUILT_DIR"/lib/{libmlx.dylib,libjaccl.dylib,mlx.metallib} "$STAGE/lib/"
cp "$ROOT/LICENSE" "$STAGE/"
cp "$WORK"/wheel/mlx_metal-*.dist-info/licenses/LICENSE "$STAGE/licenses/MLX-LICENSE"
for f in "${MLX_THIRD_PARTY[@]}"; do
  [ -f "$ROOT/packaging/licenses/$f" ] || fail "packaging/licenses/$f is missing"
  grep -qF "licenses/$f" "$ROOT/packaging/licenses/NOTICE" || fail "packaging/licenses/NOTICE does not name $f"
  cp "$ROOT/packaging/licenses/$f" "$STAGE/licenses/"
done
{ cat "$ROOT/NOTICE"; echo; cat "$ROOT/packaging/licenses/NOTICE"; } > "$STAGE/NOTICE"
python3 "$ROOT/packaging/third_party.py" > "$STAGE/licenses/THIRD_PARTY.md"

# 4. Find lib/ next to bin/, not the build machine's wheel directory.
rpaths_of "$BIN" | while IFS= read -r rpath; do
  install_name_tool -delete_rpath "$rpath" "$BIN"
done
install_name_tool -add_rpath @executable_path/../lib "$BIN"
codesign --force -s - "$BIN"

# 5. Checks.
bad=$(otool -L "$BIN" | tail -n +2 | awk '{print $1}' | grep -Ev '^(@rpath/|/usr/lib/|/System/)' || true)
[ -z "$bad" ] || fail "the binary links outside the bundle and the OS: $bad"
rpaths=$(rpaths_of "$BIN")
[ "$rpaths" = "@executable_path/../lib" ] || fail "rpaths are: $rpaths"
for f in "$BIN" "$STAGE/lib/libmlx.dylib" "$STAGE/lib/libjaccl.dylib"; do
  minos=$(vtool -show-build "$f" | awk '/minos/ {print $2; exit}')
  python3 -c 'import sys; a, b = (tuple(map(int, v.split("."))) for v in sys.argv[1:]); sys.exit(a > b)' \
    "$minos" "$TARGET" || fail "$f needs macOS $minos, above the $FLAVOR target $TARGET"
done
codesign --verify --strict "$BIN" || fail "codesign --verify failed"
want="sys1rust $VERSION (MLX 0.32.2, $FLAVOR build)"
got=$("$BIN" --version)
[ "$got" = "$want" ] || fail "--version printed '$got', expected '$want'"

# 6. The tarball. macOS tar would also store this Mac's xattrs and AppleDouble ._ files.
tar -C "$WORK" --no-mac-metadata --no-xattrs -czf "$OUT/$NAME.tar.gz" "$NAME"
echo "$OUT/$NAME.tar.gz $(stat -f %z "$OUT/$NAME.tar.gz" | awk '{printf "%.1f MB", $1 / 1e6}')"
