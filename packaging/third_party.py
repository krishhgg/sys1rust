#!/usr/bin/env python3
"""Print THIRD_PARTY.md for a release bundle: every crate compiled into the sys1rust binary
(normal dependencies on aarch64-apple-darwin) with its license expression and the license
files in its source, then mlx-c, which mlx-sys builds and links in statically.

It runs `cargo metadata` only with the Rust version in packaging/rust-toolchain-version, the
one build.sh checks before it builds.

Usage: packaging/third_party.py > THIRD_PARTY.md
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFIXES = ("LICENSE", "LICENCE", "COPYING", "NOTICE", "UNLICENSE")


def license_files(d):
    names = sorted(n for n in os.listdir(d)
                   if n.upper().startswith(PREFIXES) and os.path.isfile(os.path.join(d, n)))
    if not names:
        return ["The crate's source has no license file.", ""]
    out = []
    for n in names:
        with open(os.path.join(d, n), encoding="utf-8", errors="replace") as f:
            out += [f"### {n}", "", "```", f.read().rstrip(), "```", ""]
    return out


def main():
    with open(os.path.join(ROOT, "packaging", "rust-toolchain-version")) as f:
        rust = f.read().strip()
    got = subprocess.check_output(["cargo", "--version"], text=True).split()[1]
    if got != rust:
        sys.exit(f"third_party.py: cargo is {got}, but the release build needs {rust} "
                 f"(packaging/rust-toolchain-version). Run 'rustup toolchain install {rust}' "
                 f"and build with RUSTUP_TOOLCHAIN={rust}.")
    meta = json.loads(subprocess.check_output([
        "cargo", "metadata", "--format-version", "1", "--locked",
        "--manifest-path", os.path.join(ROOT, "runtime", "Cargo.toml"),
        "--filter-platform", "aarch64-apple-darwin"]))
    pkgs = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    members = set(meta["workspace_members"])
    root = next(i for i in members if pkgs[i]["name"] == "sys1rust")
    found, stack = set(), [root]
    while stack:
        for dep in nodes[stack.pop()]["deps"]:
            normal = any(k["kind"] is None for k in dep["dep_kinds"])
            if normal and dep["pkg"] not in found:
                found.add(dep["pkg"])
                stack.append(dep["pkg"])
    out = ["# Third-party licenses", "",
           "The sys1rust binary includes these Rust crates and mlx-c. Each section gives the "
           "license expression and the license files in the source.", ""]
    for pid in sorted(found - members, key=lambda i: (pkgs[i]["name"], pkgs[i]["version"])):
        p = pkgs[pid]
        out += [f"## {p['name']} {p['version']}", "", f"License: {p.get('license') or 'see below'}", ""]
        out += license_files(os.path.dirname(p["manifest_path"]))
    out += ["## mlx-c, built by mlx-sys and linked in", "", "License: MIT", ""]
    out += license_files(os.path.join(ROOT, "runtime", "vendor", "mlx-sys", "src", "mlx-c"))
    print("\n".join(out))


if __name__ == "__main__":
    main()
