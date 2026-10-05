#!/usr/bin/env python3
"""Print the MODELS table of runtime/crates/sys1rust/src/models.rs from a Hugging Face cache
that holds the snapshots pinned in bench/models.lock.json. The script checks every blob name
against the file's bytes. It uses sha256 for an LFS file and the git blob sha1 otherwise.

Usage: packaging/gen_manifest.py [HF_HUB_CACHE]
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = ["typed-decisions", "multilingual", "english"]
FILES = ["model.safetensors", "rl_agent_config.json", "encoder/config.json",
         "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]


def matches(path, blob):
    size = os.path.getsize(path)
    h = hashlib.sha256() if len(blob) == 64 else hashlib.sha1(b"blob %d\0" % size)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == blob


def default_cache():
    # Same order as hf_cache_dir() in laya-core's resolve.rs. Empty variables count as unset.
    env = lambda k: os.environ.get(k) or None
    if env("HF_HUB_CACHE"):
        return env("HF_HUB_CACHE")
    if env("HF_HOME"):
        return os.path.join(env("HF_HOME"), "hub")
    cache = env("XDG_CACHE_HOME") or os.path.join(env("HOME") or ".", ".cache")
    return os.path.join(cache, "huggingface", "hub")


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else default_cache()
    with open(os.path.join(ROOT, "bench", "models.lock.json")) as f:
        lock = json.load(f)
    print("pub static MODELS: [LayaModel; 3] = [")
    for name in NAMES:
        repo, rev = lock[name]["repo"], lock[name]["sha"]
        snap = os.path.join(cache, "models--" + repo.replace("/", "--"), "snapshots", rev)
        print("    LayaModel {")
        print(f'        name: "{name}",')
        print(f'        repo: "{repo}",')
        print(f'        revision: "{rev}",')
        print("        files: &[")
        for path in FILES:
            full = os.path.join(snap, path)
            blob = os.path.basename(os.readlink(full))
            if not matches(full, blob):
                sys.exit(f"{full}: its bytes do not hash to the blob name {blob}")
            print(f'            f("{path}", {os.path.getsize(full)}, "{blob}"),')
        print("        ],")
        print("    },")
    print("];")


if __name__ == "__main__":
    main()
