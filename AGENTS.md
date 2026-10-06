# AGENTS.md

sys1rust runs Laya's System 1 decision models on an Apple silicon Mac's GPU, in Rust on MLX. Its server, `sys1rust serve`, speaks the `/v1/systemone` API of `laya serve` on `http://127.0.0.1:8000`. This file tells coding agents which instructions to follow. Read the skill for your task before you run anything.

## Pick the skill for the task

| the user wants to | follow |
| --- | --- |
| install, update, start, stop or uninstall sys1rust on their Mac | [`.agents/skills/sys1rust-setup/SKILL.md`](.agents/skills/sys1rust-setup/SKILL.md) |
| call a running sys1rust from a program, a script or an agent | [`.agents/skills/sys1rust-use/SKILL.md`](.agents/skills/sys1rust-use/SKILL.md) |
| change the code, run the tests, measure speed or make a release | [`.agents/skills/sys1rust-develop/SKILL.md`](.agents/skills/sys1rust-develop/SKILL.md) |

A request like "set up sys1rust on my Mac" goes to the setup skill. If you are reading this on GitHub and have no clone, make a scratch one first, outside any project of the user's:

```sh
dir=$(mktemp -d "${TMPDIR:-/tmp}/sys1rust-agent.XXXXXX")
if xcode-select -p >/dev/null 2>&1 && git --version >/dev/null 2>&1; then
  git clone --depth 1 https://github.com/krishhgg/sys1rust "$dir"
else
  curl -fsSL -o "$dir/source.tar.gz" https://codeload.github.com/krishhgg/sys1rust/tar.gz/refs/heads/main &&
    tar -xzf "$dir/source.tar.gz" --strip-components 1 -C "$dir" && rm "$dir/source.tar.gz"
fi && cd "$dir"
```

On a new Mac, `git` can be a stub that asks the user to install Apple's command line tools. The block runs `git` only after `xcode-select -p` finds those tools, and otherwise downloads the same files as a tarball with `curl`. Don't install the command line tools for this. The block enters the folder only when the download worked. If it fails, show the user its error and stop. The installed sys1rust doesn't need the folder afterwards.

Agents that support skills also find these 3 files as the skills `sys1rust-setup`, `sys1rust-use` and `sys1rust-develop`. Codex, [Cursor](https://cursor.com/docs/skills) and [GitHub Copilot](https://docs.github.com/en/copilot/concepts/agents/about-agent-skills) read `.agents/skills/`, and Claude Code reads the links to them in `.claude/skills/`. Without skill support, open the file and follow it. Edit the skills only in `.agents/skills/`.

## Rules for every agent

- Never use `sudo`. Nothing here needs it.
- When you set sys1rust up for a user, write only to the clone, `~/.local/share/sys1rust`, `~/.local/bin`, `~/Library/LaunchAgents`, `~/Library/Logs` and the Hugging Face cache. Ask the user before you write anywhere else or edit a shell startup file such as `~/.zshrc`.
- Don't delete downloaded models without asking. The Hugging Face cache can hold other tools' models.
- Run one GPU job at a time on a Mac. A running `sys1rust serve`, a test suite that loads a model and a benchmark all use the GPU, and 2 at once slow each other.
- Keep the server on `127.0.0.1`. Bind a wider address only when the user asks, and then follow "Serving other machines" in the [README](README.md).
- Take commands, flags and numbers from `sys1rust --help`, `./install.sh --help`, the README and the code. Don't guess an option name.
- Don't commit, push or tag unless the user asks.

## Facts to get right

- sys1rust needs Apple silicon, a native shell rather than Rosetta, and macOS 14.0 or later. On macOS 26.2 or later the installer picks the `macos26` bundle, which is about 3x faster on the M5 than the `macos14` bundle.
- `./install.sh` installs the latest release into `~/.local/share/sys1rust` and links `~/.local/bin/sys1rust`. `--service` adds the LaunchAgent `io.github.krishhgg.sys1rust`, which logs to `~/Library/Logs/sys1rust.log`. `--uninstall` removes both and keeps the models.
- `sys1rust serve` downloads its model on the first start, typed-decisions (846 MB) by default, into the Hugging Face cache. `sys1rust pull [model]` downloads one ahead of time, and `sys1rust models` lists the 3 models and what the cache holds.
- `GET /health` says whether the server is up. `POST /v1/systemone` takes a state and its questions and answers each one with a choice, a score or a yes probability.

## Repository map

- `install.sh` installs a prebuilt release from GitHub Releases.
- `runtime/` is the Rust workspace, `runtime/Cargo.toml`.
  - `crates/laya-core` parses requests, tokenizes, lays out sequences and decodes answers. It has no GPU code.
  - `crates/laya-mlx` runs the forward pass on MLX through mlx-rs, with sys1rust's own Metal kernels.
  - `crates/sys1rust` is the `sys1rust` command (`serve`, `pull`, `models`) and its HTTP server.
  - `crates/sys1-bench` is the benchmark adapter and `sys1-probe`, which compares engine settings.
  - `vendor/mlx-sys` is mlx-sys 0.6.0 with a build that links a prebuilt MLX.
- `bench/` holds the benchmark harness, the workloads, the upstream reference answers and the pinned model revisions in `bench/models.lock.json`. [`bench/PLAN.md`](bench/PLAN.md) has its ground rules.
- `packaging/` builds the release bundles, smoke-tests them and holds the release checklist, [`packaging/RELEASE.md`](packaging/RELEASE.md).
- `results/` holds the measured write-ups, from the bake-off ([`results/REPORT.md`](results/REPORT.md)) to the speed rounds ([`results/SPEED.md`](results/SPEED.md)).
- `research/` holds sourced reports on the models, runtimes and hardware.
- `docs/assets/` holds the README's diagrams.
