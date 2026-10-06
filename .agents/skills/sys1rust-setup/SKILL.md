---
name: sys1rust-setup
description: Install, update, start, check, stop or uninstall sys1rust, the local server for Laya's System 1 models, on the user's Apple silicon Mac from its prebuilt GitHub release. Use when the user asks to set up, install, run, update, repair or remove sys1rust, or gives you https://github.com/krishhgg/sys1rust and asks you to set it up.
---

# Set up sys1rust

This skill installs the prebuilt `sys1rust` with `install.sh`, downloads a model, starts the server and checks that it answers. Follow the steps in order, and run each command from the root of the sys1rust clone. Without a clone, run each `./install.sh ARGS` below as `curl -fsSL https://raw.githubusercontent.com/krishhgg/sys1rust/main/install.sh | sh -s -- ARGS`.

Write only to the clone, `~/.local/share/sys1rust`, `~/.local/bin`, `~/Library/LaunchAgents`, `~/Library/Logs` and the Hugging Face cache. Ask the user before you write anywhere else, edit a shell startup file or delete a model. Never use `sudo`. If your sandbox blocks the network or the GPU, ask the user to let these commands run outside it.

## 1. Check the Mac

```sh
sysctl -n hw.optional.arm64           # 1 on Apple silicon, an error on an Intel Mac
sysctl -n sysctl.proc_translated      # 0 in a native shell, 1 under Rosetta
sw_vers -productVersion               # the macOS version
df -H ~ | awk 'NR==2 {print $4}'      # free space on the home volume
```

- If `hw.optional.arm64` prints 0 or an error, the Mac has an Intel chip. Stop and tell the user that sys1rust runs only on Apple silicon.
- If `sysctl.proc_translated` prints 1, your shell runs under Rosetta, and the installer stops in it. Write `arch -arm64 /bin/sh ./install.sh` wherever this skill says `./install.sh`, which runs it natively.
- If macOS is older than 14.0, stop and tell the user that sys1rust needs macOS 14.0 or later.
- On macOS 26.2 or later the installer picks the `macos26` bundle. On 14.0 to 26.1 it picks the `macos14` bundle, which is about 3x slower on the M5. If the Mac runs 26.0 or 26.1, tell the user that updating macOS to 26.2 and rerunning the installer gets the faster bundle.
- The install takes about 215 MB, and each model 846 MB (multilingual 678 MB). With less than 1.1 GB free, tell the user and stop.

## 2. Install

```sh
./install.sh
```

The installer downloads the latest release for this Mac, checks it against the release's `SHA256SUMS`, unpacks it into `~/.local/share/sys1rust/<version>-<flavor>/`, points `~/.local/share/sys1rust/current` at it and links `~/.local/bin/sys1rust`. It downloads no model. `--version vX.Y.Z` installs a given release instead, and `./install.sh --help` lists every option. Check the result:

```sh
~/.local/bin/sys1rust --version
```

It prints `sys1rust <version> (MLX 0.32.2, macos26 build)`, or `macos14 build` for the other bundle.

## 3. Put sys1rust on PATH

```sh
command -v sys1rust
```

If this prints nothing, `~/.local/bin` is not on your shell's `PATH`, and the installer printed the line that adds it.

1. Check whether `~/.zshrc` has such a line already with `grep -n '\.local/bin' ~/.zshrc`.
2. If it doesn't, ask the user whether to add it. If they agree, append the exact line the installer printed to `~/.zshrc`. If `echo $SHELL` is not `/bin/zsh`, ask the user which file to use.
3. Your shell doesn't reread `~/.zshrc`. While `command -v sys1rust` prints nothing, write `~/.local/bin/sys1rust` wherever this skill says `sys1rust`.

## 4. Download the model

```sh
sys1rust pull
```

This downloads typed-decisions, the default model, at the revision the release pins (846 MB) into the Hugging Face cache. The cache is `~/.cache/huggingface/hub` unless `HF_HUB_CACHE`, `HF_HOME` or `XDG_CACHE_HOME` is set. `pull` prints its progress on stderr and the snapshot directory on stdout. If it stops partway, run it again, and it resumes where it stopped. Download another model only if the user asks for it, with `sys1rust pull multilingual` (678 MB) or `sys1rust pull english` (846 MB). Then check:

```sh
sys1rust models
```

The typed-decisions row says `downloaded, 846 MB`, and the last line names the cache.

If `HF_HUB_CACHE`, `HF_HOME` or `XDG_CACHE_HOME` is set, `./install.sh --service` in step 5 writes it into the service, so the service reads the cache that `pull` filled. Run both commands in the same shell.

## 5. Start the server

Check that port 8000 is free:

```sh
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

- No output means the port is free.
- A `sys1rust` row means a sys1rust server already runs. If the user wants the service, stop that server first, because the installer won't start the service while another server answers on the port. Otherwise go to step 6.
- Any other program means you must ask the user whether to stop it. If they keep it, pick a free port such as 8001. Run `LAYA_PORT=8001 ./install.sh --service` for the service, or `sys1rust serve --port 8001` for a server you start yourself, and use that port in step 6.

Ask the user how they want to run it, unless their request already says:

- **At every login.** A LaunchAgent runs `sys1rust serve` now and at each login, and restarts it if it crashes. The model stays in memory while the user is logged in. Recommend this when apps or agents will call sys1rust often.
- **Only when they start it.** You start it now in the background, and later the user runs `sys1rust serve` when they need it.

If you can't ask, choose "only when they start it".

For the service, run:

```sh
./install.sh --service
```

The installer writes `~/Library/LaunchAgents/io.github.krishhgg.sys1rust.plist`, starts the service and waits up to 60 s until `GET /health` answers on 127.0.0.1:8000. The server logs to `~/Library/Logs/sys1rust.log`. If the installer exits with an error, read the last lines of that log.

To start it yourself, run:

```sh
nohup sys1rust serve >> ~/Library/Logs/sys1rust-serve.log 2>&1 &
echo $!
```

Keep the process id it prints, because the user needs it to stop the server. Then wait until the server answers:

```sh
for i in $(seq 60); do curl -fs 127.0.0.1:8000/health && break; sleep 1; done
```

A start with the model already downloaded takes a few seconds at most. The first start of a new install takes about 1.7 s longer while macOS compiles the GPU kernels. When the server is ready, its log ends with `sys1rust: listening on http://127.0.0.1:8000 (auth off)`. If the loop ends without a reply, read the log with `tail -n 30 ~/Library/Logs/sys1rust-serve.log` and see "When something fails" below.

## 6. Check that it answers

```sh
curl -s 127.0.0.1:8000/health
```

The reply has `"status":"ok"` and `"loaded":["typed-decisions"]`. Then send the README's example request:

```sh
curl -s 127.0.0.1:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Customer: I was charged twice for my subscription this month and I need it fixed today.",
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "payments, charges and refunds", "tech": "bugs and outages", "sales": "new purchases"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["none", "low", "high", "now"]},
    "money": {"type": "noul", "instructions": "Is money involved?"}
  }
}'
```

The reply is JSON whose `answers` holds `team`, `urgency` and `money`. With typed-decisions, `team.choice` is `billing`, `urgency.score` is about 2.5 and `money.noul` is about 0.73. A reply with `detail` instead of `answers` is an error. Read the server log.

## 7. Tell the user

Report in a few lines:

- the version and bundle that `sys1rust --version` prints;
- where it lives, `~/.local/bin/sys1rust` linked into `~/.local/share/sys1rust/current`, and the model cache that `sys1rust models` names on its last line;
- what runs, either the LaunchAgent `io.github.krishhgg.sys1rust` with its log `~/Library/Logs/sys1rust.log`, or the process id you started with its log `~/Library/Logs/sys1rust-serve.log`;
- the address `http://127.0.0.1:8000`, and the README's "Try it" section for calling it;
- how to stop and start it, from the table below;
- whether they still need to add the `PATH` line.

| it runs as | stop it with | start it again with |
| --- | --- | --- |
| the LaunchAgent | `launchctl bootout gui/$(id -u)/io.github.krishhgg.sys1rust` | `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/io.github.krishhgg.sys1rust.plist`, or the next login |
| a process you started | `kill <pid>` | `sys1rust serve` |
| `sys1rust serve` in a terminal | Ctrl-C | `sys1rust serve` |

`launchctl bootout` stops the service until the next login. `./install.sh --uninstall` removes it.

## Update

```sh
git pull --ff-only
./install.sh
```

The installer installs the latest release, moves the `current` link to it, keeps the version before it as `previous` and deletes older ones. If the service runs, the installer restarts it on the new version and waits for `/health`. Restart a server you started yourself with `kill <pid>` and step 5. If the new release pins a new model revision, run `sys1rust pull` to download it ahead of time, or the next `serve` downloads it. Then repeat step 6.

## Uninstall

1. Stop any server you started. `pgrep -fl 'sys1rust serve'` lists them, and `kill <pid>` stops one.
2. Run `./install.sh --uninstall`. It removes the service, the install, the link and `~/Library/Logs/sys1rust.log`. It keeps the downloaded models, and prints the cache path, their size and the command that deletes them. Run that command only if the user asks for the models to go too.
3. If you added the `PATH` line, ask the user whether to remove it from `~/.zshrc`. Delete `~/Library/Logs/sys1rust-serve.log` if you created it.

## When something fails

| what you see | what to do |
| --- | --- |
| `sys1rust: error: bind 127.0.0.1:8000: Address already in use (os error 48)`, or the installer says another server answers on 127.0.0.1:8000 | Another program holds port 8000. Find it with `lsof -nP -iTCP:8000 -sTCP:LISTEN` and follow step 5. |
| The installer finds no release | No release is published yet. `git ls-remote --tags https://github.com/krishhgg/sys1rust 'v*'` lists the tags. If a `vX.Y.Z-rcN` tag exists and the user accepts a prerelease, run `./install.sh --version vX.Y.Z-rcN`. Otherwise build from source as the README's "Build from source" section says, or wait for a release. |
| The installer says the shell runs under Rosetta | Run it as `arch -arm64 /bin/sh ./install.sh`. |
| The installer reports an old macOS | sys1rust needs macOS 14.0 or later. Tell the user and stop. |
| A download stopped partway | Run the same command again. `sys1rust pull` and `sys1rust serve` resume a partial model download, and the installer is safe to rerun. |
| The Mac is offline | `sys1rust serve --offline`, or `HF_HUB_OFFLINE=1`, loads only what the cache holds and never downloads. A missing model is then an error that names the `sys1rust pull <model>` command to run. Run it while online. The installer needs the network, unless `--from DIR` points it at release files downloaded earlier (see `./install.sh --help`). |
| `laya-mlx: nax is off for this load, MLX's gemms run instead: ...` on stderr | This is expected with the `macos14` bundle and on GPUs older than the M5's, and it isn't an error. sys1rust runs MLX's own matmuls instead and serves normally. Other `laya-mlx: ... is off for this load` lines are fallbacks of the same kind. |
| `curl` can't connect | The server isn't up or has exited. Read the end of its log, `~/Library/Logs/sys1rust.log` for the service or `~/Library/Logs/sys1rust-serve.log` for one you started. |
| `/health` answers 503 with `"worker":"not running"` | The inference thread stopped. Restart the server and read its log. |
| `/v1/systemone` answers 401 | The server has an API key (`--api-key` or `LAYA_API_KEY`). Send `Authorization: Bearer <key>`. |
