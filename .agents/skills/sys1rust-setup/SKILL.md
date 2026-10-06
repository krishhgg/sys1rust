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
- If `sysctl.proc_translated` prints 1, your shell runs under Rosetta, and the installer stops in it. Run the installer in a native shell instead. Write `arch -arm64 /bin/sh ./install.sh` wherever this skill says `./install.sh`, and pipe into `arch -arm64 /bin/sh -s -- ARGS` in place of `sh -s -- ARGS` without a clone. Variables set in front, such as `LAYA_PORT=8001 arch -arm64 /bin/sh ./install.sh`, still reach the installer.
- If macOS is older than 14.0, stop and tell the user that sys1rust needs macOS 14.0 or later.
- On macOS 26.2 or later the installer picks the `macos26` bundle. On 14.0 to 26.1 it picks the `macos14` bundle, which is about 3x slower on the M5. If the Mac runs 26.0 or 26.1, tell the user that updating macOS to 26.2 and rerunning the installer gets the faster bundle.
- The install takes about 215 MB, and each model 846 MB (multilingual 678 MB). With less than 1.1 GB free, tell the user and stop.

## 2. Install

```sh
./install.sh
```

The installer downloads the latest release for this Mac, checks it against the release's `SHA256SUMS`, unpacks it into `~/.local/share/sys1rust/<version>-<flavor>.<id>/`, points `~/.local/share/sys1rust/current` at it and links `~/.local/bin/sys1rust`. It records its paths in `~/.local/share/sys1rust/.sys1rust-install`, refuses a nonempty prefix without that registry and serializes changes with a prefix lock. It downloads no model. `--version vX.Y.Z` installs a given release instead, and `./install.sh --help` lists every option. Check the result:

```sh
~/.local/bin/sys1rust --version
```

It prints `sys1rust <version> (MLX 0.32.2, macos26 build)`, or `macos14 build` for the other bundle. From here on, run `~/.local/bin/sys1rust` by its full path, as the commands below do, so that no other `sys1rust` on `PATH` runs in its place.

## 3. Check PATH for the user

The user will type `sys1rust` in their own terminal, so check what that name finds:

```sh
command -v sys1rust
```

- If it prints the full path of `~/.local/bin/sys1rust`, such as `/Users/<name>/.local/bin/sys1rust`, go to step 4.
- If it prints another path, an older sys1rust, from another installer or a source build, comes first on `PATH`. The installer said so too, with `<path> comes before <link> on PATH`. Tell the user that `sys1rust` in a terminal runs that older binary. Ask whether to remove it, or to put `~/.local/bin` ahead of it in `~/.zshrc`. Change nothing until they answer.
- If it prints nothing, `~/.local/bin` is not on your shell's `PATH`, and the installer printed the line that adds it. Check whether `~/.zshrc` has such a line already with `grep -n '\.local/bin' ~/.zshrc`. If it doesn't, ask the user whether to add it. If they agree, append the exact line the installer printed to `~/.zshrc`. If `echo $SHELL` is not `/bin/zsh`, ask the user which file to use.

## 4. Download the model

```sh
~/.local/bin/sys1rust pull
```

This downloads typed-decisions, the default model, at the revision the release pins (846 MB) into the Hugging Face cache. The cache is `~/.cache/huggingface/hub` unless `HF_HUB_CACHE`, `HF_HOME` or `XDG_CACHE_HOME` is set. `pull` prints its progress on stderr and the snapshot directory on stdout. If it stops partway, run it again, and it resumes where it stopped. Download another model only if the user asks for it, with `~/.local/bin/sys1rust pull multilingual` (678 MB) or `~/.local/bin/sys1rust pull english` (846 MB). Then check:

```sh
~/.local/bin/sys1rust models
```

The typed-decisions row says `downloaded, 846 MB`, and the last line names the cache.

If `HF_HUB_CACHE`, `HF_HOME` or `XDG_CACHE_HOME` is set, `./install.sh --service` in step 5 writes it into the service, so the service reads the cache that `pull` filled. Run both commands in the same shell.

## 5. Start the server

The server and the service listen on port 8000 unless you pick another. Check that the port is free:

```sh
PORT=${PORT:-8000}
lsof -nP -iTCP:$PORT -sTCP:LISTEN
```

- No output means the port is free.
- A `sys1rust` row means a sys1rust server already runs. If the user wants the service, stop that server first, because the installer won't start the service while another server answers on the port. Otherwise go to step 6.
- Any other program means you must ask the user whether to stop it. If they keep it, pick another port, such as 8001, and check it the same way.

Set `PORT` to the chosen port and keep it for every later command. Each block defaults it to 8000 only when it is unset. If your tool starts a fresh shell for each command, include the chosen value in each block, such as `PORT=8001`.

Ask the user how they want to run it, unless their request already says:

- **At every login.** A LaunchAgent runs `sys1rust serve` now and at each login, and restarts it if it crashes. The model stays in memory while the user is logged in. Recommend this when apps or agents will call sys1rust often.
- **Only when they start it.** You start it now in the background, and later the user runs `sys1rust serve` when they need it.

If you can't ask, choose "only when they start it".

For the service, run:

```sh
PORT=${PORT:-8000}
LAYA_PORT=$PORT ./install.sh --service
```

The installer writes `~/Library/LaunchAgents/io.github.krishhgg.sys1rust.plist` with `LAYA_PORT` in it, starts the service and waits up to 60 s until `GET /health` answers on that port. The server logs to `~/Library/Logs/sys1rust.log`. If the installer exits with an error, read the last lines of that log.

To start it yourself, run:

```sh
PORT=${PORT:-8000}
mkdir -p ~/Library/Logs
nohup ~/.local/bin/sys1rust serve --port $PORT >> ~/Library/Logs/sys1rust-serve.log 2>&1 &
pid=$!
echo "$pid"
ready=0
for i in $(seq 60); do
  kill -0 "$pid" 2>/dev/null || break
  if curl -fs 127.0.0.1:$PORT/health; then ready=1; break; fi
  sleep 1
done
if [ "$ready" = 0 ]; then
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  tail -n 30 ~/Library/Logs/sys1rust-serve.log
  false
fi
```

It prints the process id, then the `/health` reply once the server answers. Keep the process id, because the user needs it to stop the server. A start with the model already downloaded takes a few seconds at most. The first start of a new install takes about 1.7 s longer while macOS compiles the GPU kernels. When the server is ready, its log ends with `sys1rust: listening on http://127.0.0.1:<port> (auth off)`. If no `/health` reply follows the process id, read the log with `tail -n 30 ~/Library/Logs/sys1rust-serve.log` and see "When something fails" below.

## 6. Check that it answers

```sh
PORT=${PORT:-8000}
curl -s 127.0.0.1:$PORT/health
echo
curl -s 127.0.0.1:$PORT/v1/systemone -H 'content-type: application/json' -d '{
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

The first reply has `"status":"ok"` and `"loaded":["typed-decisions"]`. The second sends the README's example request, and its `answers` holds `team`, `urgency` and `money`. With typed-decisions, `team.choice` is `billing`, `urgency.score` is about 2.5 and `money.noul` is about 0.73. A reply with `detail` instead of `answers` is an error. Read the server log.

## 7. Tell the user

Report in a few lines:

- the version and bundle that `~/.local/bin/sys1rust --version` prints;
- where it lives, `~/.local/bin/sys1rust` linked into `~/.local/share/sys1rust/current`, and the model cache that `~/.local/bin/sys1rust models` names on its last line;
- what runs, either the LaunchAgent `io.github.krishhgg.sys1rust` with its log `~/Library/Logs/sys1rust.log`, or the process id you started with its log `~/Library/Logs/sys1rust-serve.log`;
- the address `http://127.0.0.1:<port>` with the port from step 5, and the README's "Try it" section for calling it, which uses port 8000;
- how to stop and start it, from the table below;
- what step 3 found, so the user knows whether `sys1rust` in a new terminal runs this install, or still needs the `PATH` line or the older binary removed.

| it runs as | stop it with | start it again with |
| --- | --- | --- |
| the LaunchAgent | `launchctl bootout gui/$(id -u)/io.github.krishhgg.sys1rust` | `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/io.github.krishhgg.sys1rust.plist`, or the next login |
| a process you started | `kill <pid>` | `sys1rust serve --port <port>`, or `sys1rust serve` for port 8000 |
| `sys1rust serve` in a terminal | Ctrl-C | the same command |

`launchctl bootout` stops the service until the next login. `./install.sh --uninstall` removes it.

## Update

Refresh the clone first. Run the clone block in `AGENTS.md` again if you made the scratch clone with it, or `git pull --ff-only` in a clone of the user's own. Then run:

```sh
PORT=$(plutil -extract EnvironmentVariables.LAYA_PORT raw ~/Library/LaunchAgents/io.github.krishhgg.sys1rust.plist 2>/dev/null || echo 8000)
LAYA_PORT=$PORT ./install.sh
```

The first line reads the service's port from its LaunchAgent, and gives 8000 when there is no LaunchAgent or it sets no port. The installer itself also reads an existing service's port and cache from that plist before checking a restarted service. The installer installs the latest release, moves the `current` link to it, keeps the version before it as `previous` and deletes older ones. If the service runs, the installer restarts it on the new version and waits for `/health` on that port. Restart a server you started yourself with `kill <pid>` and the start block in step 5, with its port. If the new release pins a new model revision, run `~/.local/bin/sys1rust pull` to download it ahead of time, or the next `serve` downloads it. Then repeat step 6.

## Uninstall

1. Stop any server you started. `pgrep -fl 'sys1rust serve'` lists them, and `kill <pid>` stops one.
2. Run `./install.sh --uninstall`. It removes only the version directories, links, service and log that its registry owns. It refuses a prefix without its registry and leaves unrelated files, including replaced links, alone. If the LaunchAgent runs a sys1rust other than `~/.local/bin/sys1rust`, it leaves the LaunchAgent and its log and says so. It keeps the downloaded models, and prints the cache path, their size and the command that deletes them. Run that command only if the user asks for the models to go too.
3. If you added the `PATH` line, ask the user whether to remove it from `~/.zshrc`. Delete `~/Library/Logs/sys1rust-serve.log` if you created it.

## When something fails

| what you see | what to do |
| --- | --- |
| `sys1rust: error: bind 127.0.0.1:<port>: Address already in use (os error 48)`, or the installer says another server answers on 127.0.0.1:<port> | Another program holds the port. Follow the port check in step 5. |
| The installer finds no release | No release is published yet. Check `https://github.com/krishhgg/sys1rust/releases` for a prerelease. A tag alone does not prove that its release bundles are ready. If a `vX.Y.Z-rcN` tag exists and the user accepts a prerelease, run `./install.sh --version vX.Y.Z-rcN`. Otherwise build from source as the README's "Build from source" section says, or wait for a release. |
| The installer says the shell runs under Rosetta | Run it in a native shell, as step 1 says. |
| The installer reports an old macOS | sys1rust needs macOS 14.0 or later. Tell the user and stop. |
| A download stopped partway | Run the same command again. `sys1rust pull` and `sys1rust serve` resume a partial model download, and the installer is safe to rerun. |
| The installer reports another installer is running | Wait for that operation to finish, then rerun. For a stale lock, check that no installer runs and use the recovery command in the error. Never remove a live lock. |
| The installer reports no installer registry | Check that you used the intended prefix. Do not delete foreign files or make a registry by hand. Use an empty prefix for a new install. |
| The Mac is offline | `~/.local/bin/sys1rust serve --offline`, or `HF_HUB_OFFLINE=1`, loads only what the cache holds and never downloads. A missing model is then an error that names the `sys1rust pull <model>` command to run. Run it while online. The installer needs the network, unless `--from DIR` points it at release files downloaded earlier (see `./install.sh --help`). |
| `laya-mlx: nax is off for this load, MLX's gemms run instead: ...` on stderr | This is expected with the `macos14` bundle and on GPUs older than the M5's, and it isn't an error. sys1rust runs MLX's own matmuls instead and serves normally. Other `laya-mlx: ... is off for this load` lines are fallbacks of the same kind. |
| `curl` can't connect | The server isn't up, has exited or listens on another port. Read the end of its log, `~/Library/Logs/sys1rust.log` for the service or `~/Library/Logs/sys1rust-serve.log` for one you started. |
| `/health` answers 503 with `"worker":"not running"` | The inference thread stopped. Restart the server and read its log. |
| `/v1/systemone` answers 401 | The server has an API key (`--api-key` or `LAYA_API_KEY`). Send `Authorization: Bearer <key>`. |
