# Releasing sys1rust

Pushing a `v*` tag runs `.github/workflows/release.yml`. It builds both bundles on a macOS 26 runner, installs the macOS 14 bundle with `install.sh` on a macOS 15 runner and the macOS 26 bundle on a macOS 26 runner, smoke-tests them there, and publishes the GitHub release with both bundles and `SHA256SUMS`. A `vX.Y.Z-rcN` tag publishes a prerelease. `install.sh` installs the latest final release, and a prerelease only through `--version`. A pull request that changes `packaging/`, `install.sh` or the workflow runs the build and smoke jobs only.

The build job also runs `packaging/test_install.sh /bin/sh` and `packaging/test_install.sh /bin/dash`. They test `install.sh` against fake bundles, a local release server and a fake `launchctl`, and need no build, network or GPU. Run them after any change to `install.sh`.

Run every command below from the repository root on a Mac with an M5-class GPU (GPU generation 17 or later) and macOS 26.2 or later. The strict suite's laya-mlx `settings` tests need MLX's NAX gemms and fail on the fallback that older GPUs get, while the smoke tests run on any Apple silicon Mac. Use 1 shell for all the commands, because later steps reuse `version`, `RUSTUP_TOOLCHAIN`, `HF_HUB_CACHE` and `rc`. The tests, the smoke runs and the server in step 5 load a model on the GPU. Run each of them alone, with no other GPU work on the Mac.

1. Install the pinned Rust and select it for this shell. `packaging/build.sh` and `packaging/third_party.py` stop when `rustc` or `cargo` reports another version.
   ```sh
   rust=$(cat packaging/rust-toolchain-version)
   rustup toolchain install "$rust"
   export RUSTUP_TOOLCHAIN=$rust
   ```
2. Set the version in the `[workspace.package]` table of `runtime/Cargo.toml` and update `runtime/Cargo.lock` to match. build.sh builds with `--locked`, so a stale lock file fails the release build.
   ```sh
   cargo update --workspace --manifest-path runtime/Cargo.toml
   ```
   Merge both files to main. Then switch to main and read the version back.
   ```sh
   git switch main && git pull --ff-only
   version=$(sed -n 's/^version = "\(.*\)"$/\1/p' runtime/Cargo.toml)
   echo "$version"
   ```
3. Build both bundles, run the strict suite against the macOS 26 MLX build, and smoke-test both bundles. The laya-mlx tests do not ask for the pinned revisions. They load the snapshot that a model's `refs/main` names, or else the model's only snapshot, and `sys1rust pull` writes no `refs/main`. So this step pulls the 3 pinned models (846 MB, 678 MB and 846 MB) into a fresh cache under `packaging/.work`, which then holds only the pinned snapshots. `HF_HUB_CACHE` points every later command in this shell at that cache. Every command must pass. The laya-mlx `settings` tests take about 14 minutes.
   ```sh
   packaging/build.sh macos26 dist && packaging/build.sh macos14 dist
   export HF_HUB_CACHE=$PWD/packaging/.work/hf-cache
   rm -rf "$HF_HUB_CACHE"
   for m in typed-decisions multilingual english; do
     packaging/.work/macos26/sys1rust-$version-macos26-arm64/bin/sys1rust pull $m
   done
   export MLX_SYS_PREBUILT_DIR=$PWD/packaging/.work/macos26/wheel/mlx
   cargo test --release --manifest-path runtime/Cargo.toml --workspace
   for t in settings loading reference; do
     SYS1_TEST_ALL_CHECKPOINTS=1 cargo test --release --manifest-path runtime/Cargo.toml \
       -p laya-mlx --test $t -- --ignored --test-threads 1
   done
   cargo test --release --manifest-path runtime/Cargo.toml -p sys1rust --test live -- --ignored
   cargo test --release --manifest-path runtime/Cargo.toml -p sys1rust --test live_pull -- --ignored
   for fl in macos26 macos14; do
     python3 packaging/smoke.py packaging/.work/$fl/sys1rust-$version-$fl-arm64/bin/sys1rust
   done
   ```
4. Tag a release candidate and push the tag. `rc` holds the candidate's tag for this step and the next. The workflow builds and smoke-tests the bundles and publishes a prerelease. Follow the run on the repository's Actions page. If a job fails, merge the fix to main, run `git pull --ff-only`, set `rc` to the next candidate, such as `v$version-rc2`, and run the tag line again.
   ```sh
   rc=v$version-rc1
   git tag "$rc" && git push origin "$rc"
   ```
5. Install the candidate with `install.sh` into a scratch prefix, as a user would. `install.sh` downloads the bundle for this Mac and checks it against the prerelease's `SHA256SUMS`. Then check that it serves in this shell and as a LaunchAgent. `--service` passes this shell's `HF_HUB_CACHE` into the LaunchAgent, so the service loads typed-decisions from the cache that step 3 filled. `install.sh --service` waits up to 60 s for `/health`, and if the server fails it prints the end of `~/Library/Logs/sys1rust.log`. Every install uses the same LaunchAgent label, so `--service` here replaces a sys1rust service that this account already runs. Run `install.sh --service` again afterward to bring that one back. `--uninstall` removes the scratch install, the LaunchAgent and its log, and keeps the models.
   ```sh
   tmp=$(mktemp -d)
   ./install.sh --version "$rc" --prefix "$tmp" --bin-dir "$tmp/bin"
   "$tmp/bin/sys1rust" serve & pid=$!
   until curl -fs 127.0.0.1:8000/health; do sleep 2; done
   kill "$pid" && wait "$pid"
   ./install.sh --version "$rc" --prefix "$tmp" --bin-dir "$tmp/bin" --service
   ./install.sh --uninstall --prefix "$tmp" --bin-dir "$tmp/bin"
   ```
6. Tag the release and push the tag. The workflow publishes the release, and `releases/latest` then points to it, so `install.sh` without `--version` installs it.
   ```sh
   git tag "v$version" && git push origin "v$version"
   ```
7. On a clean macOS account, check the one-line install from main with the first line below. Add `~/.local/bin` to `PATH` if `install.sh` prints the line for it. Then check that `sys1rust serve` answers `curl 127.0.0.1:8000/health` from another shell. Its first start downloads typed-decisions (846 MB). Stop the server, then run the other 2 lines to check the service and the uninstall.
   ```sh
   curl -fsSL https://raw.githubusercontent.com/krishhgg/sys1rust/main/install.sh | sh
   curl -fsSL https://raw.githubusercontent.com/krishhgg/sys1rust/main/install.sh | sh -s -- --service
   curl -fsSL https://raw.githubusercontent.com/krishhgg/sys1rust/main/install.sh | sh -s -- --uninstall
   ```

## If the publish job fails

Open the failed run on the Actions page and choose "Re-run failed jobs". If a cancelled attempt left a draft release of the tag, the rerun stops and names it. Delete it on the Releases page, or with `gh release delete <tag>` after checking that it is still a draft, then rerun the publish job. The rerun keeps a published release only if that release's 2 bundles and `SHA256SUMS` match the run's files, and otherwise stops. "Re-run all jobs" builds new bundles, and the publish job then stops rather than replace the published files.
