# Releasing sys1rust

Pushing a `v*` tag runs `.github/workflows/release.yml`. It builds both bundles on a macOS 26 runner, smoke-tests them on macOS 14, 15 and 26 runners and publishes the GitHub release with `SHA256SUMS`. A final `vX.Y.Z` tag also updates `Formula/sys1rust.rb` in krishhgg/homebrew-tap. A `vX.Y.Z-rcN` tag publishes a prerelease and leaves the tap alone. A pull request that changes `packaging/` or the workflow runs the build and smoke jobs only.

Run every command below from the repository root on an Apple silicon Mac with macOS 26.2 or later. Use 1 shell for all of them, because later steps reuse `version` and `RUSTUP_TOOLCHAIN`. The tests, the smoke runs and the Homebrew service load a model on the GPU. Run each of them alone, with no other GPU work on the Mac.

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
3. Build both bundles, run the strict suite against the macOS 26 MLX build, and smoke-test both bundles. The tests read the pinned snapshots from the Hugging Face cache, so pull all 3 models first. Every command must pass. The laya-mlx `settings` tests take about 14 minutes.
   ```sh
   packaging/build.sh macos26 dist && packaging/build.sh macos14 dist
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
4. Tag a release candidate and push the tag. The workflow builds and smoke-tests the bundles and publishes a prerelease. It leaves the tap alone. Follow the run on the repository's Actions page. If a job fails, fix the cause on main and tag the next candidate, `v$version-rc2`. Until GitHub retires the macOS 14 image on 2026-11-02, it fails `macos-14` jobs during scheduled brownouts (actions/runner-images#13518). Rerun a job that failed that way.
   ```sh
   git tag "v$version-rc1" && git push origin "v$version-rc1"
   ```
5. Install the candidate through Homebrew from a throwaway local tap. `packaging/render_formula.sh` writes the formula for the prerelease with the hashes from its `SHA256SUMS`. The first service start downloads typed-decisions (846 MB) if `~/.cache/huggingface/hub` does not hold it yet. If `curl` never answers, read `$(brew --prefix)/var/log/sys1rust.log`.
   ```sh
   sums=$(gh release download "v$version-rc1" --repo krishhgg/sys1rust --pattern SHA256SUMS --output -)
   sha() { printf '%s\n' "$sums" | awk -v f="sys1rust-$version-$1-arm64.tar.gz" '$2 == f {print $1}'; }
   brew tap-new --no-git local/sys1rust-test
   packaging/render_formula.sh "v$version-rc1" "$(sha macos26)" "$(sha macos14)" \
     > "$(brew --repository local/sys1rust-test)/Formula/sys1rust.rb"
   HOMEBREW_NO_AUTO_UPDATE=1 brew install local/sys1rust-test/sys1rust
   brew test local/sys1rust-test/sys1rust
   brew services start local/sys1rust-test/sys1rust
   until curl -fsS 127.0.0.1:8000/health; do sleep 2; done
   brew services stop local/sys1rust-test/sys1rust
   brew uninstall sys1rust && brew untap local/sys1rust-test
   ```
6. Tag the release and push the tag. The workflow publishes the release and pushes `Formula/sys1rust.rb` to krishhgg/homebrew-tap with the `TAP_TOKEN` secret, a fine-grained token with contents read and write on that repo only. The publish job clones the tap before it creates the release, so an empty or expired token stops the job before anything goes public.
   ```sh
   git tag "v$version" && git push origin "v$version"
   ```
7. On a clean account, or after `brew uninstall sys1rust`, check that `brew install krishhgg/tap/sys1rust`, `sys1rust serve` and `brew services start sys1rust` work.

## If the tap update fails

If the publish job fails at the tap because `TAP_TOKEN` expired or lost access, make a new fine-grained token with contents read and write on krishhgg/homebrew-tap only. Store it with the command below, which prompts for the value. Then open the failed run on the Actions page and choose "Re-run failed jobs". The rerun finds the release from the first attempt with the same bundles, keeps it and pushes the formula. If the formula is already up to date, the job finishes without a commit. "Re-run all jobs" builds new bundles, and the publish job then stops rather than replace the published files.

```sh
gh secret set TAP_TOKEN --repo krishhgg/sys1rust
```
