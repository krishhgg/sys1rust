//! The `sys1rust` command line. `serve` runs the HTTP server, `pull` downloads a Laya model
//! and `models` lists the Laya models and what the local Hugging Face cache holds of them.

use crate::config::Config;
use clap::{Args, Parser, Subcommand};

/// `--version` text after the name: `0.1.0 (MLX 0.32.2, macos26 build)`. build.rs reads the
/// MLX version from `MLX_VERSION` in vendor/mlx-sys/build.rs, which refuses any other MLX. The
/// build name comes from `SYS1_BUILD_FLAVOR` at compile time (packaging/build.sh sets
/// `macos14` or `macos26`) and is `source` otherwise.
pub const VERSION_LINE: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (MLX ",
    env!("SYS1RUST_MLX_VERSION"),
    ", ",
    env!("SYS1RUST_BUILD_FLAVOR"),
    " build)"
);

#[derive(Parser, Debug)]
#[command(
    name = "sys1rust",
    version = VERSION_LINE,
    about = "Laya System 1 decisions on your Mac's GPU, over Laya's /v1/systemone API",
    arg_required_else_help = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Run the HTTP server: POST /v1/systemone and GET /health.
    Serve(Config),
    /// Download a Laya model at its pinned revision into the Hugging Face cache and print
    /// the snapshot directory.
    Pull(PullArgs),
    /// List the Laya models, their pinned revisions and what the cache holds.
    Models,
}

#[derive(Args, Debug)]
pub struct PullArgs {
    /// `typed-decisions`, `multilingual`, `english`, or one of their repo ids.
    #[arg(default_value = "typed-decisions")]
    pub model: String,
}

/// Whether `HF_HUB_OFFLINE` turns downloads off for `pull` and `serve`. Unset means online.
pub fn offline_from_env() -> bool {
    std::env::var("HF_HUB_OFFLINE").is_ok_and(|v| is_offline_value(&v))
}

/// huggingface_hub's reading of `HF_HUB_OFFLINE`, `value.upper() in {"1", "ON", "YES", "TRUE"}`.
/// It ignores case and does not trim, so `2`, ` 1` and `off` all mean online.
pub fn is_offline_value(value: &str) -> bool {
    ["1", "ON", "YES", "TRUE"]
        .iter()
        .any(|t| value.eq_ignore_ascii_case(t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::error::ErrorKind;
    use clap::Parser;

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("sys1rust").chain(args.iter().copied()))
    }

    #[test]
    fn bare_command_prints_help() {
        let e = parse(&[]).unwrap_err();
        assert_eq!(
            e.kind(),
            ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
        );
        let help = e.to_string();
        for sub in ["serve", "pull", "models"] {
            assert!(help.contains(sub), "{help}");
        }
    }

    #[test]
    fn version_line_names_mlx_and_the_build() {
        let e = parse(&["--version"]).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::DisplayVersion);
        let text = e.to_string();
        let mlx = env!("SYS1RUST_MLX_VERSION");
        assert!(
            mlx.contains('.')
                && mlx
                    .split('.')
                    .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit())),
            "{mlx:?}"
        );
        let want = format!("sys1rust {} (MLX {mlx}, ", env!("CARGO_PKG_VERSION"));
        // build.rs names an unset or empty SYS1_BUILD_FLAVOR `source`.
        let build = text
            .trim_end()
            .strip_prefix(&want)
            .and_then(|rest| rest.strip_suffix(" build)"));
        assert!(
            matches!(build, Some("source" | "macos14" | "macos26")),
            "{text}"
        );
    }

    #[test]
    fn serve_takes_the_server_options() {
        let Command::Serve(c) = parse(&["serve", "--port", "0", "--model", "english"])
            .unwrap()
            .command
        else {
            panic!("not serve")
        };
        assert_eq!((c.port, c.model.as_str()), (0, "english"));
    }

    #[test]
    fn pull_defaults_to_typed_decisions() {
        let Command::Pull(p) = parse(&["pull"]).unwrap().command else {
            panic!("not pull")
        };
        assert_eq!(p.model, "typed-decisions");
        let Command::Pull(p) = parse(&["pull", "english"]).unwrap().command else {
            panic!("not pull")
        };
        assert_eq!(p.model, "english");
    }

    #[test]
    fn serve_takes_offline() {
        let Command::Serve(c) = parse(&["serve", "--offline"]).unwrap().command else {
            panic!("not serve")
        };
        assert!(c.offline);
    }

    #[test]
    fn offline_values_match_huggingface_hub() {
        for v in ["1", "ON", "on", "Yes", "TRUE", "true"] {
            assert!(is_offline_value(v), "{v:?}");
        }
        for v in ["", "0", "2", " 1", "1 ", "off", "no", "false", "abc"] {
            assert!(!is_offline_value(v), "{v:?}");
        }
    }

    #[test]
    fn models_takes_no_arguments() {
        assert!(matches!(
            parse(&["models"]).unwrap().command,
            Command::Models
        ));
        assert!(parse(&["models", "extra"]).is_err());
    }

    #[test]
    fn the_old_server_flags_need_serve() {
        assert!(parse(&["--port", "8000"]).is_err());
        assert!(parse(&["--model", "english"]).is_err());
    }
}
