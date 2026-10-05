//! The `sys1rust` command line. `serve` runs the HTTP server and `models` lists the Laya
//! models and what the local Hugging Face cache holds of them.

use crate::config::Config;
use clap::{Parser, Subcommand};

/// `--version` text after the name: `0.1.0 (MLX 0.32.2, macos26 build)`. The MLX version is
/// fixed because vendor/mlx-sys/build.rs refuses a prebuilt MLX of any other version. The
/// build name comes from `SYS1_BUILD_FLAVOR` at compile time (packaging/build.sh sets
/// `macos14` or `macos26`) and is `source` otherwise.
pub const VERSION_LINE: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (MLX 0.32.2, ",
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
    /// List the Laya models, their pinned revisions and what the cache holds.
    Models,
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
        for sub in ["serve", "models"] {
            assert!(help.contains(sub), "{help}");
        }
    }

    #[test]
    fn version_line_names_mlx_and_the_build() {
        let e = parse(&["--version"]).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::DisplayVersion);
        let text = e.to_string();
        let want = format!("sys1rust {} (MLX 0.32.2, ", env!("CARGO_PKG_VERSION"));
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
