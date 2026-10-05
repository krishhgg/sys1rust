//! `HF_HUB_OFFLINE` turns on `serve --offline` and stops `pull`. This is the only test in its
//! binary because it sets a process environment variable.

use clap::Parser;
use sys1rust::cli::{offline_from_env, Cli, Command};

#[test]
fn hf_hub_offline_turns_downloads_off() {
    let read = |value: Option<&str>| {
        match value {
            Some(v) => std::env::set_var("HF_HUB_OFFLINE", v),
            None => std::env::remove_var("HF_HUB_OFFLINE"),
        }
        let Command::Serve(c) = Cli::try_parse_from(["sys1rust", "serve"]).unwrap().command else {
            panic!("not serve")
        };
        (c.offline, offline_from_env())
    };
    assert_eq!(read(None), (false, false));
    // clap reads the value as it is, without trimming, so `offline_from_env` does too.
    for v in ["1", "true", "YES", "on", "2", " 0"] {
        assert_eq!(read(Some(v)), (true, true), "{v:?}");
    }
    for v in ["0", "false", "No", "off", "n", "F", ""] {
        assert_eq!(read(Some(v)), (false, false), "{v:?}");
    }
    std::env::remove_var("HF_HUB_OFFLINE");
}
