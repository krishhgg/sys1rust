@AGENTS.md

## Claude Code

- `.claude/skills/` holds links to the 3 skills in `.agents/skills/`, so `/sys1rust-setup`, `/sys1rust-use` and `/sys1rust-develop` load the same files that AGENTS.md lists. Edit the files in `.agents/skills/`.
- Don't use the Bash tool's `run_in_background` for a server that the user keeps after the session. Claude Code stops background tasks, and the processes they started, when it exits. Start it with `nohup` in an ordinary Bash call, as the setup skill does, or install the service.
- A server you start only to test something can run in the background. Stop it before you finish, and check with `pgrep -fl 'sys1rust serve'` that none is left.
