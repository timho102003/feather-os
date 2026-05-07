---
name: submit-github-report
description: Use when the user asks to file a bug, open an issue, report a problem upstream, or share a fix back to the Feather project. Always call after a self-repair cycle (request_restart) so the patch can be preserved across releases. Required reading before invoking the submit_github_report tool.
---

# Submitting a GitHub report (issue or PR)

Use this skill when the user wants to share a bug or a fix back to the
Feather project, or when you have just patched feather/* code via
`request_restart` and the user is on a wheel install (your patch is
ephemeral and will be overwritten on the next `pip install --upgrade`).

The corresponding tool is `submit_github_report`. This skill exists to
make sure the report is high-quality and the user is in the loop.

## Hard rules (do not violate)

* **Never auto-submit.** Always confirm with the user first. Show them
  the title and body you plan to send, and only call the tool after
  explicit "yes".
* **One bug per report.** Do not bundle unrelated issues into one
  submission — they are harder to triage and reviewers can only merge
  parts of one.
* **Tests must pass first.** Before reporting a bug as fixed (or
  proposing a PR), run `bash uv run pytest` and confirm green. If any
  test fails, do not call the tool — report the test failure to the
  user instead.
* **Truthful reproduction.** Only describe steps and outputs you have
  actually observed in this session. Do not infer behaviour you have
  not verified.

## When to use

* The user explicitly asks to file an issue or open a PR.
* You hit a bug that looks like it could affect other users (a library
  / framework / SDK pitfall, not a project-specific configuration
  issue) and the user wants to share the fix back.
* You completed a self-repair cycle in wheel install mode — your
  patched files in `site-packages/feather/` will be overwritten on the
  next package upgrade. Filing an issue or PR is the only way the fix
  persists.

## Issue template (always use)

```
### Summary
<one-line problem statement>

### Expected behaviour
<what the user / docs say should happen>

### Actual behaviour
<what actually happened — include the exact error message or wrong output>

### Reproduction
1. <step>
2. <step>
3. <step>

### Environment
- Feather version: (run `feather --version`)
- Python version: (run `python3 --version`)
- Install mode: editable | wheel | unknown
- OS: linux / macos / windows

### Diagnostic context
<paste the relevant log lines from .feather/logs/feather.log,
 or the relevant code excerpt with file:line citations>
```

Do NOT include personal data, API keys, or full session transcripts in
the body. Strip anything sensitive before submission.

## Workflow

1. Read the in-conversation context: what is the bug, what reproduces
   it, what (if anything) did you patch?
2. Draft the title (under 70 characters) and body using the template
   above.
3. Show the draft to the user for confirmation. Make it easy to edit
   in-place — paste it back so they can copy-paste corrections.
4. After "yes", call `submit_github_report` with the confirmed title
   and body.
5. The tool returns the issue URL on success — share it with the user.

## What goes in `kind`

* `kind: "issue"` — the v1 supported path. Works on any repo the user
  has read access to (`gh` CLI handles cross-repo issue creation).
* `kind: "pr"` — not supported in this release. Surface this limit to
  the user when relevant: "PRs aren't supported by the tool yet; I can
  file a detailed issue with the patch attached, and you can open the
  PR yourself."

## Failure modes

The tool fails (returns an error string, does not raise) when:

* `gh` CLI is not installed or not on PATH → tell the user to install
  it (`brew install gh` or distro package).
* `gh auth status` reports not logged in → tell the user to run
  `gh auth login` before retrying.
* The repo doesn't exist or the user lacks issue-create permission →
  surface the gh stderr verbatim so the user can fix it.

In all failure cases, surface the tool's response to the user verbatim
— do not try to retry silently or paper over the failure.
