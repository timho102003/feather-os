"""Tests for the first-run onboarding wizard."""

from __future__ import annotations

import json
from pathlib import Path

from subprocess import CompletedProcess

from feather.onboarding import (
    DockerNotAvailable,
    OnboardingAnswers,
    OnboardingWizard,
    QdrantStartFailed,
    apply_app_yaml_toggles,
    docker_available,
    ensure_local_qdrant_container,
    is_onboarded,
    mark_onboarded,
    maybe_run_onboarding,
    qdrant_container_state,
    remove_local_qdrant_container,
    stop_local_qdrant_container,
    write_env_file,
)


def _ok(stdout: str = "", stderr: str = "") -> CompletedProcess:
    return CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=stderr)


def _fail(stderr: str) -> CompletedProcess:
    return CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)


def test_write_env_creates_file_and_appends_keys(tmp_path: Path) -> None:
    """First write creates the .env, restricts permissions, and lists keys deterministically."""

    env_path = tmp_path / ".env"
    answers = {
        "OPENAI_API_KEY": "sk-test",
        "GEMINI_API_KEY": "gem-test",
    }

    written = write_env_file(env_path, answers)

    assert written == ["OPENAI_API_KEY", "GEMINI_API_KEY"]
    text = env_path.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=sk-test" in text
    assert "GEMINI_API_KEY=gem-test" in text


def test_write_env_overwrites_existing_keys_on_rerun(tmp_path: Path) -> None:
    """Wizard re-runs are authoritative — existing values get overwritten.

    Previously the wizard's new value was silently dropped if the key
    already had a non-empty value, so re-onboarding to fix a typo
    (e.g. ``QDRANT_URL=http://localhost:6333`` → ``http://qdrant:6333``)
    appeared to succeed but left the bad URL in ``.env``.
    """

    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENAI_API_KEY=existing\n"
        "QDRANT_URL=http://localhost:6333\n",
        encoding="utf-8",
    )
    written = write_env_file(
        env_path,
        {
            "OPENAI_API_KEY": "new",
            "QDRANT_URL": "http://qdrant:6333",
            "PARALLEL_API_KEY": "p-test",
        },
    )

    assert set(written) == {"OPENAI_API_KEY", "QDRANT_URL", "PARALLEL_API_KEY"}
    text = env_path.read_text(encoding="utf-8")
    # Each key appears exactly once; new values won.
    assert text.count("OPENAI_API_KEY=") == 1
    assert "OPENAI_API_KEY=new" in text
    assert "OPENAI_API_KEY=existing" not in text
    assert text.count("QDRANT_URL=") == 1
    assert "QDRANT_URL=http://qdrant:6333" in text
    assert "QDRANT_URL=http://localhost:6333" not in text
    assert "PARALLEL_API_KEY=p-test" in text


def test_write_env_leaves_unrelated_keys_untouched(tmp_path: Path) -> None:
    """Keys not in the wizard's secrets dict survive a re-run."""

    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENAI_API_KEY=existing\nUNRELATED_KEY=keep-me\n",
        encoding="utf-8",
    )
    write_env_file(env_path, {"OPENAI_API_KEY": "new"})

    text = env_path.read_text(encoding="utf-8")
    assert "UNRELATED_KEY=keep-me" in text
    assert "OPENAI_API_KEY=new" in text


def test_write_env_skips_keys_with_empty_values(tmp_path: Path) -> None:
    """Empty values are dropped so we never write KEY=."""

    env_path = tmp_path / ".env"
    written = write_env_file(env_path, {"OPENAI_API_KEY": "sk-test", "QDRANT_API_KEY": ""})

    assert written == ["OPENAI_API_KEY"]
    text = env_path.read_text(encoding="utf-8")
    assert "QDRANT_API_KEY" not in text


def test_write_env_overwrites_existing_empty_value(tmp_path: Path) -> None:
    """A pre-existing ``KEY=`` placeholder must not block the wizard's real key."""

    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=\n", encoding="utf-8")
    written = write_env_file(env_path, {"OPENAI_API_KEY": "sk-real"})

    assert written == ["OPENAI_API_KEY"]
    text = env_path.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=sk-real" in text
    # The placeholder line is removed, not duplicated.
    assert text.count("OPENAI_API_KEY=") == 1


def test_onboarding_answers_round_trip() -> None:
    """``OnboardingAnswers.collect_secrets`` returns only non-empty key/value pairs."""

    answers = OnboardingAnswers(
        name="Tim",
        openai_api_key="sk-1",
        provider="openai",
        memory_enabled=False,
        web_search_enabled=False,
    )
    assert answers.collect_secrets() == {"OPENAI_API_KEY": "sk-1"}


def test_app_yaml_toggle_rewrites_active_provider(tmp_path: Path) -> None:
    """``apply_app_yaml_toggles`` rewrites the active_provider scalar in place."""

    yaml_path = tmp_path / "app.yaml"
    yaml_path.write_text(
        "compaction:\n  enabled: true\n\nactive_provider: openai\n\nmemory:\n  enabled: false\n",
        encoding="utf-8",
    )
    changed = apply_app_yaml_toggles(yaml_path, active_provider="openrouter", memory_enabled=True)
    assert changed == {"active_provider": True, "memory_enabled": True}
    text = yaml_path.read_text(encoding="utf-8")
    assert "active_provider: openrouter" in text
    assert "enabled: true" in text
    assert "compaction:" in text


def test_app_yaml_toggle_preserves_when_lines_missing(tmp_path: Path) -> None:
    """Missing scalars are reported as not-changed and the file is left alone."""

    yaml_path = tmp_path / "app.yaml"
    yaml_path.write_text("scheduler:\n  enabled: true\n", encoding="utf-8")
    before = yaml_path.read_text(encoding="utf-8")
    changed = apply_app_yaml_toggles(yaml_path, active_provider="openai", memory_enabled=False)
    assert changed == {"active_provider": False, "memory_enabled": False}
    assert yaml_path.read_text(encoding="utf-8") == before


def test_app_yaml_toggle_preserves_inline_comments(tmp_path: Path) -> None:
    """Inline trailing comments on the toggled lines must survive the rewrite."""

    yaml_path = tmp_path / "app.yaml"
    yaml_path.write_text(
        "active_provider: openai          # default routing\n"
        "memory:\n"
        "  enabled: false  # global gate, lead opts in\n",
        encoding="utf-8",
    )
    apply_app_yaml_toggles(yaml_path, active_provider="openrouter", memory_enabled=True)
    text = yaml_path.read_text(encoding="utf-8")
    assert "active_provider: openrouter" in text
    assert "# default routing" in text
    assert "# global gate, lead opts in" in text


def test_app_yaml_toggle_skips_when_target_has_block_scalar(tmp_path: Path) -> None:
    """A block-scalar string above the real key must not be mistaken for it."""

    yaml_path = tmp_path / "app.yaml"
    yaml_path.write_text(
        "memory:\n"
        "  description: |\n"
        "    when memory enabled: true is set...\n"
        "  enabled: false\n",
        encoding="utf-8",
    )
    changed = apply_app_yaml_toggles(yaml_path, active_provider="openai", memory_enabled=True)
    text = yaml_path.read_text(encoding="utf-8")
    # The real `enabled: false` key was rewritten — not the prose inside `description`.
    assert "enabled: true" in text
    assert "when memory enabled: true is set" in text
    assert changed["memory_enabled"] is True


def test_completion_marker_round_trip(tmp_path: Path) -> None:
    marker = tmp_path / "onboarded.json"
    assert is_onboarded(marker) is False
    mark_onboarded(
        marker,
        openai_key_configured=True,
        memory_enabled=False,
        web_search_enabled=False,
    )
    assert is_onboarded(marker) is True
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["openai_key_configured"] is True
    assert payload["version"] == 1


async def test_wizard_full_happy_path(tmp_path: Path) -> None:
    """End-to-end: wizard captures inputs, writes profile/.env/app.yaml/marker."""

    app_yaml = tmp_path / "config" / "app.yaml"
    app_yaml.parent.mkdir(parents=True)
    app_yaml.write_text(
        "active_provider: openai\nmemory:\n  enabled: false\n", encoding="utf-8"
    )

    public_iter = iter(
        [
            "Tim",
            "",
            "Engineer",
            "Python",
            "",
            "1",
            "n",
            "n",
        ]
    )
    secret_iter = iter(["sk-abc"])

    wizard = OnboardingWizard(
        root=tmp_path,
        input_fn=lambda _p: next(public_iter),
        output_fn=lambda *_a, **_k: None,
        secret_input_fn=lambda _p: next(secret_iter),
    )
    answers = await wizard.run()

    assert answers.name == "Tim"
    assert answers.openai_api_key == "sk-abc"
    profile_text = (tmp_path / ".feather" / "user.md").read_text(encoding="utf-8")
    assert "name: Tim" in profile_text
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=sk-abc" in env_text
    assert (tmp_path / ".feather" / "onboarded.json").exists()


async def test_wizard_re_prompts_on_empty_name(tmp_path: Path) -> None:
    """Empty required answers re-prompt rather than terminate."""

    app_yaml = tmp_path / "config" / "app.yaml"
    app_yaml.parent.mkdir(parents=True)
    app_yaml.write_text("active_provider: openai\nmemory:\n  enabled: false\n", encoding="utf-8")

    public_iter = iter(
        [
            "",        # empty name
            "Tim",     # retry
            "",        # preferred
            "",        # role
            "",        # expertise
            "",        # about
            "1",       # provider
            "n",       # memory
            "n",       # web search
        ]
    )
    secret_iter = iter(["sk-abc"])

    wizard = OnboardingWizard(
        root=tmp_path,
        input_fn=lambda _p: next(public_iter),
        output_fn=lambda *_a, **_k: None,
        secret_input_fn=lambda _p: next(secret_iter),
    )
    answers = await wizard.run()
    assert answers.name == "Tim"


async def test_wizard_memory_branch_collects_keys(tmp_path: Path) -> None:
    """Choosing memory pulls in Qdrant URL + Gemini key and flips app.yaml."""

    app_yaml = tmp_path / "config" / "app.yaml"
    app_yaml.parent.mkdir(parents=True)
    app_yaml.write_text("active_provider: openai\nmemory:\n  enabled: false\n", encoding="utf-8")

    public_iter = iter(
        [
            "Tim", "", "", "", "",
            "1",
            "y",
            "http://localhost:6333",
            "n",
        ]
    )
    secret_iter = iter(["sk-abc", "", "gem-key"])
    wizard = OnboardingWizard(
        root=tmp_path,
        input_fn=lambda _p: next(public_iter),
        output_fn=lambda *_a, **_k: None,
        secret_input_fn=lambda _p: next(secret_iter),
    )
    answers = await wizard.run()
    assert answers.memory_enabled is True
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "GEMINI_API_KEY=gem-key" in env_text
    assert "QDRANT_URL=http://localhost:6333" in env_text
    yaml_text = app_yaml.read_text(encoding="utf-8")
    assert "enabled: true" in yaml_text


async def test_wizard_openrouter_branch_collects_key(tmp_path: Path) -> None:
    """Selecting OpenRouter requires the OPEN_ROUTER_API_KEY and toggles the provider."""

    app_yaml = tmp_path / "config" / "app.yaml"
    app_yaml.parent.mkdir(parents=True)
    app_yaml.write_text("active_provider: openai\nmemory:\n  enabled: false\n", encoding="utf-8")

    public_iter = iter(
        [
            "Tim", "", "", "", "",
            "2",
            "n",
            "n",
        ]
    )
    secret_iter = iter(["sk-abc", "or-key"])
    wizard = OnboardingWizard(
        root=tmp_path,
        input_fn=lambda _p: next(public_iter),
        output_fn=lambda *_a, **_k: None,
        secret_input_fn=lambda _p: next(secret_iter),
    )
    answers = await wizard.run()
    assert answers.provider == "openrouter"
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "OPEN_ROUTER_API_KEY=or-key" in env_text
    yaml_text = app_yaml.read_text(encoding="utf-8")
    assert "active_provider: openrouter" in yaml_text


async def test_maybe_run_onboarding_skips_when_marker_present(tmp_path: Path) -> None:
    """An existing marker short-circuits the wizard."""

    (tmp_path / ".feather").mkdir()
    (tmp_path / ".feather" / "onboarded.json").write_text("{\"version\":1}\n", encoding="utf-8")
    (tmp_path / ".feather" / "user.md").write_text("---\nname: Tim\n---\n", encoding="utf-8")

    invoked = {"n": 0}

    class StubWizard:
        def __init__(self, **_kwargs):
            invoked["n"] += 1

        async def run(self):
            raise AssertionError("wizard should not run")

    result = await maybe_run_onboarding(tmp_path, wizard_factory=StubWizard)
    assert result is None
    assert invoked["n"] == 0


async def test_maybe_run_onboarding_runs_when_user_md_missing(tmp_path: Path) -> None:
    invoked = {"n": 0}

    class StubWizard:
        def __init__(self, **_kwargs):
            invoked["n"] += 1

        async def run(self):
            return OnboardingAnswers(name="Tim", openai_api_key="sk-x")

    result = await maybe_run_onboarding(tmp_path, wizard_factory=StubWizard)
    assert invoked["n"] == 1
    assert isinstance(result, OnboardingAnswers)


async def test_maybe_run_onboarding_skip_flag_short_circuits(tmp_path: Path) -> None:
    """``skip=True`` never runs the wizard, even with no profile present."""

    class StubWizard:
        def __init__(self, **_kwargs):
            raise AssertionError("wizard should not run")

        async def run(self):
            raise AssertionError("wizard should not run")

    result = await maybe_run_onboarding(tmp_path, skip=True, wizard_factory=StubWizard)
    assert result is None


def test_mask_secret_returns_partial_view() -> None:
    """``_mask_secret`` keeps first/last few chars and masks the middle."""

    from feather.onboarding import _mask_secret

    masked = _mask_secret("sk-proj-abcdefghijklmnopqrstuvwxyz1234")
    assert masked.startswith("sk-")
    assert masked.endswith("234")
    assert "*" * 6 in masked
    # Mask must not contain the actual middle of the key.
    assert "abcdefghij" not in masked


def test_mask_secret_redacts_short_values() -> None:
    """Short values must not leak any of their characters."""

    from feather.onboarding import _mask_secret

    assert _mask_secret("short") == "***"
    assert _mask_secret("abcdefg") == "***"


async def test_wizard_echoes_masked_confirmation_after_each_secret(tmp_path: Path, monkeypatch) -> None:
    """Every API-key prompt must echo a masked confirmation line."""

    # Cloud path probes /readyz; stub so the wizard doesn't hit the network.
    import feather.onboarding as ob
    monkeypatch.setattr(ob, "_probe_qdrant_url", lambda url, **_kw: True)
    monkeypatch.delenv("QDRANT_URL", raising=False)

    app_yaml = tmp_path / "config" / "app.yaml"
    app_yaml.parent.mkdir(parents=True)
    app_yaml.write_text(
        "active_provider: openai\nmemory:\n  enabled: false\n", encoding="utf-8"
    )
    output_lines: list[str] = []
    public_iter = iter(
        [
            "Tim", "", "", "", "",
            "1",
            "y",                      # enable memory
            "3",                      # cloud Qdrant — the only path that still
                                     # prompts for QDRANT_API_KEY
            "https://qdrant.example",
            "y",
        ]
    )
    secret_iter = iter([
        "sk-openaikey-1234567890abcdef",
        "qdrant-secret-key-9876543210xyz",
        "gem-key-1234567890abcdefghij",
        "parallel-key-zzzzzzzzzzzzzz999",
    ])
    wizard = OnboardingWizard(
        root=tmp_path,
        input_fn=lambda _p: next(public_iter),
        output_fn=lambda message: output_lines.append(message),
        secret_input_fn=lambda _p: next(secret_iter),
    )
    await wizard.run()

    full_output = "\n".join(output_lines)
    # No raw secret leaks into stdout.
    assert "openaikey-1234567890" not in full_output
    assert "qdrant-secret" not in full_output
    assert "gem-key-1234567890" not in full_output
    assert "parallel-key-zzzz" not in full_output
    # All four keys got a masked echo (one ``******`` block per key).
    assert full_output.count("******") >= 4
    # And the masked echoes preserve the prefix so the user can confirm
    # they pasted the right key family.
    assert "sk-" in full_output


async def test_wizard_uses_secret_input_for_api_keys(tmp_path: Path) -> None:
    """API keys must come through the secret-input callable, not stdout-echoing input."""

    app_yaml = tmp_path / "config" / "app.yaml"
    app_yaml.parent.mkdir(parents=True)
    app_yaml.write_text(
        "active_provider: openai\nmemory:\n  enabled: false\n", encoding="utf-8"
    )

    secret_calls: list[str] = []
    public_calls: list[str] = []

    public_iter = iter(["Tim", "", "", "", "", "1", "n", "n"])
    secret_iter = iter(["sk-secret"])

    def fake_input(prompt: str) -> str:
        public_calls.append(prompt)
        return next(public_iter)

    def fake_secret_input(prompt: str) -> str:
        secret_calls.append(prompt)
        return next(secret_iter)

    wizard = OnboardingWizard(
        root=tmp_path,
        input_fn=fake_input,
        output_fn=lambda *_a, **_k: None,
        secret_input_fn=fake_secret_input,
    )
    answers = await wizard.run()
    assert answers.openai_api_key == "sk-secret"
    # The OPENAI_API_KEY prompt must go through the secret callable, not
    # the public one.
    assert any("OPENAI_API_KEY" in call for call in secret_calls)
    assert not any("OPENAI_API_KEY" in call for call in public_calls)


async def test_wizard_force_re_run_updates_existing_profile_fields(tmp_path: Path) -> None:
    """A second run with the same profile present must not silently discard answers."""

    app_yaml = tmp_path / "config" / "app.yaml"
    app_yaml.parent.mkdir(parents=True)
    app_yaml.write_text(
        "active_provider: openai\nmemory:\n  enabled: false\n", encoding="utf-8"
    )
    profile_path = tmp_path / ".feather" / "user.md"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(
        "---\nname: OldName\nrole: OldRole\ncreated_at: 2026-01-01T00:00:00+00:00\n"
        "updated_at: 2026-01-01T00:00:00+00:00\n---\n",
        encoding="utf-8",
    )

    public_iter = iter(
        [
            "NewName",        # name (overwrite)
            "",                # preferred
            "NewRole",         # role (overwrite)
            "",                # expertise
            "",                # about
            "1",
            "n",
            "n",
        ]
    )
    secret_iter = iter(["sk-new"])
    wizard = OnboardingWizard(
        root=tmp_path,
        input_fn=lambda _p: next(public_iter),
        output_fn=lambda *_a, **_k: None,
        secret_input_fn=lambda _p: next(secret_iter),
    )
    await wizard.run()

    rendered = profile_path.read_text(encoding="utf-8")
    assert "name: NewName" in rendered
    assert "role: NewRole" in rendered
    assert "OldName" not in rendered
    assert "OldRole" not in rendered


async def test_maybe_run_onboarding_force_runs_even_when_marker_present(tmp_path: Path) -> None:
    (tmp_path / ".feather").mkdir()
    (tmp_path / ".feather" / "onboarded.json").write_text("{\"version\":1}\n", encoding="utf-8")
    (tmp_path / ".feather" / "user.md").write_text("---\nname: Tim\n---\n", encoding="utf-8")

    invoked = {"n": 0}

    class StubWizard:
        def __init__(self, **_kwargs):
            invoked["n"] += 1

        async def run(self):
            return OnboardingAnswers(name="Tim", openai_api_key="sk-x")

    await maybe_run_onboarding(tmp_path, force=True, wizard_factory=StubWizard)
    assert invoked["n"] == 1


# ---- Local Qdrant Docker management -----------------------------------------


def test_docker_available_false_when_cli_missing() -> None:
    def runner(_cmd):
        raise FileNotFoundError("docker not on PATH")

    assert docker_available(runner=runner) is False


def test_docker_available_true_when_version_returns_zero() -> None:
    calls: list[list[str]] = []

    def runner(cmd):
        calls.append(cmd)
        return _ok("Docker version 24.0.0")

    assert docker_available(runner=runner) is True
    assert calls == [["docker", "--version"]]


def test_qdrant_container_state_returns_running() -> None:
    def runner(_cmd):
        return _ok("running\n")

    assert qdrant_container_state(runner=runner).state == "running"


def test_qdrant_container_state_classifies_stopped_states() -> None:
    for raw in ("exited", "created", "paused", "dead", "restarting"):
        def runner(_cmd, raw=raw):
            return _ok(f"{raw}\n")

        assert qdrant_container_state(runner=runner).state == "stopped"


def test_qdrant_container_state_returns_absent_when_filter_empty() -> None:
    def runner(_cmd):
        return _ok("")

    assert qdrant_container_state(runner=runner).state == "absent"


def test_ensure_local_qdrant_starts_fresh_container_when_absent() -> None:
    calls: list[list[str]] = []

    def runner(cmd):
        calls.append(cmd)
        if cmd[1] == "--version":
            return _ok("Docker version 24.0.0")
        if cmd[1] == "ps":
            return _ok("")  # absent
        if cmd[1] == "run":
            return _ok("container_id_xyz")
        return _fail("unexpected docker call")

    url = ensure_local_qdrant_container(
        say=lambda _m: None,
        runner=runner,
        ready_checker=lambda _t: True,
    )
    assert url == "http://localhost:6333"
    run_calls = [c for c in calls if c[1] == "run"]
    assert len(run_calls) == 1
    flat = " ".join(run_calls[0])
    assert "feather-qdrant" in flat
    assert "127.0.0.1:6333:6333" in flat
    assert "feather-qdrant-data" in flat


def test_ensure_local_qdrant_starts_existing_stopped_container() -> None:
    calls: list[list[str]] = []

    def runner(cmd):
        calls.append(cmd)
        if cmd[1] == "--version":
            return _ok("ok")
        if cmd[1] == "ps":
            return _ok("exited\n")
        if cmd[1] == "start":
            return _ok("feather-qdrant")
        return _fail("unexpected docker call")

    ensure_local_qdrant_container(
        say=lambda _m: None,
        runner=runner,
        ready_checker=lambda _t: True,
    )
    assert any(c[:2] == ["docker", "start"] for c in calls)
    assert not any(c[:2] == ["docker", "run"] for c in calls)


def test_ensure_local_qdrant_skips_start_when_already_running() -> None:
    calls: list[list[str]] = []

    def runner(cmd):
        calls.append(cmd)
        if cmd[1] == "--version":
            return _ok("ok")
        if cmd[1] == "ps":
            return _ok("running\n")
        return _fail("unexpected")

    ensure_local_qdrant_container(
        say=lambda _m: None,
        runner=runner,
        ready_checker=lambda _t: True,
    )
    assert not any(c[:2] in (["docker", "run"], ["docker", "start"]) for c in calls)


def test_ensure_local_qdrant_raises_when_docker_missing() -> None:
    def runner(_cmd):
        raise FileNotFoundError("docker missing")

    import pytest

    with pytest.raises(DockerNotAvailable):
        ensure_local_qdrant_container(
            say=lambda _m: None,
            runner=runner,
            ready_checker=lambda _t: True,
        )


def test_ensure_local_qdrant_raises_when_run_fails() -> None:
    def runner(cmd):
        if cmd[1] == "--version":
            return _ok("ok")
        if cmd[1] == "ps":
            return _ok("")
        return _fail("port already in use")

    import pytest

    with pytest.raises(QdrantStartFailed) as exc:
        ensure_local_qdrant_container(
            say=lambda _m: None,
            runner=runner,
            ready_checker=lambda _t: True,
        )
    assert "port already in use" in str(exc.value)


def test_ensure_local_qdrant_raises_when_never_ready() -> None:
    def runner(cmd):
        if cmd[1] == "--version":
            return _ok("ok")
        if cmd[1] == "ps":
            return _ok("")
        return _ok("started")

    import pytest

    with pytest.raises(QdrantStartFailed) as exc:
        ensure_local_qdrant_container(
            say=lambda _m: None,
            runner=runner,
            ready_checker=lambda _t: False,
            ready_timeout_s=0.1,
        )
    assert "did not respond" in str(exc.value).lower()


# ---- Wizard memory branch (deployment choice) -------------------------------


async def test_wizard_local_docker_path_uses_injected_launcher(tmp_path: Path) -> None:
    """Default deployment choice spins up the local Qdrant container."""

    app_yaml = tmp_path / "config" / "app.yaml"
    app_yaml.parent.mkdir(parents=True)
    app_yaml.write_text(
        "active_provider: openai\nmemory:\n  enabled: false\n",
        encoding="utf-8",
    )

    public_iter = iter(
        [
            "Tim", "", "", "", "",
            "1",
            "y",
            "",                     # default deployment choice (1 = local-docker)
            "n",
        ]
    )
    secret_iter = iter(["sk-abc", "gem-key"])
    launcher_calls: list[str] = []

    def fake_launcher(say):
        say("LAUNCHED")
        launcher_calls.append("called")
        return "http://localhost:6333"

    wizard = OnboardingWizard(
        root=tmp_path,
        input_fn=lambda _p: next(public_iter),
        output_fn=lambda *_a, **_k: None,
        secret_input_fn=lambda _p: next(secret_iter),
        qdrant_launcher=fake_launcher,
    )

    answers = await wizard.run()

    assert answers.memory_enabled is True
    assert answers.qdrant_mode == "local-docker"
    assert answers.qdrant_url == "http://localhost:6333"
    assert answers.qdrant_api_key == ""
    assert launcher_calls == ["called"]


async def test_wizard_local_existing_path_skips_launcher(tmp_path: Path) -> None:
    """Choosing option 2 records localhost without invoking Docker."""

    app_yaml = tmp_path / "config" / "app.yaml"
    app_yaml.parent.mkdir(parents=True)
    app_yaml.write_text(
        "active_provider: openai\nmemory:\n  enabled: false\n",
        encoding="utf-8",
    )

    public_iter = iter(
        [
            "Tim", "", "", "", "",
            "1",
            "y",
            "2",
            "n",
        ]
    )
    secret_iter = iter(["sk-abc", "gem-key"])
    launcher_calls: list[str] = []

    wizard = OnboardingWizard(
        root=tmp_path,
        input_fn=lambda _p: next(public_iter),
        output_fn=lambda *_a, **_k: None,
        secret_input_fn=lambda _p: next(secret_iter),
        qdrant_launcher=lambda say: launcher_calls.append("never") or "x",
    )

    answers = await wizard.run()

    assert answers.qdrant_mode == "local-existing"
    assert answers.qdrant_url == "http://localhost:6333"
    assert answers.qdrant_api_key == ""
    assert launcher_calls == []


async def test_wizard_cloud_path_collects_url_and_key(tmp_path: Path, monkeypatch) -> None:
    """Choosing option 3 prompts for URL + optional API key."""

    # Stub the URL reachability probe so the wizard doesn't hit the network.
    import feather.onboarding as ob
    monkeypatch.setattr(ob, "_probe_qdrant_url", lambda url, **_kw: True)
    monkeypatch.delenv("QDRANT_URL", raising=False)

    app_yaml = tmp_path / "config" / "app.yaml"
    app_yaml.parent.mkdir(parents=True)
    app_yaml.write_text(
        "active_provider: openai\nmemory:\n  enabled: false\n",
        encoding="utf-8",
    )

    public_iter = iter(
        [
            "Tim", "", "", "", "",
            "1",
            "y",
            "3",
            "https://qdrant.example",
            "n",
        ]
    )
    secret_iter = iter(["sk-abc", "qdrant-cloud-key", "gem-key"])
    wizard = OnboardingWizard(
        root=tmp_path,
        input_fn=lambda _p: next(public_iter),
        output_fn=lambda *_a, **_k: None,
        secret_input_fn=lambda _p: next(secret_iter),
        qdrant_launcher=lambda say: (_ for _ in ()).throw(
            AssertionError("launcher should not run")
        ),
    )

    answers = await wizard.run()

    assert answers.qdrant_mode == "cloud"
    assert answers.qdrant_url == "https://qdrant.example"
    assert answers.qdrant_api_key == "qdrant-cloud-key"
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "QDRANT_URL=https://qdrant.example" in env_text
    assert "QDRANT_API_KEY=qdrant-cloud-key" in env_text


async def test_wizard_short_circuits_when_qdrant_url_in_env(
    tmp_path: Path, monkeypatch
) -> None:
    """Compose / explicit env => skip the 3-way deployment tree."""

    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    app_yaml = tmp_path / "config" / "app.yaml"
    app_yaml.parent.mkdir(parents=True)
    app_yaml.write_text(
        "active_provider: openai\nmemory:\n  enabled: false\n",
        encoding="utf-8",
    )

    public_iter = iter(
        [
            "Tim", "", "", "", "",
            "1",            # provider OpenAI
            "y",            # enable memory
            # No deployment-choice prompt is consumed.
            "n",            # no web search
        ]
    )
    secret_iter = iter(["sk-abc", "gem-key"])
    launcher_calls: list[str] = []

    wizard = OnboardingWizard(
        root=tmp_path,
        input_fn=lambda _p: next(public_iter),
        output_fn=lambda *_a, **_k: None,
        secret_input_fn=lambda _p: next(secret_iter),
        qdrant_launcher=lambda say: launcher_calls.append("nope") or "x",
    )

    answers = await wizard.run()

    assert answers.memory_enabled is True
    assert answers.qdrant_mode == "preconfigured"
    # Empty so the .env writeback never overrides the env var.
    assert answers.qdrant_url == ""
    assert answers.qdrant_api_key == ""
    assert launcher_calls == []
    # And the .env file does not gain a QDRANT_URL line that would
    # override the compose-supplied env on next start.
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "QDRANT_URL=" not in env_text


async def test_wizard_cloud_path_rejects_url_without_scheme(
    tmp_path: Path, monkeypatch
) -> None:
    """Bare hostnames are rejected with a clear error and re-prompt."""

    monkeypatch.delenv("QDRANT_URL", raising=False)
    import feather.onboarding as ob
    monkeypatch.setattr(ob, "_probe_qdrant_url", lambda url, **_kw: True)

    app_yaml = tmp_path / "config" / "app.yaml"
    app_yaml.parent.mkdir(parents=True)
    app_yaml.write_text(
        "active_provider: openai\nmemory:\n  enabled: false\n",
        encoding="utf-8",
    )

    public_iter = iter(
        [
            "Tim", "", "", "", "",
            "1",                       # provider OpenAI
            "y",                       # enable memory
            "3",                       # cloud
            "qdrant.example:6333",     # invalid — no scheme; wizard re-prompts
            "https://qdrant.example",  # valid second attempt
            "n",                       # no web search
        ]
    )
    secret_iter = iter(["sk-abc", "", "gem-key"])
    output_lines: list[str] = []
    wizard = OnboardingWizard(
        root=tmp_path,
        input_fn=lambda _p: next(public_iter),
        output_fn=lambda message: output_lines.append(str(message)),
        secret_input_fn=lambda _p: next(secret_iter),
        qdrant_launcher=lambda say: (_ for _ in ()).throw(
            AssertionError("launcher should not run")
        ),
    )
    answers = await wizard.run()

    assert answers.qdrant_url == "https://qdrant.example"
    rejection_seen = any(
        "must start with http://" in line for line in output_lines
    )
    assert rejection_seen, output_lines


async def test_wizard_cloud_path_unreachable_url_offers_retry_or_proceed(
    tmp_path: Path, monkeypatch
) -> None:
    """If /readyz fails, user can retry or accept the URL anyway."""

    monkeypatch.delenv("QDRANT_URL", raising=False)
    import feather.onboarding as ob
    # Probe always fails so we exercise the unreachable branch.
    monkeypatch.setattr(ob, "_probe_qdrant_url", lambda url, **_kw: False)

    app_yaml = tmp_path / "config" / "app.yaml"
    app_yaml.parent.mkdir(parents=True)
    app_yaml.write_text(
        "active_provider: openai\nmemory:\n  enabled: false\n",
        encoding="utf-8",
    )

    public_iter = iter(
        [
            "Tim", "", "", "", "",
            "1",
            "y",
            "3",
            "https://qdrant.example",  # unreachable in this test
            "y",                       # "use this URL anyway?"
            "n",                       # no web search
        ]
    )
    secret_iter = iter(["sk-abc", "", "gem-key"])
    output_lines: list[str] = []
    wizard = OnboardingWizard(
        root=tmp_path,
        input_fn=lambda _p: next(public_iter),
        output_fn=lambda message: output_lines.append(str(message)),
        secret_input_fn=lambda _p: next(secret_iter),
        qdrant_launcher=lambda say: (_ for _ in ()).throw(
            AssertionError("launcher should not run")
        ),
    )
    answers = await wizard.run()

    assert answers.qdrant_url == "https://qdrant.example"
    # The hint about TLS appears in the output so the user can spot a
    # https-vs-http typo before committing to the URL.
    assert any("TLS" in line for line in output_lines), output_lines


async def test_wizard_local_docker_falls_back_when_docker_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    """Docker missing => fall back to local-existing without crashing."""

    app_yaml = tmp_path / "config" / "app.yaml"
    app_yaml.parent.mkdir(parents=True)
    app_yaml.write_text(
        "active_provider: openai\nmemory:\n  enabled: false\n",
        encoding="utf-8",
    )

    # Make sure no QDRANT_URL is in the env so the 3-way tree runs.
    monkeypatch.delenv("QDRANT_URL", raising=False)

    public_iter = iter(
        [
            "Tim", "", "", "", "",
            "1",
            "y",
            "1",
            "n",
        ]
    )
    secret_iter = iter(["sk-abc", "gem-key"])

    def angry_launcher(_say):
        raise DockerNotAvailable("not installed")

    wizard = OnboardingWizard(
        root=tmp_path,
        input_fn=lambda _p: next(public_iter),
        output_fn=lambda *_a, **_k: None,
        secret_input_fn=lambda _p: next(secret_iter),
        qdrant_launcher=angry_launcher,
    )

    answers = await wizard.run()

    assert answers.memory_enabled is True
    assert answers.qdrant_mode == "local-existing"
    assert answers.qdrant_url == "http://localhost:6333"


# ---- Stop / remove helpers --------------------------------------------------


def test_stop_local_qdrant_is_noop_when_absent() -> None:
    def runner(cmd):
        if cmd[1] == "--version":
            return _ok("ok")
        if cmd[1] == "ps":
            return _ok("")
        raise AssertionError(f"unexpected docker call: {cmd!r}")

    assert (
        stop_local_qdrant_container(say=lambda _m: None, runner=runner)
        == "absent"
    )


def test_stop_local_qdrant_is_noop_when_already_stopped() -> None:
    def runner(cmd):
        if cmd[1] == "--version":
            return _ok("ok")
        if cmd[1] == "ps":
            return _ok("exited\n")
        raise AssertionError(f"unexpected docker call: {cmd!r}")

    assert (
        stop_local_qdrant_container(say=lambda _m: None, runner=runner)
        == "stopped"
    )


def test_stop_local_qdrant_calls_docker_stop_when_running() -> None:
    calls: list[list[str]] = []

    def runner(cmd):
        calls.append(cmd)
        if cmd[1] == "--version":
            return _ok("ok")
        if cmd[1] == "ps":
            return _ok("running\n")
        if cmd[1] == "stop":
            return _ok("feather-qdrant\n")
        raise AssertionError(f"unexpected docker call: {cmd!r}")

    state = stop_local_qdrant_container(say=lambda _m: None, runner=runner)
    assert state == "stopped"
    assert ["docker", "stop", "feather-qdrant"] in calls


def test_stop_local_qdrant_raises_when_docker_missing() -> None:
    import pytest

    def runner(_cmd):
        raise FileNotFoundError("no docker")

    with pytest.raises(DockerNotAvailable):
        stop_local_qdrant_container(say=lambda _m: None, runner=runner)


def test_stop_local_qdrant_raises_when_stop_returns_nonzero() -> None:
    import pytest

    def runner(cmd):
        if cmd[1] == "--version":
            return _ok("ok")
        if cmd[1] == "ps":
            return _ok("running\n")
        if cmd[1] == "stop":
            return _fail("permission denied")
        return _fail("unexpected")

    with pytest.raises(QdrantStartFailed) as exc:
        stop_local_qdrant_container(say=lambda _m: None, runner=runner)
    assert "permission denied" in str(exc.value)


def test_remove_local_qdrant_stops_then_removes() -> None:
    calls: list[list[str]] = []

    def runner(cmd):
        calls.append(cmd)
        if cmd[1] == "--version":
            return _ok("ok")
        if cmd[1] == "ps":
            return _ok("running\n")
        if cmd[1] == "stop":
            return _ok("feather-qdrant\n")
        if cmd[1] == "rm":
            return _ok("feather-qdrant\n")
        return _fail("unexpected")

    say: list[str] = []
    state = remove_local_qdrant_container(say=say.append, runner=runner)

    assert state == "removed"
    assert ["docker", "stop", "feather-qdrant"] in calls
    assert ["docker", "rm", "feather-qdrant"] in calls
    # User is told the volume is preserved.
    assert any("Volume 'feather-qdrant-data' was kept" in line for line in say)


def test_remove_local_qdrant_skips_stop_when_not_running() -> None:
    calls: list[list[str]] = []

    def runner(cmd):
        calls.append(cmd)
        if cmd[1] == "--version":
            return _ok("ok")
        if cmd[1] == "ps":
            return _ok("exited\n")
        if cmd[1] == "rm":
            return _ok("feather-qdrant\n")
        return _fail("unexpected")

    state = remove_local_qdrant_container(say=lambda _m: None, runner=runner)

    assert state == "removed"
    assert not any(c[:2] == ["docker", "stop"] for c in calls)
    assert ["docker", "rm", "feather-qdrant"] in calls


def test_remove_local_qdrant_returns_absent_when_no_container() -> None:
    def runner(cmd):
        if cmd[1] == "--version":
            return _ok("ok")
        if cmd[1] == "ps":
            return _ok("")
        raise AssertionError(f"unexpected docker call: {cmd!r}")

    state = remove_local_qdrant_container(say=lambda _m: None, runner=runner)
    assert state == "absent"


def test_remove_local_qdrant_raises_when_rm_fails() -> None:
    import pytest

    def runner(cmd):
        if cmd[1] == "--version":
            return _ok("ok")
        if cmd[1] == "ps":
            return _ok("exited\n")
        if cmd[1] == "rm":
            return _fail("container in use")
        return _fail("unexpected")

    with pytest.raises(QdrantStartFailed) as exc:
        remove_local_qdrant_container(say=lambda _m: None, runner=runner)
    assert "container in use" in str(exc.value)

