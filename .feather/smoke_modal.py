"""Headless smoke test for the /config modal — render it and dump layout state.

Run with: uv run python .feather/smoke_modal.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path


async def smoke() -> int:
    from textual.app import App

    from feather.config import load_app_config
    from feather.config_service import ConfigService
    from feather.paths import FeatherPaths
    from feather.textual_config_screen import ConfigScreen

    tmpdir = Path(tempfile.mkdtemp(prefix="feather-smoke-"))
    paths = FeatherPaths(project_root=tmpdir / "proj", home=tmpdir / "global")
    paths.ensure_global_dirs()
    paths.ensure_project_dirs()
    cfg = load_app_config(paths.project_root, paths=paths)
    service = ConfigService(paths=paths, app_config=cfg)

    class _Host(App):
        async def on_mount(self) -> None:
            await self.push_screen(ConfigScreen(service=service))

    host = _Host()
    findings: list[str] = []

    async with host.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        if not isinstance(screen, ConfigScreen):
            print(f"FAIL: top screen is {type(screen).__name__}, expected ConfigScreen")
            return 1

        # ---- Initial state ----
        try:
            tabs = screen.query_one("#config-tabs")
            sidebar = screen.query_one("#config-sidebar")
            form = screen.query_one("#config-form")
            footer = screen.query_one("#config-footer")
        except Exception as exc:
            print(f"FAIL: required widget missing — {exc}")
            return 1

        findings.append(f"tabs.region: y={tabs.region.y} height={tabs.region.height}")
        findings.append(f"sidebar.region: y={sidebar.region.y} h={sidebar.region.height}")
        findings.append(f"form.region: x={form.region.x} w={form.region.width} y={form.region.y} h={form.region.height}")
        findings.append(f"footer.region: y={footer.region.y} h={footer.region.height}")

        # Tab list check — expect App + Lead + Explore + Research + Validate.
        tab_labels = [t.label for t in screen._tabs]
        expected_tabs = ["App", "Lead", "Explore", "Research", "Validate"]
        if tab_labels == expected_tabs:
            findings.append(f"OK tabs: {tab_labels}")
        else:
            findings.append(f"!! tabs mismatch: got {tab_labels}, expected {expected_tabs}")

        # Footer should sit at the bottom of the modal
        # screen height is 30; modal is 90% so ~27; footer height 1 should be at y ~ near bottom
        if footer.region.y + footer.region.height > 30:
            print(f"FAIL: footer extends beyond terminal height ({footer.region.y}+{footer.region.height} > 30)")
            return 1

        # ---- Drive Enter to open inline editor ----
        await pilot.press("enter")
        await pilot.pause()

        try:
            editor = screen.query_one("#config-inline-editor")
            findings.append(f"editor.region: y={editor.region.y} h={editor.region.height}")

            # NEW: editor should be entirely within the modal container.
            root = screen.query_one("#config-root")
            editor_bottom = editor.region.y + editor.region.height
            root_bottom = root.region.y + root.region.height
            if editor_bottom > root_bottom:
                findings.append(
                    f"!! editor bottom {editor_bottom} extends past modal bottom {root_bottom}"
                )
            else:
                findings.append(
                    f"OK editor within modal: editor [{editor.region.y},{editor_bottom}) "
                    f"vs modal [{root.region.y},{root_bottom})"
                )

            # Editor should NOT overlap the footer
            editor_bottom = editor.region.y + editor.region.height
            footer_top = footer.region.y
            footer_bottom = footer.region.y + footer.region.height
            editor_top = editor.region.y
            # Detect overlap: any y-pixel shared between editor and footer
            overlap_top = max(editor_top, footer_top)
            overlap_bottom = min(editor_bottom, footer_bottom)
            if overlap_bottom > overlap_top:
                findings.append(
                    f"!! OVERLAP: editor occupies y=[{editor_top},{editor_bottom}) "
                    f"and footer occupies y=[{footer_top},{footer_bottom}) — "
                    f"they share {overlap_bottom - overlap_top} row(s)"
                )
            else:
                findings.append(
                    f"OK no editor/footer overlap: editor [{editor_top},{editor_bottom}) "
                    f"vs footer [{footer_top},{footer_bottom})"
                )
        except Exception as exc:
            findings.append(f"editor not mounted after Enter: {exc}")

        # ---- Escape to cancel edit ----
        await pilot.press("escape")
        await pilot.pause()

        try:
            screen.query_one("#config-inline-editor")
            findings.append("!! editor still mounted after Esc — cancel didn't work")
        except Exception:
            findings.append("OK Esc cancelled the inline edit")

        # ---- Modal still alive after edit cancel ----
        if not isinstance(pilot.app.screen, ConfigScreen):
            findings.append("!! modal dismissed by Esc-during-edit (should only cancel edit)")
        else:
            findings.append("OK modal stays open after Esc-cancels-edit")

        # ---- Tab cycling ----
        before = screen._active_tab_index
        await pilot.press("right")
        await pilot.pause()
        after = screen._active_tab_index
        if after != (before + 1) % len(screen._tabs):
            findings.append(f"!! tab cursor didn't advance: {before} → {after}")
        else:
            findings.append(f"OK tab cursor advanced {before} → {after}")

        # ---- Save with empty dirty ----
        await pilot.press("s")
        await pilot.pause()
        # No assertions on banner content; just ensure no crash.
        findings.append("OK save-with-empty-dirty did not crash")

        # ---- Screenshot dump ----
        svg = pilot.app.export_screenshot(title="ConfigScreen smoke")
        svg_path = tmpdir / "modal.svg"
        svg_path.write_text(svg, encoding="utf-8")
        findings.append(f"screenshot: {svg_path}")

    print("\n".join(findings))
    failed = [f for f in findings if f.startswith("!!") or f.startswith("FAIL")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(smoke()))
