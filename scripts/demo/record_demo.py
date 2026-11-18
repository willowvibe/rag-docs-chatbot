#!/usr/bin/env python3
"""Record the rag-docs-chatbot demo walkthrough at 1920x1080.

Run this with the system python3: Playwright 1.60 and its Chromium build are
installed there, and Playwright is intentionally not a repository dependency.
Start the app first (the rate-limit scene expects RATE_LIMIT_ENABLED=1 and
PER_IP_PER_HOUR=5), then point DEMO_URL or --url at it.

Each scene holds for the larger of its base duration and its measured narration
clip plus 0.5s, so the voice track and the visuals stay aligned. The timeline is
written to /tmp/rag-demo-tts/holds.json for the assembly step.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "video-work"
DEFAULT_TTS = Path("/tmp/rag-demo-tts")
DEFAULT_URL = "http://localhost:8017"
CARD_DIR = Path(__file__).resolve().parent / "cards"

BASE_HOLDS = [4.0, 10.0, 6.0, 8.0, 4.0, 9.0, 12.0, 11.0, 4.0]

SCENES = [
    "title card",
    "cited answer",
    "citation click",
    "reasoning answer",
    "refusal",
    "rate-limit UX",
    "eval card",
    "pipeline card",
    "end card",
]

QUESTION_LENGTH = (
    "What is the minimum length for a password used as a single-factor authenticator?"
)
QUESTION_MFA = (
    "If a system asks for both a PIN and a password, has it achieved multifactor authentication?"
)
QUESTION_REFUSAL = "What is the capital of France?"
QUESTION_VERIFIER = "What is a verifier?"
QUESTION_CLAIMANT = "What is a claimant?"


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffprobe failed for {path}: {result.stderr.strip()}")
    return float(result.stdout.strip())


def segment_durations(tts: Path) -> list[float]:
    durations = []
    for index in range(len(BASE_HOLDS)):
        segment = tts / f"{index:02d}.mp3"
        if not segment.exists():
            raise SystemExit(f"missing narration segment {segment}")
        durations.append(probe_duration(segment))
    return durations


def build_timeline(tts: Path) -> dict:
    durations = segment_durations(tts)
    holds = [max(base, length + 0.5) for base, length in zip(BASE_HOLDS, durations)]
    starts = []
    cursor = 0.0
    for hold in holds:
        starts.append(round(cursor, 3))
        cursor += hold
    timeline = {
        "scenes": SCENES,
        "audio_durations": [round(length, 3) for length in durations],
        "holds": [round(hold, 3) for hold in holds],
        "scene_starts": starts,
        "total": round(cursor, 3),
    }
    write_timeline(tts, timeline)
    return timeline


def write_timeline(tts: Path, timeline: dict) -> None:
    (tts / "holds.json").write_text(json.dumps(timeline, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=None,
        help=f"running app URL (default: $DEMO_URL or {DEFAULT_URL})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"directory for the raw recording (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--tts",
        type=Path,
        default=DEFAULT_TTS,
        help=f"directory with 00.mp3..08.mp3 (default: {DEFAULT_TTS})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    url = args.url or os.environ.get("DEMO_URL", DEFAULT_URL)
    out = args.out
    tts = args.tts
    out.mkdir(parents=True, exist_ok=True)

    cards = {}
    for name in ("title", "eval", "pipeline", "end"):
        path = CARD_DIR / f"{name}.html"
        if not path.exists():
            raise SystemExit(f"missing card {path}")
        cards[name] = path.resolve().as_uri()

    timeline = build_timeline(tts)
    holds = timeline["holds"]
    starts = timeline["scene_starts"]
    print(f"timeline: {timeline['total']:.2f}s", file=sys.stderr)
    for index, (start, hold) in enumerate(zip(starts, holds)):
        print(f"scene {index}: start {start:.2f}s hold {hold:.2f}s", file=sys.stderr)

    for stale in out.glob("page@*.webm"):
        stale.unlink()

    from playwright.sync_api import Error, sync_playwright

    video_path = None
    poster_at = None
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir=str(out),
            record_video_size={"width": 1920, "height": 1080},
        )
        page = context.new_page()
        page.set_default_timeout(90000)
        page.set_default_navigation_timeout(30000)

        def answer_count() -> int:
            return page.locator(".message.answer").count()

        def bring_latest_into_view() -> None:
            page.evaluate(
                """() => {
                    const transcript = document.querySelector('#transcript');
                    if (transcript) transcript.scrollTop = transcript.scrollHeight;
                    // Zoomed page is taller than the frame: keep the input,
                    // quota line and newest answer in view.
                    window.scrollTo(0, document.documentElement.scrollHeight);
                }"""
            )

        def reveal_quota() -> None:
            page.evaluate(
                """() => {
                    const quota = document.querySelector('#quota');
                    if (quota) quota.scrollIntoView({block: 'center'});
                }"""
            )

        def ask(question: str, type_delay: float | None = None) -> None:
            before = answer_count()
            box = page.locator("#question")
            box.click()
            if type_delay is None:
                box.fill(question)
            else:
                box.press_sequentially(question, delay=type_delay)
            page.locator("#send").click()
            page.wait_for_function(
                """([count]) => {
                    const answers = document.querySelectorAll('.message.answer');
                    const status = document.querySelector('#status');
                    return answers.length > count &&
                        (status?.textContent || '').startsWith('Answer ready');
                }""",
                arg=[before],
            )
            bring_latest_into_view()

        def click_first_citation() -> None:
            button = page.locator(".message.answer").last.locator("button.citation").first
            if button.count() == 0:
                raise SystemExit("latest answer has no citation button")
            button.click()
            page.wait_for_timeout(900)
            focused = page.evaluate(
                """() => {
                    const active = document.querySelector('.source-card.is-active');
                    const element = document.activeElement;
                    return Boolean(
                        active && element && element.classList.contains('source-card')
                    );
                }"""
            )
            if not focused:
                raise SystemExit("citation click did not highlight and focus its source card")

        def lock_and_show_quota() -> None:
            ask(QUESTION_VERIFIER)
            ask(QUESTION_CLAIMANT)
            page.wait_for_function(
                """() => {
                    const input = document.querySelector('#question');
                    const quota = document.querySelector('#quota');
                    return Boolean(
                        input && input.disabled && quota && !quota.hidden &&
                        quota.textContent.trim().length > 0
                    );
                }"""
            )
            reveal_quota()
            page.wait_for_timeout(500)

        def scene(index: int, action) -> float:
            started = time.monotonic()
            action()
            elapsed = time.monotonic() - started
            hold = holds[index]
            print(f"scene {index}: actions {elapsed:.2f}s / hold {hold:.2f}s", file=sys.stderr)
            if elapsed > hold:
                raise SystemExit(
                    f"scene {index} took {elapsed:.2f}s, longer than its "
                    f"{hold:.2f}s hold; narration would desync"
                )
            remaining = hold - elapsed
            if remaining > 0:
                page.wait_for_timeout(int(remaining * 1000))
            return elapsed

        try:
            scene(0, lambda: page.goto(cards["title"], wait_until="load"))
            def open_app_and_ask() -> None:
                # Scene 0 left the title card loaded; return to the app first.
                page.goto(url, wait_until="load")
                # 1:1 app text is unreadable once the video is embedded or
                # shrunk to the README GIF; zoom the app (not the cards).
                page.evaluate("document.documentElement.style.zoom = '1.35'")
                ask(QUESTION_LENGTH, type_delay=35)

            answer_elapsed = scene(1, open_app_and_ask)
            scene(2, click_first_citation)
            scene(3, lambda: ask(QUESTION_MFA))
            scene(4, lambda: ask(QUESTION_REFUSAL))
            scene(5, lock_and_show_quota)
            scene(6, lambda: page.goto(cards["eval"], wait_until="load"))
            scene(7, lambda: page.goto(cards["pipeline"], wait_until="load"))
            scene(8, lambda: page.goto(cards["end"], wait_until="load"))

            poster_at = starts[1] + min(
                answer_elapsed + 0.5, max(0.0, holds[1] - 0.25)
            )

            video = page.video
            if video is not None:
                try:
                    video_path = video.path()
                except Error:
                    video_path = None
        finally:
            context.close()
            browser.close()

    if video_path is None:
        candidates = sorted(out.glob("page@*.webm"), key=lambda path: path.stat().st_mtime)
        if not candidates:
            raise SystemExit("Playwright did not produce a video")
        raw = candidates[-1]
    else:
        raw = Path(video_path)

    target = out / "rag-raw.webm"
    if target.exists():
        target.unlink()
    raw.replace(target)

    recorded = probe_duration(target)
    if abs(recorded - timeline["total"]) > 0.5:
        raise SystemExit(
            f"recording is {recorded:.2f}s but the planned timeline is "
            f"{timeline['total']:.2f}s; narration would desync"
        )

    timeline["poster_at"] = round(float(poster_at), 3)
    write_timeline(tts, timeline)

    print(f"recording: {recorded:.2f}s (timeline {timeline['total']:.2f}s)")
    print(f"poster at: {timeline['poster_at']:.2f}s")
    print(target)


if __name__ == "__main__":
    main()
