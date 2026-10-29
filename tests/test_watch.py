"""Tests for the status feed (scanner.watch) and the `scanner` CLI script.

    python3 -m unittest discover tests
"""

import re
import subprocess
import unittest
from pathlib import Path

from scanner import watch

ROOT = Path(__file__).resolve().parents[1]
MAIN = (ROOT / "scanner" / "main.py").read_text()
CLI = ROOT / "deploy" / "scanner-cli"


def line(msg, level="INFO"):
    return f"2026-10-29 14:00:15,123 {level} scanner: {msg}\n"


def fmt(raw):
    """Format without colour, split into (time, label, message)."""
    out = watch.format_line(raw)
    if out is None:
        return None
    time, label, msg = re.match(r"^(\S+)\s+(\S+)\s+(.*)$", out).groups()
    return time, label, msg


# what the scanner logs at each phase -> (label, message shown)
PHASES = [
    ("ready to calibrate: put the open book down and hold it still", "place",
     "put the open book down to calibrate focus"),
    ("1 page(s) in view, waiting for them to hold still", "hold", "page in view; hold still"),
    ("focus calibrated on first page: 15", "done", "focus calibrated (15)"),
    ("take the book away; scanning starts in 5 s", "wait", "take the book away; scanning starts in 5 s"),
    ("ready to scan", "ready", "ready: hold a page still to scan it"),
    ("2 page(s) captured (sharpness 636), queued for processing", "done", "captured 2 page(s); turn the page"),
    ("page left before capture finished; skipped", "missed", "page moved before capture finished; try again"),
    ("uploaded page to Are.na: block 52008978 (812 chars OCR)", "upload", "uploaded to Are.na (block 52008978)"),
    ("ISBN detected: 9780262537018", "isbn", "ISBN 9780262537018"),
]


class FormatTest(unittest.TestCase):
    def setUp(self):
        watch.COLOR = False

    def test_phases(self):
        for msg, label, shown in PHASES:
            with self.subTest(msg=msg):
                self.assertEqual(fmt(line(msg)), ("14:00:15", label, shown))

    def test_warnings(self):
        self.assertEqual(fmt(line("something is lying on the mat (5% of view); clear it to finish starting up",
                                  "WARNING"))[1:], ("clear", "clear the mat to finish starting up"))
        self.assertEqual(fmt(line("page 2 of 2 too blurry (focus 0.28 < 0.33), discarded; rescan it",
                                  "WARNING"))[1:], ("blurry", "page 2 of 2 too blurry, discarded; rescan it"))
        self.assertEqual(fmt(line("job abc failed (attempt 1), will retry", "ERROR"))[1], "error")
        # unrecognised warnings still show, flagged
        self.assertEqual(fmt(line("something new went wrong", "WARNING"))[1:], ("warn", "something new went wrong"))

    def test_unknown_info_passes_through(self):
        self.assertEqual(fmt(line("some new message"))[1:], ("info", "some new message"))

    def test_quiet(self):
        for msg in ("learning the empty mat; keep it clear", "background learned (searching on the mat)",
                    "calibrating focus; hold still", "capturing; hold still", "focus sweep: {10: 366, 15: 925}", "job 20261028-165030-page-eaf90b done",
                    "colour correction matrix loaded from /home/pi/.config/scanner/color.json",
                    "camera /dev/video0 at 1920x1080", "debug snapshot: 1 page(s), saved to /x"):
            with self.subTest(msg=msg):
                self.assertIsNone(watch.format_line(line(msg)))

    def test_non_log_lines(self):
        self.assertIsNone(watch.format_line("\n"))
        self.assertIn("Traceback", watch.format_line("Traceback (most recent call last):\n"))
        self.assertIn("cv2.error", watch.format_line("cv2.error: OpenCV(4.10) error: (-215)\n"))
        self.assertIn("ValueError", watch.format_line("ValueError: bad value\n"))
        self.assertIsNone(watch.format_line('  File "main.py", line 3, in run\n'))

    def test_service_lines(self):
        self.assertEqual(fmt(line("cannot open camera /dev/video0"))[1], "camera")
        self.assertEqual(watch.format_line("scanner.service: Failed with result 'exit-code'.\n")
                         .split()[0], "error")
        self.assertEqual(watch.format_line("scanner.service: Scheduled restart job, restart counter is at 1.\n")
                         .split()[0], "wait")
        self.assertIn("starting up", watch.format_line("Started scanner.service - Book/page scanner.\n"))

    def test_feed_session(self):
        """A whole calibrate-and-scan session: only instructions and results, no repeats."""
        log = [
            "Started scanner.service - Book/page scanner.\n",
            line("learning the empty mat; keep it clear"),
            line("background learned (searching on the mat)"),
            line("ready to calibrate: put the open book down and hold it still"),
            line("1 page(s) in view, waiting for them to hold still"),
            line("calibrating focus; hold still"),
            line("1 page(s) in view, waiting for them to hold still"),
            line("focus sweep: {10: 366, 15: 925}"),
            line("focus calibrated on first page: 15"),
            line("take the book away; scanning starts in 5 s"),
            line("ready to scan"),
            line("1 page(s) in view, waiting for them to hold still"),
            line("1 page(s) in view, waiting for them to hold still"),
            line("capturing; hold still"),
            line("1 page(s) captured (sharpness 600), queued for processing"),
            line("1 page(s) in view, waiting for them to hold still"),
            line("1 page(s) captured (sharpness 610), queued for processing"),
        ]
        labels = [out.split()[1] if out[0].isdigit() else out.split()[0] for out in watch.feed(log)]
        self.assertEqual(labels, ["wait", "place", "hold", "done", "wait", "ready", "hold", "done", "hold", "done"])

    def test_colour(self):
        self.assertNotIn("\033[", watch.format_line(line("ready to scan")))
        watch.COLOR = True
        out = watch.format_line(line("ready to scan"))
        self.assertIn(watch.CYAN, out)
        self.assertIn(watch.BOLD, out)  # instructions stand out
        self.assertIn(watch.GREEN, watch.format_line(line("focus calibrated on first page: 15")))
        self.assertIn(watch.RED, watch.format_line(line("page left before capture finished; skipped")))


class DriftTest(unittest.TestCase):
    """The feed matches on message text, so the scanner must keep logging it."""

    # fixed text of each phase message as written in main.py, with a sample as logged
    LOGGED = [
        ("ready to calibrate: put the open book down", "ready to calibrate: put the open book down and hold it still"),
        ("page(s) in view, waiting", "1 page(s) in view, waiting for them to hold still"),
        ("focus calibrated on first page: ", "focus calibrated on first page: 15"),
        ("take the book away; scanning starts in ", "take the book away; scanning starts in 5 s"),
        ("ready to scan", "ready to scan"),
        ("page(s) captured (sharpness", "1 page(s) captured (sharpness 600), queued for processing"),
        ("page left before capture finished", "page left before capture finished; skipped"),
        ("something is lying on the mat", "something is lying on the mat (5% of view)"),
        ("too blurry (focus ", "page 2 of 2 too blurry (focus 0.28 < 0.33), discarded; rescan it"),
        ("uploaded page to Are.na: block ", "uploaded page to Are.na: block 1 (10 chars OCR)"),
        ("ISBN detected: ", "ISBN detected: 9780262537018"),
    ]

    def test_main_logs_every_phase(self):
        for text, sample in self.LOGGED:
            with self.subTest(text=text):
                self.assertIn(text, MAIN)
                self.assertIn(text, sample)
                # and the feed recognises it (not the generic fallback)
                self.assertNotIn(fmt(line(sample))[1], ("info", "warn"))

    def test_main_phase_order(self):
        """Ready-to-calibrate is logged before the sweep, and ready-to-scan after it."""
        calibrating = MAIN.index('"calibrating focus; hold still"')
        self.assertLess(MAIN.index('"ready to calibrate'), calibrating)
        # ready-to-scan follows the pause (it is logged once the pause runs out)
        self.assertLess(MAIN.index("paused_until = None\n"), MAIN.rindex('"ready to scan"'))
        self.assertLess(calibrating, MAIN.index('"focus calibrated on first page'))
        # the pause to take the book away comes between calibration and scanning
        self.assertLess(MAIN.index('"focus calibrated on first page'), MAIN.index('"take the book away'))
        self.assertIn("CALIBRATION_PAUSE = 5", MAIN)


class CliTest(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(["bash", str(CLI), *args], capture_output=True, text=True)

    def test_syntax(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(CLI)]).returncode, 0)

    def test_help(self):
        out = self.run_cli()
        self.assertEqual(out.returncode, 0)
        for cmd in ("start", "stop", "restart", "status", "logs", "calibrate", "color", "snapshot", "boot"):
            self.assertIn(cmd, out.stdout)
        self.assertNotIn("\033[", out.stdout)  # no colour codes when not a terminal

    def test_bad_input(self):
        self.assertEqual(self.run_cli("frobnicate").returncode, 1)
        self.assertEqual(self.run_cli("boot", "sideways").returncode, 1)
        self.assertNotEqual(self.run_cli("color").returncode, 0)  # REF is required


if __name__ == "__main__":
    unittest.main()
