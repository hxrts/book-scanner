"""Pretty-print scanner log lines (from journalctl -o cat) as a status feed.

    journalctl -u scanner -f -o cat | python -m scanner.watch

Each line gets a phase label, coloured by what it asks of you:
cyan = do something, yellow = hold still / don't touch, green = done, red = problem.
"""

import re
import sys

COLOR = sys.stdout.isatty()
RESET, DIM, BOLD = "\033[0m", "\033[2m", "\033[1m"
CYAN, YELLOW, GREEN, RED = "\033[36m", "\033[33m", "\033[32m", "\033[31m"

LINE = re.compile(r"^\d{4}-\d\d-\d\d (\d\d:\d\d:\d\d),\d+ (\w+) [\w.]+: (.*)$")

# Only lines that ask something of you or report a result are shown; steps that
# continue an instruction already on screen (calibrating, capturing) are hidden.
# (pattern, label, colour, rewritten message or None to keep it); first match wins
RULES = [
    (r"something is lying on the mat", "clear", RED, "clear the mat to finish starting up"),
    (r"ready to calibrate", "place", CYAN, "put the open book down to calibrate focus"),
    (r"focus calibrated on first page: (\d+)", "done", GREEN, r"focus calibrated (\1)"),
    (r"take the book away; scanning starts in (\d+) s", "wait", YELLOW, r"take the book away; scanning starts in \1 s"),
    (r"ready to scan", "ready", CYAN, "ready: hold a page still to scan it"),
    (r"page\(s\) in view", "hold", YELLOW, "page in view; hold still"),
    (r"(\d+) page\(s\) captured", "done", GREEN, r"captured \1 page(s); turn the page"),
    (r"page left before capture", "missed", RED, "page moved before capture finished; try again"),
    (r"page (\d+) of (\d+) too blurry", "blurry", RED, r"page \1 of \2 too blurry, discarded; rescan it"),
    (r"uploaded page to Are\.na: block (\S+)", "upload", GREEN, r"uploaded to Are.na (block \1)"),
    (r"ISBN detected: (\S+)", "isbn", CYAN, r"ISBN \1"),
    (r"added to sheet: (.*)", "upload", GREEN, r"added to sheet: \1"),
    (r"cannot open camera", "camera", RED, "webcam not found; check its USB cable and the Pi's power supply"),
    (r"failed|Traceback|error", "error", RED, None),
    (r"camera read failed", "camera", RED, None),
]
# traceback header and the final "SomeError: ..." / "cv2.error: ..." line
PROBLEM = re.compile(r"^Traceback|^[\w.]*(Error|error|Exception)\b:")
# systemd's own lines about the service (journalctl -o cat strips the prefix)
SYSTEMD = [
    (re.compile(r"Failed with result"), "error", RED, "scanner stopped with an error"),
    (re.compile(r"Scheduled restart job"), "wait", YELLOW, "restarting in a few seconds; keep the mat clear"),
    (re.compile(r"^Started scanner"), "wait", YELLOW, "starting up; keep the mat clear"),
    (re.compile(r"^Stopped scanner"), "stop", DIM, "scanner stopped"),
]
QUIET = re.compile(r"learning the empty mat|background learned|calibrating focus|capturing; hold still|"
                   r"focus sweep|job \S+ done|pruned|colour correction matrix|^camera /dev|debug snapshot|"
                   r"Started|Stopped|Consumed|building language model|page was sideways")


def paint(text, *codes):
    return "".join(codes) + text + RESET if COLOR else text


def format_line(raw):
    m = LINE.match(raw.rstrip("\n"))
    if not m:
        raw = raw.strip()
        for pattern, label, colour, text in SYSTEMD:
            if pattern.search(raw):
                return _render(None, label, colour, text)
        # tracebacks and systemd chatter: show problems only
        return paint(f"{'':8}  {raw}", RED) if PROBLEM.search(raw) else None
    time, level, msg = m.groups()
    if QUIET.search(msg):
        return None
    for pattern, label, colour, text in RULES:
        hit = re.search(pattern, msg)
        if hit:
            msg = hit.expand(text) if text else msg
            break
    else:
        label, colour = ("warn", RED) if level in ("WARNING", "ERROR") else ("info", DIM)
    return _render(time, label, colour, msg)


def _render(time, label, colour, msg):
    body = paint(msg, BOLD) if colour in (CYAN, YELLOW, RED) else msg
    return f"{paint(time or ' ' * 8, DIM)}  {paint(label.ljust(6), colour)}  {body}"


def feed(lines):
    """Formatted lines, skipping a message identical to the one just shown
    (the scanner repeats "page in view" every few seconds while you hold still)."""
    last = None
    for raw in lines:
        out = format_line(raw)
        if out is None:
            continue
        message = out[10:]  # without the time
        if message != last:
            yield out
        last = message


def main():
    try:
        for out in feed(sys.stdin):
            print(out, flush=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
