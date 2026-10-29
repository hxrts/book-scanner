"""Pretty-print scanner log lines (from journalctl -o cat) as a status feed.

    journalctl -u scanner -f -o cat | python -m scanner.watch

Each line gets a phase label, coloured by what it asks of you:
cyan = do something, yellow = hold still / don't touch, green = done, red = problem.
"""

import re
import sys

COLOR = sys.stdout.isatty()
RESET, DIM, BOLD = "\033[0m", "\033[2m", "\033[1m"
CYAN, YELLOW, GREEN, RED, BLUE = "\033[36m", "\033[33m", "\033[32m", "\033[31m", "\033[34m"

LINE = re.compile(r"^\d{4}-\d\d-\d\d (\d\d:\d\d:\d\d),\d+ (\w+) [\w.]+: (.*)$")

# (pattern, label, colour, rewritten message or None to keep it); first match wins
RULES = [
    (r"learning the empty mat", "wait", YELLOW, "learning the empty mat; keep it clear"),
    (r"something is lying on the mat", "clear", RED, "clear the mat to finish starting up"),
    (r"background learned", "setup", BLUE, "mat learned"),
    (r"ready to calibrate", "place", CYAN, "put the open book down and hold it still to calibrate focus"),
    (r"calibrating focus", "hold", YELLOW, "calibrating focus; hold still"),
    (r"focus calibrated on first page: (\d+)", "done", GREEN, r"focus calibrated (\1)"),
    (r"ready to scan", "ready", CYAN, "ready: hold a page still to scan it"),
    (r"page\(s\) in view", "seen", DIM, "page in view; hold still"),
    (r"capturing; hold still", "hold", YELLOW, "capturing; hold still"),
    (r"(\d+) page\(s\) captured", "done", GREEN, r"captured \1 page(s); turn the page"),
    (r"page left before capture", "missed", RED, "page moved before capture finished; try again"),
    (r"too blurry", "missed", RED, "page too blurry; move it and try again"),
    (r"uploaded page to Are\.na: block (\S+)", "upload", GREEN, r"uploaded to Are.na (block \1)"),
    (r"ISBN detected: (\S+)", "isbn", CYAN, r"ISBN \1"),
    (r"added to sheet: (.*)", "upload", GREEN, r"added to sheet: \1"),
    (r"failed|Traceback|error", "error", RED, None),
    (r"camera read failed", "camera", RED, None),
]
# traceback header and the final "SomeError: ..." / "cv2.error: ..." line
PROBLEM = re.compile(r"^Traceback|^[\w.]*(Error|error|Exception)\b:")
QUIET = re.compile(r"focus sweep|job \S+ done|pruned|colour correction matrix|camera /dev|debug snapshot|"
                   r"Started|Stopped|Consumed|building language model|page was sideways")


def paint(text, *codes):
    return "".join(codes) + text + RESET if COLOR else text


def format_line(raw):
    m = LINE.match(raw.rstrip("\n"))
    if not m:
        raw = raw.strip()
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
    body = paint(msg, BOLD) if colour in (CYAN, YELLOW) else msg
    return f"{paint(time, DIM)}  {paint(label.ljust(6), colour)}  {body}"


def main():
    try:
        for raw in sys.stdin:
            out = format_line(raw)
            if out:
                print(out, flush=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
