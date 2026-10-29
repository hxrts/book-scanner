"""Camera loop: ISBN barcodes go to Google Sheets, book pages go to Are.na.

    python -m scanner.main                 # run against the webcam
    python -m scanner.main --image p.jpg   # process one still image and exit (no uploads)
"""

import argparse
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from . import books, detect, page
from .correct import Corrector
from .config import Config
from .jobs import JobQueue, PermanentError
from .led import StatusLed

log = logging.getLogger("scanner")

BARCODE_EVERY_N_FRAMES = 3
TITLE_MAX = 80
DUMP_FLAG = Path("/tmp/scanner-dump")  # touch to save a debug snapshot of detection
CALIBRATION_PAUSE = 5  # seconds after focus calibration before scanning starts
DEFAULT_FOCUS = 15  # BRIO over the scanning mat; calibration searches around the last value
ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
DESCRIPTION_MAX = 20000


def load_color_matrix(path):
    if path.exists():
        page.COLOR_MATRIX = np.array(json.loads(path.read_text())["matrix"], dtype=np.float32)
        log.info("colour correction matrix loaded from %s", path)


def set_focus(device, value):
    v4l2(device, focus_automatic_continuous=0, focus_absolute=value)


def calibrate_focus(read, device, corners, center, steps=3, step=5):
    """Try focus values within `steps` steps of `center` (the last calibration)
    over the page, extending past the edge if the peak lies there, and return the sharpest. Takes a few seconds; the page should
    be held still meanwhile."""
    def score(value):
        set_focus(device, value)
        time.sleep(0.3)
        for _ in range(4):  # flush frames from before the lens moved
            read()
        return float(np.median([detect.sharpness(read(), corners) for _ in range(3)]))

    values = range(max(0, center - steps * step), min(255, center + steps * step) + 1, step)
    scores = {v: score(v) for v in values}
    # if the peak is at the edge (camera moved), keep extending that way
    while True:
        best = max(scores, key=scores.get)
        nxt = best + step if best == max(scores) else best - step if best == min(scores) else None
        if nxt is None or not 0 <= nxt <= 255:
            break
        scores[nxt] = score(nxt)
    log.info("focus sweep: %s", {v: round(s) for v, s in scores.items()})
    best = max(scores, key=scores.get)
    set_focus(device, best)
    time.sleep(0.3)
    for _ in range(4):
        read()
    return best


def find_pages(frame, min_area, margin, background=None, region=None):
    """Pages in a full-size frame: against the learned background if there is one
    (any page content), else by their text, else by their edges."""
    if background is not None:
        return detect.find_pages_by_background(frame, background, region)
    pages = detect.find_text_pages(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), margin=margin)
    if pages:
        return pages
    gray, small, scale = detect.downscale(frame)
    corners = detect.find_page(gray, min_area, color=small)
    return [] if corners is None else [corners / scale]


def v4l2(device, **controls):
    for name, value in controls.items():
        subprocess.run(["v4l2-ctl", "-d", device, "-c", f"{name}={value}"], check=True)


def v4l2_get(device, name):
    out = subprocess.run(["v4l2-ctl", "-d", device, "-C", name], capture_output=True, text=True, check=True)
    return int(out.stdout.split(":")[1])


class Scanner:
    def __init__(self, cfg):
        self.cfg = cfg
        page.TESSDATA_DIR = cfg.tessdata_dir or None
        load_color_matrix(cfg.color_matrix)
        if cfg.lm_corpus:
            page.CORRECTOR = Corrector.load(cfg.lm_corpus, cfg.data_dir / "lm.pickle")
        self.focus = int(cfg.focus) if cfg.focus.isdigit() else None  # fixed lens focus, once known
        self.background = None  # empty-surface frame, learned at startup
        self.region = None  # where books can be (the mat), from the background
        self.last_isbn = {}  # isbn -> time last seen
        self.recent_book = None  # most recent book, attached to subsequent page scans
        self._sheet = None
        self._arena = None
        self.jobs = JobQueue(cfg.data_dir, {"isbn": self.handle_isbn, "page": self.handle_page},
                             max_image_bytes=cfg.keep_mb * 1024 * 1024)

    # lazily connect so a missing credential only blocks its own job type
    @property
    def sheet(self):
        if self._sheet is None:
            from .sheets import BookSheet
            self._sheet = BookSheet(self.cfg.sheet_id, self.cfg.sheet_tab, self.cfg.google_credentials)
        return self._sheet

    @property
    def arena(self):
        if self._arena is None:
            from .arena import Arena
            self._arena = Arena(self.cfg.arena_token, self.cfg.arena_channel, self.cfg.user_agent)
        return self._arena

    # --- job handlers (run on the worker thread) ---

    def handle_isbn(self, job, job_dir):
        book = books.lookup(job["isbn"], self.cfg.user_agent)
        if book is None:
            raise PermanentError(f"Open Library has no record for ISBN {job['isbn']}")
        self.recent_book = book
        self.sheet.append(book)
        log.info("added to sheet: %s — %s", book["title"], book["authors"])
        return book

    def handle_page(self, job, job_dir):
        out = job_dir / "page.jpg"
        if not out.exists():
            raw = cv2.imread(str(job_dir / "raw.jpg"))
            processed = page.process(raw, np.array(job["corners"], dtype=np.float32))
            cv2.imwrite(str(out), processed, [cv2.IMWRITE_JPEG_QUALITY, 92])

        if detect.has_text(cv2.imread(str(out))):
            text, confidence, rotated = page.ocr_oriented(out, self.cfg.ocr_lang)
            if rotated:
                log.info("page was sideways; rotated %d°", rotated)
        else:
            text, confidence, rotated = "", 0.0, 0  # pictures only; nothing to OCR
        (job_dir / "ocr.txt").write_text(text)
        has_text = len(text) >= 20 and confidence >= 40

        first_line = next((l.strip() for l in text.splitlines() if l.strip()), "") if has_text else ""
        title = first_line[:TITLE_MAX] or f"Page scan {job['captured_at']}"
        metadata = {
            "captured_at": job["captured_at"],
            "ocr_confidence": round(confidence, 1),
            "sharpness": job.get("sharpness", 0),
            "rotated_degrees": rotated,
            "ocr_chars": len(text) if has_text else 0,
            "scanner": "rpi-page-scanner",
        }
        book = job.get("book")
        if book:
            metadata.update(book_isbn=book["isbn"], book_title=book["title"][:200], book_authors=book["authors"][:200])

        block = self.arena.upload_image(
            out, title=title,
            description=text[:DESCRIPTION_MAX] if has_text else "",
            metadata=metadata,
            alt_text=text[:500] if has_text else "Scanned book page",
        )
        log.info("uploaded page to Are.na: block %s (%d chars OCR)", block.get("id"), metadata["ocr_chars"])
        return {"block_id": block.get("id"), "ocr_chars": metadata["ocr_chars"]}

    # --- capture side (main thread) ---

    def on_isbn(self, isbn, now):
        # duplicates are suppressed until the barcode has been out of view for the cooldown,
        # so a barcode left in front of the camera isn't re-added
        last_seen = self.last_isbn.get(isbn)
        self.last_isbn[isbn] = now
        if last_seen is not None and now - last_seen < self.cfg.isbn_cooldown_seconds:
            return
        log.info("ISBN detected: %s", isbn)
        self.jobs.submit("isbn", {"isbn": isbn})

    def on_page(self, frame, pages, sharpness):
        # judge each page on its own: one page of a spread can be out of focus (lifted
        # off the mat) while the other is fine
        sharp = []
        for i, corners in enumerate(pages, 1):
            focus = detect.focus_score(page.warp(frame, corners))
            if focus < self.cfg.min_focus:
                log.warning("page %d of %d too blurry (focus %.2f < %.2f), discarded; rescan it",
                            i, len(pages), focus, self.cfg.min_focus)
            else:
                sharp.append(corners)
        if len(sharp) < len(pages):
            self.led.blink()
        if not sharp:
            return
        log.info("%d page(s) captured (sharpness %.0f), queued for processing", len(sharp), sharpness)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        for corners in sharp:  # a spread becomes one job per page, left/top first
            self.jobs.submit(
                "page",
                {
                    "corners": corners.tolist(),
                    "sharpness": round(sharpness, 1),
                    "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "book": self.recent_book,
                },
                files={"raw.jpg": jpg.tobytes()},
            )

    def read_frame(self, cap, tries=30):
        """Next complete frame. The camera is read as raw MJPEG so frames truncated
        in transfer (common over the Pi's USB) can be dropped before decoding
        rather than decoded into garbage."""
        for _ in range(tries):
            ok, buf = cap.read()
            if not ok:
                return False, None
            data = np.trim_zeros(buf.reshape(-1), "b")  # buffers are zero-padded
            if len(data) < 2 or data[-2] != 0xFF or data[-1] != 0xD9:
                continue  # no end-of-image marker: truncated
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if self.cfg.rotate in ROTATIONS:
                frame = cv2.rotate(frame, ROTATIONS[self.cfg.rotate])
            return True, frame
        return False, None

    def open_camera(self):
        cap = cv2.VideoCapture(self.cfg.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # raw MJPEG; read_frame checks and decodes
        # the camera sits at a fixed height, so a fixed focus (calibrated on the first
        # page) beats autofocus, which tends to settle on the background
        if self.focus is not None:
            set_focus(self.cfg.device, self.focus)
        else:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # FocusPicker picks the sharpest frame
        if not cap.isOpened():
            log.error("cannot open camera %s", self.cfg.device)
            raise SystemExit(1)
        v4l2(self.cfg.device, power_line_frequency=1 if self.cfg.mains_hz == 50 else 2,
             auto_exposure=3, white_balance_automatic=1)  # undo any manual settings
        if self.cfg.use_background and self.background is None:  # not again on a reconnect,
            self.learn_background(cap)                            # when a book may be in view
        log.info("camera %s at %dx%d", self.cfg.device,
                 cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cap

    def learn_background(self, cap, settle=3.0):
        """Let exposure settle, then remember the empty surface; pages are found as
        whatever differs from it. Waits until nothing is lying on the mat."""
        log.info("learning the empty mat; keep it clear")
        end = time.monotonic() + settle
        while time.monotonic() < end:
            cap.read()
        warned = False
        while True:
            frames = [detect.background_frame(self.read_frame(cap)[1]) for _ in range(5)]
            background = np.median(frames, axis=0).astype(np.uint8)
            covered = detect.objects_on_mat(background)
            if not covered:
                break
            if not warned:
                log.warning("something is lying on the mat (%.0f%% of view); clear it to finish starting up",
                            covered * 100)
                warned = True
            time.sleep(2)
        self.background = background
        self.region = detect.surface_region(background)
        log.info("background learned%s", "" if self.region is None else " (searching on the mat)")

    def find_pages(self, frame):
        return find_pages(frame, self.cfg.page_min_area, self.cfg.page_margin, self.background, self.region)

    def dump(self, frame, found, tracker):
        """Save the frame (with detected pages outlined) and the background, then clear the flag."""
        out = self.cfg.data_dir / "debug"
        out.mkdir(parents=True, exist_ok=True)
        marked = frame.copy()
        for corners in found:
            cv2.polylines(marked, [corners.astype(np.int32)], True, (0, 0, 255), 3)
        cv2.imwrite(str(out / "frame.jpg"), marked)
        if self.background is not None:
            cv2.imwrite(str(out / "background.png"), self.background)
        DUMP_FLAG.unlink(missing_ok=True)
        log.info("debug snapshot: %d page(s), saved to %s", len(found), out)

    def run(self):
        self.jobs.start()
        self.led = StatusLed()
        cap = self.open_camera()
        if self.focus is None and self.cfg.focus == "calibrate":
            log.info("ready to calibrate: put the open book down and hold it still")
        else:
            log.info("ready to scan")
        tracker = detect.StabilityTracker(self.cfg.page_stable_seconds)
        focus = None  # FocusPicker while settling on a capture
        stacker = None  # FrameStacker once focus has settled
        pages = []  # page corners being captured
        last_barcode_seen = 0.0
        last_seen_log = 0.0
        paused_until = None  # after calibration, time to take the book away
        seen_streak = 0  # consecutive frames with a page; one-off corrupt frames give false ones
        n = 0
        while True:
            ok, frame = self.read_frame(cap)
            if not ok:
                log.warning("camera read failed; reopening")
                cap.release()
                time.sleep(2)
                cap = self.open_camera()
                continue
            n += 1
            now = time.monotonic()

            if stacker is not None:
                if stacker.update(frame, now):
                    log.debug("stacked %d frames", len(stacker.frames))
                    result = stacker.result()
                    # make sure the page is still there (it may have been lifted mid-capture)
                    self.led.off()  # ready for the next page, unless on_page finds it blurry
                    if self.find_pages(result):
                        self.on_page(result, pages, stacker.ref_score)
                    else:
                        log.info("page left before capture finished; skipped")
                    stacker = None
                continue

            if focus is not None:
                if focus.update(frame, now):
                    log.debug("focus scores: %s", [round(s) for s in focus.scores])
                    stacker = detect.FrameStacker(focus.best_frame, focus.corners, focus.best_score,
                                                  self.cfg.stack_frames, now)
                    focus = None
                continue

            if paused_until is not None:
                if now < paused_until:
                    continue
                paused_until = None
                tracker = detect.StabilityTracker(self.cfg.page_stable_seconds)
                log.info("ready to scan")

            if n % BARCODE_EVERY_N_FRAMES == 0:
                isbn = detect.find_isbn(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                if isbn:
                    last_barcode_seen = now
                    self.on_isbn(isbn, now)

            # a visible barcode means a cover, not a page
            if now - last_barcode_seen < 3:
                tracker.update(None, 0, now)
                continue

            try:
                found = self.find_pages(frame)
            except cv2.error:
                log.exception("page detection failed on a frame; skipped")
                found = []
            if DUMP_FLAG.exists():  # debugging: snapshot what the detector sees
                self.dump(frame, found, tracker)
            # follow slow lighting changes while the surface is clear
            if not found and self.background is not None and n % 30 == 0 \
                    and detect.foreground_fraction(frame, self.background) < 0.01:
                self.background = cv2.addWeighted(self.background, 0.9, detect.background_frame(frame), 0.1, 0)
            # all pages' corners together, so the tracker and focus measure cover every page
            corners = np.vstack(found) if found else None
            seen_streak = seen_streak + 1 if found else 0
            # LED: on while a new page is held, off once it's gone (a blurry-page blink
            # stays until the next page is put down)
            if seen_streak >= 3 and not tracker.fired:
                self.led.on()
            elif not found and self.led.state == "on":
                self.led.off()
            if seen_streak >= 3 and now - last_seen_log >= 3 and not tracker.fired:
                log.info("%d page(s) in view, waiting for them to hold still", len(found))
                last_seen_log = now
            if tracker.update(corners, frame.shape[1], now):
                if self.focus is None and self.cfg.focus == "calibrate":
                    focus_file = self.cfg.data_dir / "focus.txt"
                    center = int(focus_file.read_text()) if focus_file.exists() else DEFAULT_FOCUS
                    log.info("calibrating focus; hold still")
                    self.focus = calibrate_focus(lambda: self.read_frame(cap)[1], self.cfg.device,
                                                 corners, center)
                    focus_file.write_text(str(self.focus))
                    log.info("focus calibrated on first page: %d", self.focus)
                    log.info("take the book away; scanning starts in %d s", CALIBRATION_PAUSE)
                    paused_until = time.monotonic() + CALIBRATION_PAUSE
                    self.led.off()
                    # the sweep took a while and the page may have moved or gone, so
                    # wait for a fresh steady placement rather than capturing now
                    tracker = detect.StabilityTracker(self.cfg.page_stable_seconds)
                    continue
                pages = found
                focus = detect.FocusPicker(frame, corners, now)
                log.info("capturing; hold still")


def process_still(path, cfg):
    """Offline test: detect and process one image, writing results next to it."""
    load_color_matrix(cfg.color_matrix)
    if cfg.lm_corpus:
        page.CORRECTOR = Corrector.load(cfg.lm_corpus, cfg.data_dir / "lm.pickle")
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"cannot read {path}")
    if cfg.rotate in ROTATIONS:
        frame = cv2.rotate(frame, ROTATIONS[cfg.rotate])
    isbn = detect.find_isbn(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if isbn:
        print("ISBN:", isbn)
        print(books.lookup(isbn, cfg.user_agent))
        return
    pages = find_pages(frame, cfg.page_min_area, cfg.page_margin)
    if not pages:
        print("no page or ISBN found")
        return
    for i, corners in enumerate(pages):
        out = path.with_name(f"{path.stem}.page{i}.jpg")
        cv2.imwrite(str(out), page.process(frame, corners))
        text, conf, rotated = page.ocr_oriented(out, cfg.ocr_lang)
        print(f"wrote {out}\nOCR confidence {conf:.0f}, rotated {rotated}°\n---\n{text}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, help="process a single still image instead of the camera")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config()
    if args.image:
        process_still(args.image, cfg)
    else:
        Scanner(cfg).run()


if __name__ == "__main__":
    main()
