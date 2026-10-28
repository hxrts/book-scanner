"""Camera loop: ISBN barcodes go to Google Sheets, book pages go to Are.na.

    python -m scanner.main                 # run against the webcam
    python -m scanner.main --image p.jpg   # process one still image and exit (no uploads)
"""

import argparse
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

log = logging.getLogger("scanner")

BARCODE_EVERY_N_FRAMES = 3
TITLE_MAX = 80
DEFAULT_FOCUS = 15  # BRIO over the scanning mat; calibration searches around the last value
ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
DESCRIPTION_MAX = 20000


def set_focus(device, value):
    subprocess.run(["v4l2-ctl", "-d", device, "-c", "focus_automatic_continuous=0"], check=True)
    subprocess.run(["v4l2-ctl", "-d", device, "-c", f"focus_absolute={value}"], check=True)


def calibrate_focus(cap, device, corners, center, steps=3, step=5):
    """Try focus values within `steps` steps of `center` (the last calibration)
    over the page, extending past the edge if the peak lies there, and return the sharpest. Takes a few seconds; the page should
    be held still meanwhile."""
    def score(value):
        set_focus(device, value)
        time.sleep(0.3)
        for _ in range(4):  # flush frames from before the lens moved
            cap.read()
        return float(np.median([detect.sharpness(cap.read()[1], corners) for _ in range(3)]))

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
        cap.read()
    return best


def find_pages(frame, min_area):
    """Pages in a full-size frame: a fully visible page by its edges, else by its text."""
    gray, small, scale = detect.downscale(frame)
    corners = detect.find_page(gray, min_area, color=small)
    if corners is not None:
        return [corners / scale]
    return detect.find_text_pages(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))


class Scanner:
    def __init__(self, cfg):
        self.cfg = cfg
        page.TESSDATA_DIR = cfg.tessdata_dir or None
        if cfg.lm_corpus:
            page.CORRECTOR = Corrector.load(cfg.lm_corpus, cfg.data_dir / "lm.pickle")
        self.focus = int(cfg.focus) if cfg.focus.isdigit() else None  # fixed lens focus, once known
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

        text, confidence, rotated = page.ocr_oriented(out, self.cfg.ocr_lang)
        if rotated:
            log.info("page was sideways; rotated %d°", rotated)
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
        if sharpness < self.cfg.min_sharpness:
            log.warning("page too blurry (sharpness %.0f < %.0f), skipped; move the page to retry",
                        sharpness, self.cfg.min_sharpness)
            return
        log.info("%d page(s) captured (sharpness %.0f), queued for processing", len(pages), sharpness)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        for corners in pages:  # a spread becomes one job per page, left to right
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

    def read_frame(self, cap):
        ok, frame = cap.read()
        if ok and self.cfg.rotate in ROTATIONS:
            frame = cv2.rotate(frame, ROTATIONS[self.cfg.rotate])
        return ok, frame

    def open_camera(self):
        cap = cv2.VideoCapture(self.cfg.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # the camera sits at a fixed height, so a fixed focus (calibrated on the first
        # page) beats autofocus, which tends to settle on the background
        if self.focus is not None:
            set_focus(self.cfg.device, self.focus)
        else:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # FocusPicker picks the sharpest frame
        if not cap.isOpened():
            raise SystemExit(f"cannot open camera {self.cfg.device}")
        log.info("camera %s at %dx%d", self.cfg.device,
                 cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cap

    def run(self):
        self.jobs.start()
        cap = self.open_camera()
        tracker = detect.StabilityTracker(self.cfg.page_stable_seconds)
        focus = None  # FocusPicker while settling on a capture
        stacker = None  # FrameStacker once focus has settled
        pages = []  # page corners being captured
        last_barcode_seen = 0.0
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
                    if find_pages(result, self.cfg.page_min_area):
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

            if n % BARCODE_EVERY_N_FRAMES == 0:
                isbn = detect.find_isbn(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                if isbn:
                    last_barcode_seen = now
                    self.on_isbn(isbn, now)

            # a visible barcode means a cover, not a page
            if now - last_barcode_seen < 3:
                tracker.update(None, 0, now)
                continue

            found = find_pages(frame, self.cfg.page_min_area)
            # all pages' corners together, so the tracker and focus measure cover every page
            corners = np.vstack(found) if found else None
            if tracker.update(corners, frame.shape[1], now):
                if self.focus is None and self.cfg.focus == "calibrate":
                    focus_file = self.cfg.data_dir / "focus.txt"
                    center = int(focus_file.read_text()) if focus_file.exists() else DEFAULT_FOCUS
                    self.focus = calibrate_focus(cap, self.cfg.device, corners, center)
                    focus_file.write_text(str(self.focus))
                    log.info("focus calibrated on first page: %d", self.focus)
                    # the sweep took a while and the page may have moved or gone, so
                    # wait for a fresh steady placement rather than capturing now
                    tracker = detect.StabilityTracker(self.cfg.page_stable_seconds)
                    continue
                pages = found
                focus = detect.FocusPicker(frame, corners, now)


def process_still(path, cfg):
    """Offline test: detect and process one image, writing results next to it."""
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
    pages = find_pages(frame, cfg.page_min_area)
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
