"""Camera loop: ISBN barcodes go to Google Sheets, book pages go to Are.na.

    python -m scanner.main                 # run against the webcam
    python -m scanner.main --image p.jpg   # process one still image and exit (no uploads)
"""

import argparse
import logging
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
DESCRIPTION_MAX = 20000


class Scanner:
    def __init__(self, cfg):
        self.cfg = cfg
        page.TESSDATA_DIR = cfg.tessdata_dir or None
        if cfg.lm_corpus:
            page.CORRECTOR = Corrector.load(cfg.lm_corpus, cfg.data_dir / "lm.pickle")
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

    def on_page(self, frame, corners, sharpness):
        if sharpness < self.cfg.min_sharpness:
            log.warning("page too blurry (sharpness %.0f < %.0f), skipped; move the page to retry",
                        sharpness, self.cfg.min_sharpness)
            return
        log.info("page captured (sharpness %.0f), queued for processing", sharpness)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
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

    def open_camera(self):
        cap = cv2.VideoCapture(self.cfg.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # continuous autofocus; FocusPicker picks the sharpest frame
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
        last_barcode_seen = 0.0
        n = 0
        while True:
            ok, frame = cap.read()
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
                    self.on_page(stacker.result(), stacker.corners, stacker.ref_score)
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

            gray, scale = detect.downscale(frame)
            corners = detect.find_page(gray, self.cfg.page_min_area)
            if tracker.update(corners, gray.shape[1], now):
                focus = detect.FocusPicker(frame, corners / scale, now)


def process_still(path, cfg):
    """Offline test: detect and process one image, writing results next to it."""
    if cfg.lm_corpus:
        page.CORRECTOR = Corrector.load(cfg.lm_corpus, cfg.data_dir / "lm.pickle")
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"cannot read {path}")
    isbn = detect.find_isbn(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if isbn:
        print("ISBN:", isbn)
        print(books.lookup(isbn, cfg.user_agent))
        return
    gray, scale = detect.downscale(frame)
    corners = detect.find_page(gray, cfg.page_min_area)
    if corners is None:
        print("no page or ISBN found")
        return
    out = path.with_name(path.stem + ".page.jpg")
    cv2.imwrite(str(out), page.process(frame, corners / scale))
    text, conf, rotated = page.ocr_oriented(out, cfg.ocr_lang)
    print(f"wrote {out}\nOCR confidence {conf:.0f}, rotated {rotated}°\n---\n{text}")


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
