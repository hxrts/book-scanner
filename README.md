# book-scanner

A Raspberry Pi + webcam system that watches a surface and performs one of two functions depending on what is in view:

- **ISBN barcode** → looks the book up on Open Library → appends a row to a Google Sheet.
- **Book page** → finds the page (by its edges, or by its block of text when hands hold the book open), averages a few frames, crops and flattens it, corrects colour and lighting, OCRs it and fixes uncertain words using their context → uploads the image, text and metadata to an Are.na channel.

Captures are written to disk before processing and retried every 5 minutes on failure, so nothing is lost offline.

## Hardware

Raspberry Pi 3 B (Raspberry Pi OS Lite, Trixie, 32-bit) and a USB webcam (tested with a Logitech BRIO). A plain, dark surface under the camera gives the most reliable page detection. Set `SCANNER_ROTATE` if the camera sees pages upside down or sideways.

**OCR quality depends mostly on how large the text is in the frame.** The BRIO tops out at 1080p over the Pi's USB 2.0, so mount the camera close enough that a single page fills most of the frame. In testing, that gives about 1% character errors, while a two-page spread with the same type gives about 25%.

## Setup

On the Pi:

```sh
sudo apt install python3-opencv python3-numpy python3-requests python3-pyzbar \
  python3-venv tesseract-ocr tesseract-ocr-eng v4l-utils rsync wamerican
```

From your machine:

```sh
./deploy.sh            # syncs code to pi@raspberrypi.local, builds venv, installs service
```

Then fill in `~/.config/scanner/scanner.env` on the Pi (template: `deploy/scanner.env.example`):

- **Google Sheets:** create a service account with the Sheets API enabled, put its JSON key at `~/.config/scanner/service-account.json`, and share the sheet with its `client_email` as Editor. Set `GOOGLE_SHEET_ID`.
- **Are.na:** a personal access token with **write** scope (`ARENA_TOKEN`) and the channel slug (`ARENA_CHANNEL`).

Run `./deploy.sh` again to start the service. Logs: `journalctl -u scanner -f`.

## Usage

- Hold a barcode up to the camera. Rescanning the same ISBN works once it has been out of view for 1 min.
- **First page after start:** the lens sweeps focus over it for a few seconds, locks the sharpest setting, and saves it as the starting point for next time. Hold still until the log shows `focus calibrated`; that placement only calibrates and isn't uploaded.
- Hold the book open (hands along the edges are fine, but keep them off the text) and keep it still for about 4 s. Each visible page becomes its own Are.na block, so an open spread gives two, left page first. One capture per placement; move or turn the page for the next one.

Recommended timings (Pi 3, BRIO at 1080p):

| Step | Hold for | Why |
|---|---|---|
| Focus calibration (first page after start) | ~10 s | 1 s to settle, then the lens tries ~7 focus values at ~1 s each |
| Each capture | ≥ 4 s | 1 s to settle, ~1 s picking the sharpest frame, ~1–1.5 s averaging frames |
| Between captures | Turn the page once the log shows `page(s) captured` | Moving the book re-arms capture; there's no cooldown, but uploads lag ~1–1.5 min per page as the queue drains |

Pages scanned after a barcode are tagged with that book in their Are.na metadata.

Test on a still image without uploading:

```sh
python -m scanner.main --image photo.jpg
```

## Layout

| Path | Purpose |
|---|---|
| `scanner/main.py` | Camera loop, focus calibration, job handlers, `--image` mode |
| `scanner/detect.py` | Barcode, page (edge- and text-based) and stability detection; focus selection |
| `scanner/page.py` | Perspective warp, deskew, colour correction, OCR with sideways check |
| `scanner/correct.py` | Context-aware correction of uncertain OCR words (bigram language model) |
| `scanner/books.py` | Open Library lookup |
| `scanner/sheets.py` | Google Sheets append |
| `scanner/arena.py` | Are.na presigned upload and block creation |
| `scanner/jobs.py` | Durable job queue, retries, FIFO image pruning |
| `scanner/config.py` | Settings from environment |

## Tuning

Set these in `scanner.env`:

| Variable | Default | Meaning |
|---|---|---|
| `SCANNER_FOCUS` | `calibrate` | `calibrate` (lock on the first page each start), `auto` (continuous autofocus), or a fixed lens value 0–255 |
| `SCANNER_ROTATE` | `0` | Rotate frames clockwise (0/90/180/270) so pages read upright |
| `SCANNER_PAGE_MIN_AREA` | `0.15` | Smallest page, as a fraction of the frame |
| `SCANNER_PAGE_MARGIN` | `0.12` | Padding around a page found by its text, as a fraction of its longer side |
| `SCANNER_PAGE_STABLE_S` | `1.0` | How long a page must be still before capture |
| `SCANNER_MIN_SHARPNESS` | `30` | Focus score below which a capture is skipped (each capture's score is logged) |
| `SCANNER_ISBN_COOLDOWN_S` | `60` | How long a barcode must be out of view before it can be added again |
| `SCANNER_KEEP_MB` | `2048` | Image storage cap; oldest finished scans are deleted first, pending ones never |
| `SCANNER_OCR_LANG` | `eng` | Tesseract language(s), e.g. `eng+deu` |
| `SCANNER_STACK_FRAMES` | `5` | Frames averaged per capture to reduce noise (`1` disables) |
| `SCANNER_LM_CORPUS` | `~/scanner/data/big.txt` | Plain-text corpus for context-aware OCR correction (empty disables); `deploy.sh` downloads it |
| `SCANNER_TESSDATA_DIR` | unset | Directory with [tessdata_best](https://github.com/tesseract-ocr/tessdata_best) models: slightly better OCR, about 3x slower |

## Limitations

- Lines curve near the spine of thick books, since only perspective is corrected, not page curl.
- Pages found by their text are cropped to the text block plus a margin, not the paper edge; a page of only pictures is found only if all four of its edges are visible.
- Text under fingers is lost, so keep hands at the margins.
- On a Pi 3, processing and upload take about 1–1.5 min per page; captures queue while you keep scanning.
