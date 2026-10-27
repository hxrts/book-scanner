# book-scanner

A Raspberry Pi + webcam that watches a desk and:

- **ISBN barcode** → looks the book up on Open Library → appends a row to a Google Sheet.
- **Book page** → waits for focus, crops and flattens the page, corrects colour and lighting, OCRs it → uploads the image, text and metadata to an Are.na channel.

Captures are written to disk before processing and retried every 5 minutes on failure, so nothing is lost offline.

## Hardware

Raspberry Pi 3 B (Raspberry Pi OS Lite, Trixie, 32-bit) and a USB webcam (tested with a Logitech BRIO). A plain, dark surface under the camera gives the most reliable page detection.

## Setup

On the Pi:

```sh
sudo apt install python3-opencv python3-numpy python3-requests python3-pyzbar \
  python3-venv tesseract-ocr tesseract-ocr-eng v4l-utils rsync
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

- Hold a barcode up to the camera. The same ISBN is ignored for 30 s.
- Lay a page flat and hold it still for about 1 s. It's captured once per placement; move or replace the page for the next one.

Pages scanned after a barcode are tagged with that book in their Are.na metadata.

Test on a still image without uploading:

```sh
python -m scanner.main --image photo.jpg
```

## Layout

| Path | Purpose |
|---|---|
| `scanner/main.py` | Camera loop, job handlers, `--image` mode |
| `scanner/detect.py` | Barcode, page and stability detection; focus selection |
| `scanner/page.py` | Perspective warp, deskew, colour correction, OCR with sideways check |
| `scanner/books.py` | Open Library lookup |
| `scanner/sheets.py` | Google Sheets append |
| `scanner/arena.py` | Are.na presigned upload and block creation |
| `scanner/jobs.py` | Durable job queue, retries, FIFO image pruning |
| `scanner/config.py` | Settings from environment |

## Tuning

Set these in `scanner.env`:

| Variable | Default | Meaning |
|---|---|---|
| `SCANNER_PAGE_MIN_AREA` | `0.15` | Smallest page, as a fraction of the frame |
| `SCANNER_PAGE_STABLE_S` | `1.0` | How long a page must be still before capture |
| `SCANNER_MIN_SHARPNESS` | `30` | Focus score below which a capture is skipped (each capture's score is logged) |
| `SCANNER_ISBN_COOLDOWN_S` | `30` | How long a repeat ISBN is ignored |
| `SCANNER_KEEP_MB` | `2048` | Image storage cap; oldest finished scans are deleted first, pending ones never |
| `SCANNER_OCR_LANG` | `eng` | Tesseract language(s), e.g. `eng+deu` |

## Limitations

- Lines curve near the spine of thick books, since only perspective is corrected, not page curl.
- A page partly covered by objects on a busy background is skipped rather than cropped wrongly.
- On a Pi 3, OCR takes 1–12 s per page, and 20–30 s for a page that is sideways.
