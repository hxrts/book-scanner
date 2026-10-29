"""Run the reference shots through the page pipeline and save sheets to inspect.

    python -m scanner.refcheck test-images/reference /tmp/refcheck [--ocr] [--only 03,07]

For each NN_page.jpg (raw camera JPEG plus NN_page.json) this detects pages
against 00_background.png the way the live scanner does, processes each page,
and writes:
  NN_detect.jpg   the frame with page outlines (red) and the mat region (yellow)
  NN_pK.jpg       each processed page
  NN_pK.txt       OCR text (with --ocr)
  sheet.jpg       all detections and pages, one row per shot
  summary.json    per-shot numbers (pages, sizes, paper fraction, text, timings)
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import detect, page
from .config import Config
from .main import ROTATIONS, load_color_matrix


def load_shot(path, rotate):
    frame = cv2.imread(str(path))
    return cv2.rotate(frame, ROTATIONS[rotate]) if rotate in ROTATIONS else frame


def thumb(img, height):
    return cv2.resize(img, (max(1, int(img.shape[1] * height / img.shape[0])), height), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("refdir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--ocr", action="store_true", help="also OCR pages that have text (slow)")
    parser.add_argument("--only", help="comma-separated shot numbers, e.g. 03,07")
    args = parser.parse_args()
    cfg = Config()
    load_color_matrix(cfg.color_matrix)
    if args.ocr and cfg.tessdata_dir:
        page.TESSDATA_DIR = cfg.tessdata_dir
    args.out.mkdir(parents=True, exist_ok=True)

    background = cv2.imread(str(args.refdir / "00_background.png"))
    region = detect.surface_region(background)
    only = set(args.only.split(",")) if args.only else None
    rows, summary = [], {}
    for meta_path in sorted(args.refdir.glob("[0-9][0-9]_page.json")):
        meta = json.loads(meta_path.read_text())
        num = meta_path.name[:2]
        if only and num not in only:
            continue
        frame = load_shot(meta_path.with_suffix(".jpg"), meta.get("rotate", cfg.rotate))
        t0 = time.monotonic()
        found = detect.find_pages_by_background(frame, background, region)
        t_detect = time.monotonic() - t0

        marked = frame.copy()
        if region is not None:
            outline, _ = cv2.findContours((region == 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            scale = frame.shape[1] / region.shape[1]
            cv2.drawContours(marked, [(c * scale).astype(np.int32) for c in outline], -1, (0, 220, 255), 3)
        for corners in found:
            cv2.polylines(marked, [corners.astype(np.int32)], True, (0, 0, 255), 6)
        cv2.imwrite(str(args.out / f"{num}_detect.jpg"), marked)

        pages, info = [], []
        for k, corners in enumerate(found):
            t0 = time.monotonic()
            img = page.process(frame, corners)
            t_proc = time.monotonic() - t0
            path = args.out / f"{num}_p{k}.jpg"
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            entry = {"size": img.shape[1::-1], "paper": round(page.paper_fraction(img), 2),
                     "has_text": detect.has_text(img), "t_process": round(t_proc, 2)}
            if args.ocr and entry["has_text"]:
                t0 = time.monotonic()
                text, conf, rot = page.ocr_oriented(path, cfg.ocr_lang)
                entry.update(ocr_conf=round(conf, 1), ocr_chars=len(text), rotated=rot,
                             t_ocr=round(time.monotonic() - t0, 1))
                path.with_suffix(".txt").write_text(text)
            pages.append(img)
            info.append(entry)
        summary[num] = {"pages": len(found), "t_detect": round(t_detect, 2), "page_info": info,
                        "note": meta.get("note", "")}
        print(num, json.dumps(summary[num]["page_info"]) if info else "no pages", flush=True)

        h = 300
        row = [thumb(marked, h)] + [thumb(p, h) for p in pages]
        row = np.hstack([np.pad(x, ((0, 0), (0, 6), (0, 0)), constant_values=255) for x in row])
        cv2.putText(row, num, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        rows.append(row)

    width = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.pad(r, ((0, 6), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows])
    cv2.imwrite(str(args.out / "sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
