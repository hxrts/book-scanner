"""Fit a colour correction matrix from a scan of a known image.

The webcam renders some hues wrongly under a given light (an orange cover comes
out red). Scanning something whose true colours are known, such as a book
cover from the publisher's site, gives pixel pairs to fit a 3x4 matrix that
maps captured colours to true ones; page.py applies it to every scan.

    python -m scanner.colorfit reference.png ~/scanner-data/done/<job>/page.jpg

The matrix is regularised towards no change, since one cover only samples a
few colours; scanning a colour chart would calibrate more of the gamut.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_OUT = Path("~/.config/scanner/color.json").expanduser()


def register(capture, reference):
    """Warp the reference onto the capture (trying all four rotations). Returns
    the warped reference and a mask of where it's valid, or None."""
    orb = cv2.ORB_create(4000)
    gray = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
    kc, dc = orb.detectAndCompute(gray, None)
    best = None
    for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
        ref = reference if rot is None else cv2.rotate(reference, rot)
        scale = capture.shape[0] / ref.shape[0]
        ref = cv2.resize(ref, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        kr, dr = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
        if dr is None:
            continue
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(dc, dr)
        if len(matches) < 20:
            continue
        H, inliers = cv2.findHomography(np.float32([kr[m.trainIdx].pt for m in matches]),
                                        np.float32([kc[m.queryIdx].pt for m in matches]), cv2.RANSAC, 4.0)
        if H is not None and (best is None or inliers.sum() > best[0]):
            best = (inliers.sum(), H, ref)
    if best is None or best[0] < 50:
        return None
    _, H, ref = best
    size = (capture.shape[1], capture.shape[0])
    valid = cv2.warpPerspective(np.ones(ref.shape[:2], np.uint8), H, size)
    valid = cv2.erode(valid, np.ones((25, 25), np.uint8)) > 0  # away from the edges
    return cv2.warpPerspective(ref, H, size), valid


def _flat(img):
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    return cv2.dilate(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), np.ones((9, 9))) < 4


def fit(capture, reference, valid, strength=0.05):
    """Least-squares 3x4 matrix (BGR, 0-1) from flat regions plus solid blacks,
    regularised towards identity."""
    flat = valid & _flat(capture) & _flat(reference)
    flat = cv2.erode(flat.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    x = cv2.GaussianBlur(capture, (5, 5), 0)[flat].astype(np.float64)
    y = reference[flat].astype(np.float64)
    # fine black print is never "flat", so sample it separately
    black = cv2.erode(((reference.max(axis=2) < 40) & valid).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    xb = cv2.erode(capture, np.ones((3, 3), np.uint8))[black].astype(np.float64)
    yb = reference[black].astype(np.float64)
    if len(xb):
        pick = np.random.default_rng(0).choice(len(xb), min(len(xb), len(x) // 2), replace=False)
        x, y = np.vstack([x, xb[pick]]), np.vstack([y, yb[pick]])
    a = np.hstack([x / 255, np.ones((len(x), 1))])
    reg = np.diag([1.0, 1.0, 1.0, 0.0]) * strength * len(a) / 1000
    prior = np.vstack([np.eye(3), np.zeros((1, 3))])
    m = np.linalg.solve(a.T @ a + reg, a.T @ (y / 255) + reg @ prior)
    error = np.abs(a @ m * 255 - y).mean()
    return m.T, error, len(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reference", type=Path, help="image with the true colours")
    parser.add_argument("capture", type=Path, help="processed scan of the same thing (a page.jpg)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    capture, reference = cv2.imread(str(args.capture)), cv2.imread(str(args.reference))
    if capture is None or reference is None:
        raise SystemExit("cannot read the images")
    registered = register(capture, reference)
    if registered is None:
        raise SystemExit("could not line the reference up with the scan")
    matrix, error, samples = fit(capture, *registered)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"matrix": matrix.tolist()}, indent=1))
    print(f"fitted on {samples} pixels, mean error {error:.1f}/255; wrote {args.out}")
    print("restart the scanner to apply it")


if __name__ == "__main__":
    main()
