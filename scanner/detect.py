"""Per-frame detection: ISBN barcodes and book-page quadrilaterals."""

import cv2
import numpy as np
from pyzbar import pyzbar

DETECT_WIDTH = 640  # frames are downscaled to this width for detection


def isbn13_valid(code):
    if len(code) != 13 or not code.isdigit():
        return False
    total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(code[:12]))
    return (10 - total % 10) % 10 == int(code[12])


def find_isbn(gray):
    """Return an ISBN-13 if a Bookland EAN (978/979) barcode is visible."""
    for symbol in pyzbar.decode(gray, symbols=[pyzbar.ZBarSymbol.EAN13]):
        code = symbol.data.decode("ascii", "ignore")
        if code[:3] in ("978", "979") and isbn13_valid(code):
            return code
    return None


def order_corners(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype=np.float32)


def _quad(contour):
    """Simplify a contour's hull to 4 corners, loosening tolerance for rounded/occluded corners."""
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.045, 0.06):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            return approx
        if len(approx) < 4:
            return None
    return None


def _edge_support(quad, edges):
    """Fraction of the quad's outline that lies on (or within a few px of) an image edge."""
    edges = cv2.dilate(edges, np.ones((7, 7), np.uint8))
    outline = np.zeros_like(edges)
    cv2.polylines(outline, [quad], True, 255, 1)
    on_outline = outline > 0
    return np.count_nonzero(edges[on_outline]) / max(1, np.count_nonzero(on_outline))


def find_page(gray, min_area, max_area=0.98, min_support=0.6):
    """Find the page: the largest convex quadrilateral covering min_area..max_area
    of the frame whose border mostly follows real edges.

    Tries the edge map first and falls back to bright regions (paper on a busy
    background). Returns ordered corners in `gray` coordinates, or None.
    """
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(blur, 40, 120), np.ones((3, 3), np.uint8), iterations=2)
    # paper is usually the brightest large region; close over the text so the page is solid
    bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

    for mask in (edges, bright):
        best, best_area = None, min_area * w * h
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < best_area * 0.5:
                continue
            quad = _quad(c)
            if quad is None or not cv2.isContourConvex(quad):
                continue
            area = cv2.contourArea(quad)
            if best_area < area < max_area * w * h and _edge_support(quad, edges) >= min_support:
                best, best_area = quad, area
        if best is not None:
            return order_corners(best)
    return None


class StabilityTracker:
    """Reports a page once its corners have stayed put for `hold` seconds."""

    def __init__(self, hold, tolerance=0.02):
        self.hold = hold
        self.tolerance = tolerance  # fraction of frame width
        self.anchor = None
        self.since = None
        self.fired = False

    def update(self, corners, width, now):
        if corners is None:
            self.anchor, self.fired = None, False
            return False
        moved = self.anchor is None or np.abs(corners - self.anchor).max() > self.tolerance * width
        if moved:
            self.anchor, self.since, self.fired = corners, now, False
            return False
        if not self.fired and now - self.since >= self.hold:
            self.fired = True  # one capture per placement; re-arms when the page moves or leaves
            return True
        return False


def sharpness(frame, corners, width=800):
    """Focus measure: variance of the Laplacian over the page region, at a fixed scale
    so values are comparable between captures."""
    x, y, w, h = cv2.boundingRect(corners.astype(np.int32))
    crop = cv2.cvtColor(frame[max(y, 0):y + h, max(x, 0):x + w], cv2.COLOR_BGR2GRAY)
    if crop.size == 0:
        return 0.0
    crop = cv2.resize(crop, (width, max(1, int(crop.shape[0] * width / crop.shape[1]))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())


class FocusPicker:
    """After a page settles, watch frames while autofocus works and keep the sharpest.

    Finishes when sharpness stops improving for `patience` frames or `window`
    seconds pass.
    """

    def __init__(self, frame, corners, now, window=1.5, patience=3):
        self.corners = corners
        self.started = now
        self.window = window
        self.patience = patience
        self.best_frame, self.best_score = frame, sharpness(frame, corners)
        self.stale = 0
        self.scores = [self.best_score]

    def update(self, frame, now):
        """Returns True when done; result is in best_frame / best_score."""
        score = sharpness(frame, self.corners)
        self.scores.append(score)
        if score > self.best_score * 1.03:
            self.best_frame, self.best_score, self.stale = frame, score, 0
        else:
            self.stale += 1
        return self.stale >= self.patience or now - self.started >= self.window


def downscale(frame):
    scale = DETECT_WIDTH / frame.shape[1]
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), scale
