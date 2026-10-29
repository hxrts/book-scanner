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


def find_page(gray, min_area, max_area=0.98, min_support=0.6, color=None):
    """Find the page: the largest convex quadrilateral covering min_area..max_area
    of the frame whose border mostly follows real edges.

    Tries the edge map first and falls back to bright regions (paper on a busy
    background). If `color` (the same frame in BGR) is given, the region must also
    look like paper. Returns ordered corners in `gray` coordinates, or None.
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
            if (best_area < area < max_area * w * h and _plausible_page(order_corners(quad), w)
                    and not _touches_frame(order_corners(quad), w, h)
                    and _edge_support(quad, edges) >= min_support
                    and (color is None or _looks_like_paper(color, quad))):
                best, best_area = quad, area
        if best is not None:
            return order_corners(best)
    return None


def _looks_like_paper(color, quad, max_green=0.6):
    """Reject large uniformly green/teal regions (cutting mats, desk pads). Paper is
    neutral or warm; illustrations are mixed, so only a dominant colour counts."""
    mask = np.zeros(color.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, quad.reshape(-1, 2), 255)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[mask > 0]
    green = (hsv[:, 0] >= 35) & (hsv[:, 0] <= 100) & (hsv[:, 1] > 20)
    return green.mean() <= max_green


def _touches_frame(corners, w, h, margin=3):
    """A quad with two or more corners on the frame edge is the frame itself or a
    surface running out of view, not a page whose edges we can crop to."""
    on_edge = [x <= margin or y <= margin or x >= w - 1 - margin or y >= h - 1 - margin for x, y in corners]
    return sum(on_edge) >= 2


def _plausible_page(corners, frame_width, min_side=0.1, min_angle=45, min_side_ratio=0.8):
    """Reject shapes no photographed page could make: repeated corners (really a
    triangle), very short sides, corners far from square, or opposite sides of very
    different length (an overhead camera sees a page as nearly a rectangle; a lopsided
    quad is usually a page edge joined to a hand or the mat)."""
    sides = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    for a, b in ((sides[0], sides[2]), (sides[1], sides[3])):
        if min(a, b) < min_side_ratio * max(a, b):
            return False
    for i in range(4):
        prev, cur, nxt = corners[i - 1], corners[i], corners[(i + 1) % 4]
        a, b = prev - cur, nxt - cur
        if min(np.linalg.norm(a), np.linalg.norm(b)) < min_side * frame_width:
            return False
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        if not min_angle <= np.degrees(np.arccos(np.clip(cos, -1, 1))) <= 180 - min_angle:
            return False
    return True


def find_text_pages(gray, work_width=960, min_rows=15, min_area=0.03, margin=0.05):
    """Find pages by their printed text, for when page edges are hidden (hands
    holding the book open, a spread running out of frame).

    Dark word-shaped blobs on bright paper are grouped into text blocks; blocks
    in the same column are merged (a hand across a page splits it), and each
    column becomes one page, so a spread yields its two pages left to right.
    Returns a list of corner arrays (padded by `margin`) in `gray` coordinates.
    """
    scale = work_width / gray.shape[1]
    g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    H, W = g.shape
    ink = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    ink = cv2.threshold(ink, 30, 255, cv2.THRESH_BINARY)[1]
    words = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(words)
    x, y, w, h = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]
    cx, cy = np.clip(x + w // 2, 0, W - 1), np.clip(y + h // 2, 0, H - 1)
    paper = cv2.dilate(g, np.ones((7, 7), np.uint8))  # brightest nearby pixel = the paper behind a word
    is_word = (h >= 2) & (h <= 14) & (w >= 1.5 * h) & (w <= 0.3 * W) & (paper[cy, cx] > 150)
    is_word[0] = False  # background
    mask = is_word[labels].astype(np.uint8) * 255

    blocks = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 19)))
    nb, block_labels, block_stats, _ = cv2.connectedComponentsWithStats(blocks)
    ids = np.nonzero(is_word)[0]
    owner = block_labels[cy[ids], cx[ids]]
    columns = []
    for b in range(1, nb):
        mine = ids[owner == b]
        bw, bh = block_stats[b, 2:4]
        if bw * bh < min_area * W * H or len(np.unique(cy[mine] // 6)) < min_rows:
            continue
        pts = np.concatenate([np.stack([x[mine], y[mine]], 1),
                              np.stack([x[mine] + w[mine], y[mine] + h[mine]], 1)]).astype(np.float32)
        x0, x1 = pts[:, 0].min(), pts[:, 0].max()
        for i, col in enumerate(columns):
            c0, c1 = col[:, 0].min(), col[:, 0].max()
            if min(x1, c1) - max(x0, c0) > 0.5 * min(x1 - x0, c1 - c0):
                columns[i] = np.vstack([col, pts])
                break
        else:
            columns.append(pts)

    pages = []
    for pts in sorted(columns, key=lambda p: p[:, 0].min()):
        (px, py), (pw, ph), angle = cv2.minAreaRect(pts)
        if angle > 45:
            pw, ph, angle = ph, pw, angle - 90
        pad = margin * max(pw, ph)
        box = cv2.boxPoints(((px, py), (pw + 2 * pad, ph + 2 * pad), angle)) / scale
        pages.append(order_corners(np.clip(box, 0, [gray.shape[1] - 1, gray.shape[0] - 1])))
    return pages


BG_WIDTH = 480  # working width for background subtraction


def background_frame(frame):
    """Downscaled, blurred copy of a frame, as used for background subtraction."""
    scale = BG_WIDTH / frame.shape[1]
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (5, 5), 0)


def _mat_coloured(small):
    """Pixels coloured like a green/teal cutting mat."""
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    return (hsv[..., 0] >= 70) & (hsv[..., 0] <= 100) & (hsv[..., 1] > 25) & (hsv[..., 2] < 170)


def _mat_shape(background):
    """The cutting mat in a background frame: (solid mat mask, its filled convex
    outline), or None if there's no mat."""
    seed = _mat_coloured(background)
    if seed.sum() < 0.05 * seed.size:
        return None
    # lamp light shifts the mat's hue and saturation across it, so grow from the
    # clearly mat-coloured pixels to anything close to this mat's own colour
    lab = cv2.cvtColor(background, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = np.median(lab[seed], axis=0)
    chroma = np.hypot(lab[..., 1] - ref[1], lab[..., 2] - ref[2])
    near = (chroma < 10) & (np.abs(lab[..., 0] - ref[0]) < 45)
    mat = (seed | near).astype(np.uint8) * 255
    mat = cv2.morphologyEx(mat, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))  # close over the grid lines
    mat = cv2.morphologyEx(mat, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # drop stray specks off the mat
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mat)
    if n < 2:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[biggest, cv2.CC_STAT_AREA] < 0.2 * mat.size:
        return None
    solid = (labels == biggest).astype(np.uint8) * 255
    # a cutting mat is a rectangle; fitting one also covers parts hidden by the
    # lamp's clamp or a label
    outline = np.zeros_like(mat)
    box = cv2.boxPoints(cv2.minAreaRect(cv2.findNonZero(solid)))
    cv2.fillConvexPoly(outline, box.astype(np.int32), 255)
    return solid, outline


def surface_region(background, overhang=0.06):
    """Where books can be: 255 on the cutting mat, 128 in a margin around it (a
    book overhanging the edge), 0 elsewhere. None if there's no mat."""
    shape = _mat_shape(background)
    if shape is None:
        return None
    _, outline = shape
    grow = int(overhang * outline.shape[1]) | 1
    region = cv2.dilate(outline, np.ones((grow, grow), np.uint8)) // 2
    region[outline > 0] = 255
    return region


def objects_on_mat(background, min_area=0.03):
    """Fraction of the view taken by things lying on the mat, so startup can wait
    until it's clear before learning the background. Small intrusions (lamp feet,
    a label) are ignored."""
    shape = _mat_shape(background)
    if shape is None:
        return 0.0
    solid, outline = shape
    holes = cv2.bitwise_and(cv2.erode(outline, np.ones((9, 9), np.uint8)), cv2.bitwise_not(solid))
    n, _, stats, _ = cv2.connectedComponentsWithStats(holes)
    big = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area * holes.size]
    return sum(big) / holes.size


GUTTER_OVERLAP = 0.03  # fraction of a page's width each crop extends past the gutter


def find_pages_by_background(frame, background, region=None, min_area=0.04, margin=0.02, spread_ratio=1.0):
    """Find the book as whatever differs from the empty surface, so pages of any
    colour or content are found. Needs a fixed camera.

    Skin is removed so hands holding the book don't count, a rectangle is fitted
    to what's left, and a wide one (an open spread) is split at the gutter. Returns
    corner arrays in `frame` coordinates, left page first.
    """
    small = background_frame(frame)
    scale = small.shape[1] / frame.shape[1]
    fg = (_background_diff(small, background, None if region is None else region < 255) > 30).astype(np.uint8) * 255
    fg[cv2.dilate(_hands(small), np.ones((15, 15), np.uint8)) > 0] = 0
    # a shifted cutting mat differs from the background but is never the book
    fg[_mat_coloured(small)] = 0
    if region is not None:
        fg[region == 0] = 0  # off the mat, only exposure shifts happen, not books
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))  # cut off wrists, cables
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2:
        return []
    areas = stats[:, cv2.CC_STAT_AREA].copy()
    areas[0] = 0
    if areas.max() < min_area * fg.size:
        return []
    # the book can fall apart into pieces (a spread split at a dark gutter); keep all big ones
    mask = (areas[labels] >= 0.25 * areas.max()).astype(np.uint8)
    # a book lies mostly on the mat; strips only in the margin are lighting changes
    if region is not None and np.count_nonzero(region[mask > 0] == 255) < 0.6 * np.count_nonzero(mask):
        return []
    # fill holes (dark print can match the dark mat) so the book is one solid shape
    outline, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, outline, -1, 1, cv2.FILLED)
    # fit a rotated box, trim sparse fringes (leftover fingers, shadows) by keeping
    # the well-filled rows and columns, then refit on what's left so stray pixels
    # don't skew the angle
    for _ in range(2):
        points = cv2.findNonZero(mask)
        if points is None or len(points) < 20:  # trimming left nothing solid
            return []
        (cx, cy), (w, h), angle = cv2.minAreaRect(points)
        if angle > 45:
            w, h, angle = h, w, angle - 90
        # work in the book's own axes: x across the spread, y down the page
        to_book = cv2.getRotationMatrix2D((cx, cy), angle, 1)
        upright = cv2.warpAffine(mask, to_book, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)
        x0, x1 = _filled_span(upright.mean(axis=0))
        y0, y1 = _filled_span(upright.mean(axis=1))
        if x1 - x0 < 10 or y1 - y0 < 10:
            return []
        trimmed = np.zeros_like(upright)
        trimmed[y0:y1, x0:x1] = upright[y0:y1, x0:x1]
        mask = cv2.warpAffine(trimmed, cv2.invertAffineTransform(to_book), (mask.shape[1], mask.shape[0]),
                              flags=cv2.INTER_NEAREST)
    pad_x, pad_y = margin * (x1 - x0), margin * (y1 - y0)
    x0, x1, y0, y1 = x0 - pad_x, x1 + pad_x, y0 - pad_y, y1 + pad_y
    w, h = x1 - x0, y1 - y0
    upright = cv2.warpAffine(small, to_book, (small.shape[1], small.shape[0]))
    gray = cv2.cvtColor(upright[max(int(y0), 0):int(y1), max(int(x0), 0):int(x1)], cv2.COLOR_BGR2GRAY)
    # boxes in the book's axes, as (left, top, right, bottom)
    if w > spread_ratio * h:
        # side by side: the gutter is where the paper itself is darkest (its shadow),
        # near the middle. A high percentile per column measures the paper between
        # the ink, so dark illustrations or text columns don't pull the split onto them
        cols = np.convolve(np.percentile(gray, 95, axis=0), np.ones(3) / 3, mode="same")
        mid, reach = len(cols) // 2, len(cols) // 8
        window = cols[mid - reach:mid + reach]
        # a flat-lying booklet has no gutter shadow, and the darkest column is then
        # just the edge of the search; take the middle instead
        darkest = int(np.argmin(window))
        if darkest <= 2 or darkest >= len(window) - 3:
            darkest = reach
        cut = max(x0, 0) + mid - reach + darkest
        overlap = GUTTER_OVERLAP * w / 2  # so a split a few pixels off doesn't cut line ends
        boxes = [(x0, y0, cut + overlap, y1), (cut - overlap, y0, x1, y1)]
    else:
        # the fold is a thin line, lost at the detection scale, so look at the book in
        # the full frame
        box = np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], np.float32)
        corners = order_corners(cv2.transform(box, cv2.invertAffineTransform(to_book))[0] / scale)
        fold = _horizontal_fold(frame, corners)
        if fold is None:
            boxes = [(x0, y0, x1, y1)]
        else:
            cut = y0 + fold * h
            overlap = GUTTER_OVERLAP * h / 2
            boxes = [(x0, y0, x1, cut + overlap), (x0, cut - overlap, x1, y1)]
    from_book = cv2.invertAffineTransform(to_book)
    pages = []
    for left, top, right, bottom in boxes:
        box = np.array([[[left, top], [right, top], [right, bottom], [left, bottom]]], np.float32)
        pages.append(order_corners(cv2.transform(box, from_book)[0] / scale))
    return pages


def _horizontal_fold(frame, corners, reach=0.25, min_dip=0.04, height=600):
    """Where the fold is, as a fraction of the book's height, when a book lies opened
    top to bottom; else None.

    Single pages are tall too, so this needs more evidence than a side-by-side
    split: the book must be mostly bare paper (coloured covers make false dips)
    and the paper must darken clearly along one row near the middle, the shadow
    where the pages bend into the binding."""
    tl, tr, br, bl = corners
    width = max(2, int(height * np.linalg.norm(tr - tl) / max(np.linalg.norm(bl - tl), 1)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    book = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(corners.astype(np.float32), dst), (width, height))
    gray = cv2.cvtColor(book, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(book, cv2.COLOR_BGR2HSV)
    if np.count_nonzero((hsv[..., 2] > 140) & (hsv[..., 1] < 60)) < 0.5 * gray.size:
        return None
    rows = np.convolve(np.percentile(gray, 95, axis=1), np.ones(5) / 5, mode="same")
    lo, hi = int(len(rows) * (0.5 - reach)), int(len(rows) * (0.5 + reach))
    window = rows[lo:hi]
    darkest = int(np.argmin(window))
    if darkest <= 2 or darkest >= len(window) - 3:
        return None  # no minimum inside the search, just a gradient
    if np.median(window) - window[darkest] < min_dip * np.median(window):
        return None
    return (lo + darkest) / len(rows)


def _background_diff(small, background, steady=None):
    """Per-pixel colour difference from the background, after matching the
    background's overall colour to the frame. Auto exposure and white balance shift
    the whole image when a book comes into view. The shift is measured as the
    per-channel gain that leaves the most of the view unchanged: a big book can
    cover most of the view, so a plain median over everything would measure the
    book instead. `steady` marks where books rarely are (off the mat); candidates
    are judged there when enough of it is visible, since a big plain book can
    otherwise match the mat under a wrong gain."""
    f, b = small.astype(np.float32), background.astype(np.float32)
    fs, bs = f[::4, ::4].reshape(-1, 3), b[::4, ::4].reshape(-1, 3)
    # clipped highlights can't follow an exposure change, so measure it on mid-tones
    mid = ((fs > 15) & (fs < 235) & (bs > 15) & (bs < 235)).all(axis=1)
    ratio = (fs + 1) / (bs + 1)
    candidates = [np.ones(3)]
    if mid.sum() > 100:
        candidates.append(np.median(ratio[mid], axis=0))
        # medians over the darker and lighter halves of the background, so one of
        # them comes from whatever the book doesn't cover (mat or desk)
        lum = bs.mean(axis=1)
        for part in (mid & (lum < np.median(lum[mid])), mid & (lum >= np.median(lum[mid]))):
            if part.sum() > 100:
                candidates.append(np.median(ratio[part], axis=0))

    judge = np.ones(len(fs), bool)
    if steady is not None:
        steady = steady[::4, ::4].reshape(-1)
        if steady.mean() > 0.05:
            judge = steady

    def unchanged(g):
        return np.count_nonzero((np.abs(fs[judge] - bs[judge] * g) < 20).all(axis=1))

    gains = max(candidates, key=unchanged)
    # refine on the pixels that agree with it
    agree = mid & (np.abs(fs - bs * gains) < 20).all(axis=1)
    if agree.sum() > 100:
        gains = np.median(ratio[agree], axis=0)
    matched = np.clip(b * gains, 0, 255).astype(np.uint8)
    diff = cv2.absdiff(cv2.cvtColor(small, cv2.COLOR_BGR2LAB), cv2.cvtColor(matched, cv2.COLOR_BGR2LAB)).max(axis=2)
    # near-white in both (a desk blown out by exposure) is not a change
    diff[(small.min(axis=2) > 215) & (matched.min(axis=2) > 200)] = 0
    return diff


def _filled_span(profile, frac=0.4, max_gap=25):
    """Longest run of a fill profile above `frac` of its peak, as (start, end)."""
    profile = np.convolve(profile, np.ones(9) / 9, mode="same")
    above = (profile >= frac * profile.max()).astype(np.uint8)
    # bridge narrow dips such as the gutter of a spread
    above = cv2.morphologyEx(above[None], cv2.MORPH_CLOSE, np.ones((1, max_gap), np.uint8))[0].astype(bool)
    best, start = (0, 0), None
    for i, on in enumerate(np.append(above, False)):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def _hands(small):
    """Mask of hands and arms: solid skin-coloured regions reaching the frame edge.
    Red or orange covers have skin-like pixels too, but scattered among others and
    not attached to an arm coming in from outside the frame."""
    # Cr from 145: cream and off-white paper sits at ~140, hands at ~150-160
    skin = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb), (0, 145, 70), (255, 185, 125))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(skin)
    h, w = skin.shape
    x, y, bw, bh = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]
    at_edge = (x == 0) | (y == 0) | (x + bw >= w) | (y + bh >= h)
    at_edge[0] = False
    return at_edge[labels].astype(np.uint8) * 255


def foreground_fraction(frame, background):
    """Share of the view that differs from the background (for drift updates)."""
    diff = _background_diff(background_frame(frame), background)
    return float(np.count_nonzero(diff > 30)) / diff.size


def focus_score(img, width=600):
    """How in focus a page image is, independent of what's printed on it.

    Compares the finest detail at edges (lost to even slight defocus) with
    medium-scale detail (which survives it), so a sparse cover and a dense text
    page score alike. On the reference shots sharp pages scored 0.36-0.55, and
    copies blurred by 1 pixel or more 0.31 or less."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.resize(g, (width, max(1, int(width * g.shape[0] / g.shape[1]))), interpolation=cv2.INTER_AREA)
    b1, b3 = cv2.GaussianBlur(g, (0, 0), 1.0), cv2.GaussianBlur(g, (0, 0), 3.0)
    fine, medium = np.abs(g - b1), np.abs(b1 - b3)
    edges = medium > np.percentile(medium, 95)
    if edges.sum() < 50 or medium[edges].mean() < 1:
        return 1.0  # blank: nothing to judge, so don't reject it
    return float(fine[edges].mean() / medium[edges].mean())


def has_text(img, min_rows=12, min_cover=0.04):
    """Whether an image (a processed page) is a text page worth OCR.

    Counts line-shaped blobs: characters smeared sideways into long, thin bars.
    Covers, photos and illustrations with a caption or title have a few such
    bars; a text page has many, covering a good part of it."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 800 / gray.shape[1]
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    lines = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    rows, area = 0, 0
    for c in contours:
        _, _, cw, ch = cv2.boundingRect(c)
        if cw > 0.15 * w and cw > 6 * ch and 4 <= ch <= 0.06 * h:
            rows += 1
            area += cw * ch
    return rows >= min_rows and area >= min_cover * h * w


class StabilityTracker:
    """Reports a page once its corners have stayed put for `hold` seconds.

    A single odd frame (the webcam sends occasional corrupt JPEGs, and detection
    can drop out for a frame) is ignored; it takes `patience` disagreeing frames
    in a row to reset.
    """

    def __init__(self, hold, tolerance=0.03, patience=3):
        self.hold = hold
        self.tolerance = tolerance  # fraction of frame width
        self.patience = patience
        self.anchor = None
        self.since = None
        self.fired = False
        self.misses = 0  # consecutive frames disagreeing with the anchor

    def update(self, corners, width, now):
        agrees = (corners is not None and self.anchor is not None and corners.shape == self.anchor.shape
                  and np.abs(corners - self.anchor).max() <= self.tolerance * width)
        if agrees:
            self.misses = 0
            if not self.fired and now - self.since >= self.hold:
                self.fired = True  # one capture per placement; re-arms when the page moves or leaves
                return True
            return False
        self.misses += 1
        if self.anchor is None or self.misses >= self.patience:
            self.anchor, self.since, self.fired, self.misses = corners, now, False, 0
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


class FrameStacker:
    """Average a few frames once focus has settled, cancelling sensor noise and
    MJPEG artifacts. Frames that are blurrier or show movement are skipped, so a
    bumped page degrades to the single sharpest frame rather than a smeared one."""

    def __init__(self, frame, corners, sharpness_score, count, now, window=1.5,
                 min_sharpness_ratio=0.85, max_motion=6.0):
        self.corners = corners
        self.ref_score = sharpness_score
        self.count = count
        self.deadline = now + window
        self.min_ratio = min_sharpness_ratio
        self.max_motion = max_motion  # mean abs grey-level change over the page
        self.ref_small = self._small(frame)
        self.frames = [frame]

    def _small(self, frame):
        x, y, w, h = cv2.boundingRect(self.corners.astype(np.int32))
        crop = cv2.cvtColor(frame[max(y, 0):y + h, max(x, 0):x + w], cv2.COLOR_BGR2GRAY)
        return cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA).astype(np.float32)

    def update(self, frame, now):
        """Returns True when done; result() gives the averaged frame."""
        moved = np.abs(self._small(frame) - self.ref_small).mean() > self.max_motion
        if not moved and sharpness(frame, self.corners) >= self.min_ratio * self.ref_score:
            self.frames.append(frame)
        return len(self.frames) >= self.count or now >= self.deadline

    def result(self):
        if len(self.frames) == 1:
            return self.frames[0]
        return np.clip(np.mean([f.astype(np.float32) for f in self.frames], axis=0), 0, 255).astype(np.uint8)


def downscale(frame):
    """Returns (grey, colour, scale) at detection size."""
    scale = DETECT_WIDTH / frame.shape[1]
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), small, scale
