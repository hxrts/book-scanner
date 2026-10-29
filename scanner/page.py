"""Page processing: perspective correction, deskew, colour correction, sharpening, OCR."""

import re
import subprocess

import cv2
import numpy as np


def warp(frame, corners):
    """Map the page quadrilateral onto a flat rectangle."""
    tl, tr, br, bl = corners
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(frame, matrix, (width, height), flags=cv2.INTER_CUBIC)


MAX_DESKEW = 5  # degrees


def deskew(img):
    """Rotate so text lines are horizontal, using the dominant angle of text blobs."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # a local threshold keeps small text on tinted or unevenly lit paper
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    # smear characters horizontally into line-shaped blobs
    smear = max(15, img.shape[1] // 40)
    lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (smear, 3)))
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_len = img.shape[1] * 0.15
    angles, weights = [], []
    for c in contours:
        (_, _), (w, h), angle = cv2.minAreaRect(c)
        if w < h:
            w, h, angle = h, w, angle - 90
        if w > min_len and w > 8 * h:
            angles.append(angle)
            weights.append(w)
    # need several agreeing lines, otherwise it's likely an illustration
    if len(angles) < 6:
        return img
    angles, weights = np.array(angles), np.array(weights)
    # weighted median, then the mean of the lines that agree with it: edges of
    # pictures and hands give stray angles that would skew a plain average
    order = np.argsort(angles)
    median = angles[order][np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)]
    agree = np.abs(angles - median) < 1.0
    if weights[agree].sum() < 0.5 * weights.sum() or agree.sum() < 6:
        return img  # no clear text direction
    angle = float(np.average(angles[agree], weights=weights[agree]))
    # the perspective warp levels the page itself; only correct residual skew of the
    # print (a page turned in the book's binding or a tilted crop)
    if abs(angle) < 0.2 or abs(angle) > MAX_DESKEW:
        return img

    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def color_correct(img, strength=0.8):
    """White-balance on the paper colour, then flatten uneven lighting.

    The paper (brightest ~half of pixels) is pushed towards neutral off-white,
    and lightness is divided by a smooth background estimate so lamp falloff
    disappears without boosting faint show-through.
    """
    f = img.astype(np.float32)
    lum = f.mean(axis=2)
    paper = f[lum >= np.percentile(lum, 50)]
    gains = paper.mean(axis=0).mean() / np.maximum(paper.mean(axis=0), 1)
    f *= 1 + strength * (gains - 1)

    lab = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    background = paper_lightness(l)
    target = np.percentile(background, 90)
    # only even out lamp falloff; never brighten by more than a third, so large
    # dark areas misread as shadow can't wash out
    gain = np.clip(target / np.maximum(background, 1), 0.85, 1.35)
    # and lift the paper towards white (webcam paper reads as light grey)
    gain *= min(PAPER_WHITE / max(target, 1), 1.25)
    flat = l.astype(np.float32) * gain
    l = np.clip((1 - strength) * l + strength * flat, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


PAPER_WHITE = 242  # LAB lightness (0-255) the paper is lifted to


def paper_lightness(l, cells=12):
    """Smooth estimate of the bare paper's lightness across the page.

    Each cell of a coarse grid takes its brightest pixels (the paper between the
    ink); cells mostly covered by ink or pictures are replaced by their neighbours'
    paper. Unlike blurring the image, a large black illustration can't pull the
    estimate down and leave a halo around itself."""
    h, w = l.shape
    gh, gw = cells, max(2, round(cells * w / h))
    grid = np.zeros((gh, gw), np.float32)
    for i in range(gh):
        for j in range(gw):
            cell = l[i * h // gh:(i + 1) * h // gh, j * w // gw:(j + 1) * w // gw]
            grid[i, j] = np.percentile(cell, 95)
    # a cell darker than its neighbourhood's paper is covered by ink, not shadow
    neighbourhood = cv2.dilate(grid, np.ones((3, 3), np.uint8))
    covered = grid < 0.85 * neighbourhood
    grid[covered] = cv2.medianBlur(neighbourhood, 3)[covered]
    grid = cv2.GaussianBlur(grid, (3, 3), 0)
    return cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)


def trim_border(img, fraction=0.01):
    """Shave a thin margin to drop edge artefacts left by the warp."""
    h, w = img.shape[:2]
    dy, dx = int(h * fraction), int(w * fraction)
    return img[dy:h - dy, dx:w - dx]


def sharpen(img, amount=0.6, sigma=1.2):
    """Light unsharp mask to crisp up text softened by the webcam and frame averaging."""
    return cv2.addWeighted(img, 1 + amount, cv2.GaussianBlur(img, (0, 0), sigma), -amount, 0)


def paper_fraction(img):
    """Share of the page that is plain paper (light and nearly colourless)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(np.count_nonzero((hsv[..., 2] > 140) & (hsv[..., 1] < 60))) / (img.shape[0] * img.shape[1])


# 3x4 colour correction matrix (BGR, 0-1) fitted by colorfit.py; set from config at startup
COLOR_MATRIX = None


def apply_color_matrix(img, matrix):
    x = img.reshape(-1, 3).astype(np.float32) / 255
    out = x @ matrix[:, :3].T + matrix[:, 3]
    return np.clip(out * 255, 0, 255).astype(np.uint8).reshape(img.shape)


def process(frame, corners):
    img = deskew(trim_border(warp(frame, corners)))
    # paper white-balance and lighting flattening would wash out photos and colour pages
    if paper_fraction(img) >= 0.6:
        img = color_correct(img)
    img = sharpen(img)
    # the matrix is fitted on output of the steps above, so it goes last
    return img if COLOR_MATRIX is None else apply_color_matrix(img, COLOR_MATRIX)


PROBE_WIDTH = 1200


def ocr_oriented(image_path, lang, low_confidence=40, margin=15):
    """OCR, and if the result is poor, check whether the page is sideways.

    Pages are assumed roughly upright, so only ±90° is tried, on a copy capped at
    PROBE_WIDTH to keep it cheap. If a rotation reads clearly better, the image file is
    rotated in place and OCR is rerun at full size. Returns (text, confidence, degrees).
    """
    text, conf, vertical = ocr(image_path, lang, correct=True)
    if conf < SMALL_TEXT_RETRY_CONF:
        # tiny or faint print: Tesseract's layout analysis can drop whole blocks at the
        # usual settings; read it again enlarged further, and keep the more readable result
        alt = ocr(image_path, lang, correct=True, small_text=True)
        if _real_words(alt[0]) > _real_words(text):
            text, conf, vertical = alt
    # Tesseract can sometimes read sideways text well, so also check word shapes
    if conf >= low_confidence and len(text) >= 20 and vertical < 0.5:
        return text, conf, 0

    img = cv2.imread(str(image_path))
    probe = image_path.with_name("orientation-probe.jpg")
    # when the text is clearly vertical, the better-reading rotation wins outright;
    # otherwise a rotation must beat the original by `margin`
    sideways = vertical >= 0.5
    best_rot, best_conf = None, (low_confidence if sideways else conf + margin)
    try:
        for rot, degrees in ((cv2.ROTATE_90_CLOCKWISE, 90), (cv2.ROTATE_90_COUNTERCLOCKWISE, 270)):
            small = cv2.rotate(img, rot)
            if small.shape[1] > PROBE_WIDTH:
                scale = PROBE_WIDTH / small.shape[1]
                small = cv2.resize(small, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(probe), small, [cv2.IMWRITE_JPEG_QUALITY, 90])
            rot_text, rot_conf, rot_vertical = ocr(probe, lang)
            if len(rot_text) >= 20 and rot_vertical < 0.5 and rot_conf > best_conf:
                best_rot, best_conf = (rot, degrees), rot_conf
    finally:
        probe.unlink(missing_ok=True)

    if best_rot is None:
        return text, conf, 0
    cv2.imwrite(str(image_path), cv2.rotate(img, best_rot[0]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    text, conf, _ = ocr(image_path, lang, correct=True)
    return text, conf, best_rot[1]


OCR_UPSCALE = 2
SMALL_TEXT_RETRY_CONF = 70  # below this mean confidence, also try the small-text settings
_DICTIONARY = None


def _real_words(text):
    """Number of dictionary words in text, as a measure of how readable an OCR result is."""
    global _DICTIONARY
    if _DICTIONARY is None:
        if CORRECTOR is not None:
            _DICTIONARY = CORRECTOR.dictionary
        else:
            try:
                _DICTIONARY = {w.strip().lower() for w in open("/usr/share/dict/words")}
            except OSError:
                _DICTIONARY = set()
    return sum(1 for w in re.findall(r"[A-Za-z]{2,}", text) if w.lower() in _DICTIONARY)
# alternative Tesseract model dir (e.g. tessdata_best); set from config at startup
TESSDATA_DIR = None
# correct.Corrector applied to final OCR text; set from config at startup
CORRECTOR = None
OCR_MAX_WIDTH = 4000  # cap so very large pages don't blow up OCR time


def prepare_for_ocr(img, small_text=False):
    """Grayscale, 2x upscale and unsharp mask: webcam text is only a few pixels tall,
    and Tesseract reads it best at ~2x with crisp edges. (Benchmarked against
    binarization and 3x, both of which did worse on normal pages.)

    small_text: for tiny or faint print, stretch the contrast so grey ink reads as
    black and upscale 3x (on the reference shots this read 73 real words from a
    page where the normal settings got 2)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if small_text:
        lo, hi = np.percentile(gray, 1), np.percentile(gray, 90)
        gray = np.clip((gray.astype(np.float32) - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    scale = min(3 if small_text else OCR_UPSCALE, OCR_MAX_WIDTH / gray.shape[1])
    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.addWeighted(gray, 2.0, cv2.GaussianBlur(gray, (0, 0), 1.5), -1.0, 0)


def _join_hyphenated(words, confs):
    """Rejoin words split across lines ("stu-" + newline + "dents" -> "students")."""
    out, out_confs, ci = [], [], 0
    i = 0
    while i < len(words):
        w = words[i]
        if w in ("\n", "\n\n"):
            out.append(w)
            i += 1
            continue
        c = confs[ci]
        ci += 1
        nxt = words[i + 2] if i + 2 < len(words) and words[i + 1] == "\n" else None
        if w.endswith("-") and w[:-1].isalpha() and nxt and nxt[:1].islower():
            out.append(w[:-1] + nxt)
            out_confs.append(min(c, confs[ci]))
            ci += 1
            i += 3  # the fragment, the line break, and the continuation
            continue
        out.append(w)
        out_confs.append(c)
        i += 1
    return out, out_confs


def ocr(image_path, lang, correct=False, small_text=False):
    """Run Tesseract; returns (text, mean word confidence 0-100, fraction of words taller than wide).

    With correct=True and a CORRECTOR configured, uncertain words are fixed using context.
    """
    prepared = image_path.with_name(image_path.stem + ".ocr.png")
    cv2.imwrite(str(prepared), prepare_for_ocr(cv2.imread(str(image_path)), small_text))
    # psm 4 (a single column) finds the lines that full layout analysis drops on small print
    cmd = ["tesseract", str(prepared), "stdout", "-l", lang, "--psm", "4" if small_text else "3", "--dpi", "300"]
    if TESSDATA_DIR:
        cmd += ["--tessdata-dir", TESSDATA_DIR]
    try:
        result = subprocess.run(cmd + ["tsv"], capture_output=True, text=True, timeout=600, check=True)
    finally:
        prepared.unlink(missing_ok=True)
    words, confs, last_line, tall = [], [], None, 0  # words holds tokens and line-break markers
    for row in result.stdout.splitlines()[1:]:
        cols = row.split("\t")
        if len(cols) < 12 or not cols[11].strip() or float(cols[10]) < 0:
            continue
        line = tuple(cols[2:5])  # block, paragraph, line
        if last_line is not None and line != last_line:
            words.append("\n" if line[:2] == last_line[:2] else "\n\n")
        words.append(cols[11])
        confs.append(float(cols[10]))
        # multi-letter words are wider than tall when upright
        if len(cols[11]) >= 3 and int(cols[9]) > int(cols[8]):
            tall += 1
        last_line = line
    words, confs = _join_hyphenated(words, confs)
    if correct and CORRECTOR is not None:
        positions = [i for i, w in enumerate(words) if w not in ("\n", "\n\n")]
        fixed = CORRECTOR.correct([(words[i], c) for i, c in zip(positions, confs)])
        for i, w in zip(positions, fixed):
            words[i] = w
    text = " ".join(words).replace(" \n ", "\n").replace(" \n\n ", "\n\n").strip()
    long_words = sum(1 for w in words if len(w) >= 3)
    return text, (sum(confs) / len(confs) if confs else 0.0), (tall / long_words if long_words else 0.0)
