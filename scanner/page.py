"""Page processing: perspective correction, deskew, colour correction, sharpening, OCR."""

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


def deskew(img):
    """Rotate so text lines are horizontal, using the dominant angle of text blobs."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # smear characters horizontally into line-shaped blobs
    lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)))
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
    angle = float(np.average(angles, weights=weights))
    spread = float(np.sqrt(np.average((angles - angle) ** 2, weights=weights)))
    # the perspective warp already levels the page; only correct small residual skew
    if spread > 1.0 or abs(angle) < 0.2 or abs(angle) > 3:
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
    # background = lightness with text removed (closing) and heavily blurred,
    # computed at 1/4 scale since it's smooth anyway
    small = cv2.resize(l, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    k = max(5, (min(small.shape) // 30) | 1)
    small = cv2.GaussianBlur(cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)), (0, 0), k)
    background = cv2.resize(small, (l.shape[1], l.shape[0]), interpolation=cv2.INTER_LINEAR)
    target = np.percentile(background, 90)
    flat = l.astype(np.float32) * (target / np.maximum(background.astype(np.float32), 1))
    l = np.clip((1 - strength) * l + strength * flat, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def trim_border(img, fraction=0.01):
    """Shave a thin margin to drop edge artefacts left by the warp."""
    h, w = img.shape[:2]
    dy, dx = int(h * fraction), int(w * fraction)
    return img[dy:h - dy, dx:w - dx]


def sharpen(img, amount=0.6, sigma=1.2):
    """Light unsharp mask to crisp up text softened by the webcam and frame averaging."""
    return cv2.addWeighted(img, 1 + amount, cv2.GaussianBlur(img, (0, 0), sigma), -amount, 0)


def process(frame, corners):
    return sharpen(color_correct(deskew(trim_border(warp(frame, corners)))))


PROBE_WIDTH = 1200


def ocr_oriented(image_path, lang, low_confidence=40, margin=15):
    """OCR, and if the result is poor, check whether the page is sideways.

    Pages are assumed roughly upright, so only ±90° is tried, on a copy capped at
    PROBE_WIDTH to keep it cheap. If a rotation reads clearly better, the image file is
    rotated in place and OCR is rerun at full size. Returns (text, confidence, degrees).
    """
    text, conf, vertical = ocr(image_path, lang, correct=True)
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
# alternative Tesseract model dir (e.g. tessdata_best); set from config at startup
TESSDATA_DIR = None
# correct.Corrector applied to final OCR text; set from config at startup
CORRECTOR = None
OCR_MAX_WIDTH = 4000  # cap so very large pages don't blow up OCR time


def prepare_for_ocr(img):
    """Grayscale, 2x upscale and unsharp mask: webcam text is only a few pixels tall,
    and Tesseract reads it best at ~2x with crisp edges. (Benchmarked against
    binarization and 3x, both of which did worse.)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = min(OCR_UPSCALE, OCR_MAX_WIDTH / gray.shape[1])
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


def ocr(image_path, lang, correct=False):
    """Run Tesseract; returns (text, mean word confidence 0-100, fraction of words taller than wide).

    With correct=True and a CORRECTOR configured, uncertain words are fixed using context.
    """
    prepared = image_path.with_name(image_path.stem + ".ocr.png")
    cv2.imwrite(str(prepared), prepare_for_ocr(cv2.imread(str(image_path))))
    cmd = ["tesseract", str(prepared), "stdout", "-l", lang, "--psm", "3", "--dpi", "300"]
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
