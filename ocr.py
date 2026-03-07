import cv2
import re
import numpy as np
from math import floor
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Engine bootstrap — PaddleOCR with EasyOCR as automatic fallback
# ---------------------------------------------------------------------------
try:
    from paddleocr import PaddleOCR
    _paddle = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    def _ocr_engine(img_array: np.ndarray) -> list[tuple[str, float]]:
        """Returns list of (text, confidence) via PaddleOCR."""
        result = _paddle.ocr(img_array, cls=True)
        tokens = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0].strip()
                conf = float(line[1][1])
                if text:
                    tokens.append((text, conf))
        return tokens

    ENGINE = "paddle"

except ImportError:
    import easyocr
    _easy = easyocr.Reader(["en"], gpu=False)
    _ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"

    def _ocr_engine(img_array: np.ndarray) -> list[tuple[str, float]]:
        """Returns list of (text, confidence) via EasyOCR."""
        results = _easy.readtext(
            img_array,
            allowlist=_ALLOWLIST,
            detail=1,
            paragraph=False,
        )
        return [
            (text.strip(), float(conf))
            for (_, text, conf) in results
            if text.strip()
        ]

    ENGINE = "easyocr"

print(f"[ALPR] OCR engine: {ENGINE}")


# ---------------------------------------------------------------------------
# Plate regex patterns  (add your region's formats here)
# ---------------------------------------------------------------------------
_PLATE_RE = [
    re.compile(p) for p in [
        r"^[A-Z]{2,3}[0-9]{3,4}[A-Z]{0,2}$",   # ABC1234 / AB1234CD
        r"^[A-Z]{1,2}[0-9]{2,4}[A-Z]{1,3}$",   # AB12CDE
        r"^[0-9]{1,3}[A-Z]{2,3}[0-9]{0,4}$",   # 123AB / 1ABC234
        r"^[A-Z]{3}[0-9]{3}$",                  # ABC123
        r"^[0-9]{3}[A-Z]{3}$",                  # 123ABC
    ]
]


def _validate_plate(text: str) -> bool:
    clean = re.sub(r"[-\s]", "", text.upper())
    return any(p.match(clean) for p in _PLATE_RE)


# ---------------------------------------------------------------------------
# Preprocessing variants
# ---------------------------------------------------------------------------

def _deskew(gray: np.ndarray) -> np.ndarray:
    """Correct small rotations using Hough line voting."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=60)
    if lines is None:
        return gray
    angles = [np.degrees(ln[0][1]) - 90 for ln in lines[:15]]
    angle = float(np.median(angles))
    if abs(angle) > 15:          # ignore implausible skew
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _preprocess_standard(img: np.ndarray) -> np.ndarray:
    """
    Standard pipeline: upscale → bilateral → CLAHE → sharpen → adaptive threshold.
    Good for well-lit, high-contrast plates.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = _deskew(gray)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    bw = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        19, 8,          # smaller block size — better for compact chars
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)


def _preprocess_light(img: np.ndarray) -> np.ndarray:
    """
    Lighter pipeline: upscale → CLAHE → Otsu threshold.
    Better for already-clean or very high-resolution crops.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = _deskew(gray)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    gray = clahe.apply(gray)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def _preprocess_inverted(img: np.ndarray) -> np.ndarray:
    """Inversion of standard pipeline — handles dark-background plates."""
    return cv2.bitwise_not(_preprocess_standard(img))


def _preprocess_raw_gray(img: np.ndarray) -> np.ndarray:
    """Minimal preprocessing — pure upscaled grayscale."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)


_PREPROCESSING_PIPELINE = [
    _preprocess_standard,
    _preprocess_light,
    _preprocess_inverted,
    _preprocess_raw_gray,
]


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def _run_ocr_on(img_array: np.ndarray) -> tuple[str, float]:
    """Run OCR engine on a single image array, return (text, avg_conf)."""
    tokens = _ocr_engine(img_array)
    if not tokens:
        return "", 0.0
    text = "".join(t for t, _ in tokens)
    conf = sum(c for _, c in tokens) / len(tokens)
    return text, round(conf, 4)


def ocr_plate(plate_img: np.ndarray) -> tuple[str, float]:
    """
    Multi-pass OCR: try every preprocessing variant, return the highest-
    confidence result. Validated plates (matching a known regex) are
    strongly preferred over unvalidated ones.
    """
    validated: list[tuple[str, float]] = []
    candidates: list[tuple[str, float]] = []

    for fn in _PREPROCESSING_PIPELINE:
        try:
            processed = fn(plate_img)
            text, conf = _run_ocr_on(processed)
        except Exception:
            continue

        if not text:
            continue

        clean = re.sub(r"[-\s]", "", text.upper())
        if _validate_plate(clean):
            validated.append((clean, conf))
        else:
            candidates.append((clean, conf))

    if validated:
        return max(validated, key=lambda x: x[1])
    if candidates:
        return max(candidates, key=lambda x: x[1])
    return "", 0.0


# ---------------------------------------------------------------------------
# Crop helper
# ---------------------------------------------------------------------------

def crop_plate(img: np.ndarray, box: list[float], pad: int = 8) -> np.ndarray:
    """
    Crop license plate region from image using YOLO bounding box.
    Adds `pad` pixels of context on every side (clamped to image bounds).
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [floor(c) for c in box]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return img[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Full ALPR pipeline
# ---------------------------------------------------------------------------

def run_alpr(image_path: str, model_path: str) -> list[dict]:
    """
    Full ALPR pipeline: detect plates with YOLO, OCR each crop.

    Args:
        image_path: Path to input image.
        model_path: Path to YOLO .pt model file.

    Returns:
        List of dicts with keys:
            - 'plate_text'   (str)   — recognised plate string
            - 'ocr_conf'     (float) — OCR confidence 0–1
            - 'detect_conf'  (float) — YOLO detection confidence 0–1
            - 'bbox'         (list[int]) — [x1, y1, x2, y2]
            - 'validated'    (bool)  — True if text matches a known plate pattern
    """
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    results = model.predict(image_path, save=False)
    boxes = results[0].boxes
    coords = boxes.xyxy.cpu().numpy().tolist()
    detect_confs = boxes.conf.cpu().numpy().tolist()

    output = []
    for box, det_conf in zip(coords, detect_confs):
        plate_crop = crop_plate(img, box)
        plate_text, ocr_conf = ocr_plate(plate_crop)
        output.append({
            "plate_text":  plate_text,
            "ocr_conf":    ocr_conf,
            "detect_conf": round(float(det_conf), 4),
            "bbox":        [floor(c) for c in box],
            "validated":   _validate_plate(plate_text),
        })

    return output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    image = sys.argv[1] if len(sys.argv) > 1 else r"test_images\car3.jpg"
    model = sys.argv[2] if len(sys.argv) > 2 else "YOLO_model.pt"

    detections = run_alpr(image_path=image, model_path=model)

    if not detections:
        print("No plates detected.")
    else:
        for i, r in enumerate(detections, 1):
            validated_tag = "✓" if r["validated"] else "?"
            print(
                f"[{i}] {validated_tag} Plate: {r['plate_text']:<12} | "
                f"OCR: {r['ocr_conf']:.1%}  "
                f"Det: {r['detect_conf']:.1%}  "
                f"BBox: {r['bbox']}"
            )