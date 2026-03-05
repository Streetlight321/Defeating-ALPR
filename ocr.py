import cv2
from ultralytics import YOLO
from math import floor
import numpy as np
import easyocr

# Initialize EasyOCR reader once (expensive to instantiate)
reader = easyocr.Reader(['en'], gpu=False)

def preprocess_plate(img):
    """Preprocess cropped plate image for OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        31, 10
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    return bw

def crop_plate(img, box):
    """Crop license plate region from image using YOLO bounding box."""
    x1, y1, x2, y2 = [floor(c) for c in box]
    return img[y1:y2, x1:x2]

def ocr_plate(plate_img):
    """
    Run OCR on a preprocessed plate image using EasyOCR.
    Returns (text: str, confidence: float) tuple.
    Confidence is averaged across all detected tokens.
    """
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    processed = preprocess_plate(plate_img)

    results = reader.readtext(
        processed,
        allowlist=allowlist,
        detail=1,          # return bounding box + text + confidence
        paragraph=False,   # treat each detection independently
    )

    if not results:
        return "", 0.0

    texts = [text.strip() for (_, text, _) in results if text.strip()]
    confs = [conf for (_, text, conf) in results if text.strip()]

    if not texts:
        return "", 0.0

    plate_text = "".join(texts)
    avg_conf = sum(confs) / len(confs)  # already 0–1 in EasyOCR
    return plate_text, round(avg_conf, 4)

def run_alpr(image_path: str, model_path: str) -> list[dict]:
    """
    Full ALPR pipeline: detect plates in an image, OCR each one.
    Args:
        image_path: Path to input image.
        model_path: Path to YOLO .pt model file.
    Returns:
        List of dicts with keys:
            - 'plate_text'   (str)
            - 'ocr_conf'     (float, 0–1)
            - 'detect_conf'  (float, 0–1)
            - 'bbox'         (list[int], [x1, y1, x2, y2])
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
        })
    return output

# --- Example usage ---
if __name__ == "__main__":
    results = run_alpr(
        image_path="figure1(1).png",
        model_path="YOLO_model.pt"
    )
    for r in results:
        print(f"Plate: {r['plate_text']}  |  OCR conf: {r['ocr_conf']:.2%}  |  Detect conf: {r['detect_conf']:.2%}  |  BBox: {r['bbox']}")