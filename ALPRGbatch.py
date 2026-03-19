import cv2
from fast_alpr import ALPR
import tkinter as tk
from tkinter import filedialog
import os
import csv
import shutil
import re

# --- Directory Selection Dialog ---
root = tk.Tk()
root.withdraw()

directory_path = filedialog.askdirectory(
    title="Select a Folder of Images for ALPR"
)

# --- Helper: Extract ground truth plate from filename ---
def extract_ground_truth(filename: str) -> str | None:
    """
    Expects filename format: WhiteBackground_PLATEVALUE_1-10.ext
    Returns the plate value in uppercase, or None if the format doesn't match.
    """
    stem = os.path.splitext(filename)[0]  # Strip extension
    parts = stem.split("_")
    # Expecting at least 3 parts: WhiteBackground, PLATEVALUE, variant-number
    if len(parts) < 3:
        return None
    # The plate value is everything between the first and last underscore segment
    # This handles plate values that might contain underscores
    plate_value = "_".join(parts[1:-1])
    return plate_value.upper().strip()


# --- Helper: Normalize plate text for comparison ---
def normalize_plate(text: str) -> str:
    """Strip spaces, dashes, and uppercase for fair comparison."""
    return re.sub(r"[\s\-]", "", text.upper().strip())


# --- ALPR Processing ---
if directory_path:
    print(f"Directory selected: {directory_path}")
    print("Initializing ALPR models... This might take a moment.")

    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-xs-v1-global-model",
    )

    # --- Setup output directories ---
    csv_path = os.path.join(directory_path, "alpr_results.csv")
    annotated_dir = os.path.join(directory_path, "annotated_output")
    class_a_dir = os.path.join(directory_path, "Class A")  # Correct read
    class_b_dir = os.path.join(directory_path, "Class B")  # Wrong read
    class_c_dir = os.path.join(directory_path, "Class C")  # No detection

    for folder in [annotated_dir, class_a_dir, class_b_dir, class_c_dir]:
        os.makedirs(folder, exist_ok=True)

    # --- Find images ---
    image_files = [
        f for f in os.listdir(directory_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if not image_files:
        print("No image files (.png, .jpg, .jpeg) found in the selected directory.")
    else:
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                'filename',
                'ground_truth',
                'plate_text',
                'confidence',
                'classification',
                'box_x1', 'box_y1', 'box_x2', 'box_y2'
            ])

            print(f"Processing {len(image_files)} images. Results will be saved to {csv_path}")

            # Counters for summary
            counts = {"Class A": 0, "Class B": 0, "Class C": 0, "skipped": 0}

            for i, image_name in enumerate(image_files):
                image_path = os.path.join(directory_path, image_name)
                print(f"  ({i+1}/{len(image_files)}) Processing: {image_name}")

                # --- Extract ground truth from filename ---
                ground_truth = extract_ground_truth(image_name)
                if ground_truth is None:
                    print(f"    - Skipping: filename doesn't match expected format "
                          f"(WhiteBackground_PLATE_1.ext)")
                    counts["skipped"] += 1
                    continue

                print(f"    - Ground truth plate: {ground_truth}")

                # --- Load image ---
                frame = cv2.imread(image_path)
                if frame is None:
                    print(f"    - Warning: Could not load image, skipping.")
                    counts["skipped"] += 1
                    continue

                # --- Run ALPR ---
                predictions = alpr.predict(frame)

                # --- Classify ---
                if not predictions:
                    # Class C: no plate detected at all
                    classification = "Class C"
                    print(f"    - No plate detected → Class C")
                    csv_writer.writerow([
                        image_name, ground_truth, "", "", classification,
                        "", "", "", ""
                    ])
                    shutil.copy2(image_path, os.path.join(class_c_dir, image_name))

                else:
                    # Use the highest-confidence prediction
                    best = max(predictions, key=lambda p: p.ocr.confidence)
                    detection = best.detection
                    ocr = best.ocr
                    bb = detection.bounding_box

                    read_plate = normalize_plate(ocr.text)
                    truth_normalized = normalize_plate(ground_truth)

                    if read_plate == truth_normalized:
                        classification = "Class A"
                        print(f"    - Read '{ocr.text}' == '{ground_truth}' → Class A ✓")
                        shutil.copy2(image_path, os.path.join(class_a_dir, image_name))
                    else:
                        classification = "Class B"
                        print(f"    - Read '{ocr.text}' != '{ground_truth}' → Class B ✗")
                        shutil.copy2(image_path, os.path.join(class_b_dir, image_name))

                    csv_writer.writerow([
                        image_name,
                        ground_truth,
                        ocr.text,
                        f"{ocr.confidence:.4f}",
                        classification,
                        bb.x1, bb.y1, bb.x2, bb.y2
                    ])

                    # Also write any secondary detections to CSV (unclassified)
                    for pred in predictions[1:]:
                        det = pred.detection
                        o = pred.ocr
                        b = det.bounding_box
                        csv_writer.writerow([
                            image_name, ground_truth, o.text,
                            f"{o.confidence:.4f}", "(secondary)",
                            b.x1, b.y1, b.x2, b.y2
                        ])

                counts[classification] += 1

                # --- Save annotated image ---
                annotated_frame = alpr.draw_predictions(frame)
                cv2.imwrite(os.path.join(annotated_dir, image_name), annotated_frame)

        # --- Summary ---
        print("\n" + "=" * 50)
        print("Processing complete!")
        print(f"  Class A (correct read):  {counts['Class A']}")
        print(f"  Class B (wrong read):    {counts['Class B']}")
        print(f"  Class C (no detection):  {counts['Class C']}")
        print(f"  Skipped (bad filename):  {counts['skipped']}")
        print(f"\nCSV saved to:             {csv_path}")
        print(f"Annotated images saved to: {annotated_dir}")
        print(f"Class A images saved to:   {class_a_dir}")
        print(f"Class B images saved to:   {class_b_dir}")
        print(f"Class C images saved to:   {class_c_dir}")

else:
    print("No directory was selected. Exiting program.")