**Project Summary**

This repository contains research tools and artifacts used to study adversarial methods against automatic license plate recognition (ALPR) systems. This work is strictly research-focused: do NOT use these tools to evade law enforcement, commit crimes, harass individuals, or otherwise break the law.

**Strict Warning**: This repository is for controlled, ethical research only. Do NOT deploy these techniques against real vehicles, real people, or in public spaces. Do NOT share adversarial patterns or tooling with third parties who could misuse them. Misuse may be illegal, dangerous, and harmful — you are responsible for complying with all applicable laws and institutional policies.

**Files & Purpose**

- **`File_Organizer.py`**: Prepares the UFPR dataset into a YOLO-compatible layout. It creates a `yolo_fuck` folder with `images/{train,val,test}` and `labels/{train,val,test}`, copies PNGs and label .txt files from the UFPR-ALPR dataset structure, and rewrites label files into YOLO-format bounding boxes derived from `corners:` entries in the original txt files. Run from the repository root; Pillow is required.
- **`YOLO_model.pt`**: Project's trained YOLO model (provided by me for the group's internal research).
- **`yolo11n.pt`**: A smaller YOLO weight file included for training/testing compatibility.
- **`OCR.ipynb`**: Notebook with experiments and transformations intended to help EasyOCR read license plates (preprocessing, augmentation, and evaluation).
- **`data.yaml`**: (Training configuration) dataset manifest used for YOLO training.

**How `File_Organizer.py` works (quick)**

- Creates `yolo_fuck/images` and `yolo_fuck/labels` with `train`, `val`, `test` subfolders.
- Copies images and original label txts from `UFPR-ALPR dataset/training`, `/validation`, `/testing` track directories into the corresponding YOLO folders.
- For each label file, it reads a line that starts with `corners:` (expects 4+ corner points), computes a tight bounding box, converts it to YOLO format `class x_center y_center width height`, and overwrites the label file.
- Usage: run `python File_Organizer.py` from the repo root. Ensure the `UFPR-ALPR dataset` directory is present and that `Pillow` is installed.

**Methodology (next steps / workflow)**

1. Collect a new batch of license-plate images (controlled dataset or images you have permission to use).
2. For each image, apply an adversarial noise overlay to the license-plate region (generate multiple variants per image using different overlay parameters).
3. Evaluate each noisy image with the detection + OCR pipeline (YOLO detector -> OCR recognizer such as EasyOCR).
   - If OCR reads the plate correctly: move the image to an `unaffected/` folder.
   - If detector finds the plate but OCR misreads it: move to `OCR_Error/`.
   - If the detector does not detect a plate at all: move to `No_Detection/`.
4. Collect overlays that caused `No_Detection`. Use this collection as training data for a model whose goal is to predict or synthesize the minimal overlay/noise patterns that prevent detection.
5. Iterate: refine noise generation using the trained model and re-evaluate until desired robustness or research objectives are met.

**Recommended safety, legal & ethical precautions**

- Only run experiments on images you own or have explicit permission to use.
- Do not apply or test adversarial patterns on real vehicles or in public without authorization.
- Keep datasets and adversarial artifacts private; avoid publishing raw adversarial patterns that could enable misuse.
- Seek institutional review (IRB) or legal review where appropriate and follow responsible disclosure practices.

**Dependencies & Notes**

- Python 3.x, Pillow (used by `File_Organizer.py`).
- The `File_Organizer.py` script assumes the UFPR dataset layout (folders named `track0001` etc.) under `UFPR-ALPR dataset`.
- The repo contains model weights for internal research; handle them responsibly.

**Contact & Attribution**

This repository is maintained for internal research. If you have questions about the methodology or need help reproducing results, contact the project lead (do not publish or distribute adversarial artifacts).

