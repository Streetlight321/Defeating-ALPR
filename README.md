# PlateCloak — Adversarial ALPR Research Toolkit

A collection of research tools for studying adversarial methods against Automatic License Plate Recognition (ALPR) systems. This work is **strictly research-focused**.

> **Warning:** Do NOT use these tools to evade law enforcement, commit crimes, harass individuals, or otherwise break the law. Only run experiments on images you own or have explicit permission to use. Do not apply or test adversarial patterns on real vehicles or in public without authorization.

---

## 📁 Repository Structure

| File / Folder | Purpose |
|---|---|
| `File_Organizer.py` | Prepares the UFPR-ALPR dataset into a YOLO-compatible folder layout |
| `ALPRGbatch.py` | Batch-processes a folder of images through an ALPR pipeline to check adversarial effectiveness |
| `ocr.py` | Full ALPR pipeline: YOLO detection → multi-pass OCR with preprocessing variants |
| `target.py` | Resizes raw overlay PNGs to a consistent target size before dataset generation |
| `PlateShapeCreator/` | Python package (`plateshapez`) for generating adversarially perturbed overlay datasets |
| `YOLO_model.pt` | Trained YOLO model for internal research use |
| `yolo11n.pt` | Smaller YOLO weights for training/testing compatibility |
| `OCR.ipynb` | Notebook with EasyOCR preprocessing experiments |
| `data.yaml` | Dataset manifest for YOLO training |
| `requirements.txt` | Python dependencies |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

Install uv if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Workflow: Generate Adversarial Overlays

This is the core loop for creating and testing adversarial license plate overlays.

### Step 1 — Add your license plate images

Place your license plate PNG images in the `overlays_raw/` folder. They must be `.png` files. Transparent-background PNGs work best.

```
overlays_raw/
├── plate_001.png
├── plate_002.png
└── ...
```

### Step 2 — Resize overlays to target dimensions

Run `target.py` to resize all PNGs from `overlays_raw/` into the `overlays/` folder at a consistent resolution:

```bash
python target.py
```

You can adjust `TARGET_WIDTH` and `TARGET_HEIGHT` inside `target.py` to match the scale of your background images.

### Step 3 — Generate adversarial dataset

Place your vehicle background images (JPG) in a `backgrounds/` folder, then run:

```bash
uv run generate.py
```

This uses the `plateshapez` library to composite your overlays onto backgrounds and apply adversarial perturbations (noise, shapes, warp, texture). Output images and metadata are saved to `dataset/`.

To customize perturbations, edit `generate.py` or pass a config file. See `PlateShapeCreator/README.md` for full options.

---

## Check Adversarial Effectiveness

Use `ALPRGbatch.py` to run a folder of (potentially adversarially perturbed) images through the ALPR pipeline and collect results:

```bash
python ALPRGbatch.py
```

A folder picker dialog will open. Select the folder containing the images you want to evaluate. The script will:

- Run ALPR on every `.png`, `.jpg`, and `.jpeg` in the folder
- Save a `alpr_results.csv` with plate text, confidence scores, and bounding boxes
- Save annotated images with detections drawn to an `annotated_output/` subfolder

Use the CSV to compare which adversarial variants caused misreads or missed detections.

---

## Installing Dependencies

### Using uv (recommended)

```bash
# For the core toolkit (ocr.py, ALPRGbatch.py, etc.)
uv pip install -r requirements.txt

# For the plateshapez dataset generator
cd PlateShapeCreator
uv sync
```

### Using pip

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation (UFPR-ALPR)

If you're using the UFPR-ALPR dataset, `File_Organizer.py` converts it into a YOLO-compatible layout:

```bash
python File_Organizer.py
```

This creates a `yolo_check/` folder with `images/{train,val,test}` and `labels/{train,val,test}` subdirectories, copies images and labels from the UFPR dataset's track structure, and rewrites label files into YOLO bounding box format.

**Requires:** the `UFPR-ALPR dataset/` directory to be present at the repo root, and `Pillow` installed.

---

## Research Methodology

The general workflow for studying adversarial robustness:

1. Collect background vehicle images you have permission to use
2. Generate adversarial overlay variants using `generate.py` / `plateshapez`
3. Evaluate each variant with `ALPRGbatch.py`:
   - Plate read correctly → `unaffected`
   - Plate detected but misread → `OCR_Error`
   - Plate not detected at all → `No_Detection`
4. Collect overlays that caused `No_Detection` as training signal
5. Iterate: refine noise generation and re-evaluate

![Methodology](https://github.com/Streetlight321/PlatePeeper/blob/main/figure1(1).png)

---

## Ethics & Legal

- Only use images you own or have explicit, documented permission to use
- Do not test adversarial patterns on real vehicles or in public spaces without authorization
- Keep datasets and adversarial artifacts private; avoid publishing raw patterns that could enable misuse
- Seek institutional or legal review (IRB) where appropriate
- Follow responsible disclosure practices

This repository is maintained for **internal research only**. Contact the project lead with questions. Do not distribute adversarial artifacts.