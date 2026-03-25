# PlateCloak — Adversarial ALPR Research Toolkit

A research toolkit for studying adversarial robustness of Automatic License Plate Recognition (ALPR) systems. This project implements a three-stage ALPR pipeline (detection → preprocessing → OCR), a configurable perturbation framework, and batch evaluation tools to systematically measure how controlled image degradation affects recognition performance.

> **Warning:** This toolkit is for **academic research only**. Do NOT use these tools to evade law enforcement, commit crimes, harass individuals, or otherwise break the law. Only run experiments on images you own or have explicit permission to use. Do not apply or test adversarial patterns on real vehicles or in public without authorization.

---

## Key Findings

**Perturbation effectiveness is plate-specific and non-transferable.** Adversarial patterns optimized for one license plate do not generalize to other plates. Each plate's unique character arrangement, font weight, spacing, and contrast profile creates a distinct feature landscape, meaning perturbation parameters that successfully degrade recognition for one plate may have little or no effect on another. This implies that ALPR systems are not vulnerable to a single universal adversarial pattern, and any adversarial strategy must be individually calibrated per plate instance.

**Small, numerous geometric shapes are more effective than large ones.** Configurations using many small shapes (40–50 shapes at 1–5 pixel size) consistently outperformed fewer, larger shapes at disrupting detection. Small shapes distributed across the plate region interfere with feature extraction at the convolutional level without creating obvious visual artifacts.

**Noise intensity exhibits threshold behavior.** ALPR detection does not degrade linearly with increasing Gaussian noise. Instead, detection remains stable up to a critical noise intensity, after which performance drops sharply. This suggests the detection model applies internal confidence thresholds rather than degrading gracefully under noise.

**Combined perturbations outperform isolated techniques.** Applying shapes, noise, warp, and texture simultaneously produces higher evasion rates than any single perturbation type alone. The interaction between perturbation types creates compound degradation that the detection and recognition stages cannot individually compensate for.

**Visual stealth and evasion effectiveness are competing objectives.** Perturbation configurations that maximize evasion tend to introduce visible artifacts. Balancing evasion rate against visual conspicuousness requires careful parameter tuning, as aggressive settings achieve high Class C rates but at the cost of human-noticeable distortion.

---

## Repository Structure

| File / Folder | Purpose |
|---|---|
| `PlateShapeCreator/` | Python package (`plateshapez`) for generating adversarially perturbed overlay datasets |
| `ALPRGbatch.py` | Batch evaluation: runs ALPR on a folder and classifies results into Class A/B/C |
| `ocr.py` | Full ALPR pipeline: YOLO detection → multi-pass OCR with preprocessing variants |
| `File_Organizer.py` | Converts the UFPR-ALPR dataset into YOLO-compatible folder layout |
| `neat_integration/` | Analysis and parameter sweep tools |
| `YOLO_model.pt` | Trained YOLO model for license plate detection |
| `data.yaml` | Dataset manifest for YOLO training |
| `requirements.txt` | Python dependencies |

---

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/platecloaker.git
cd platecloaker

# Install core dependencies
pip install -r requirements.txt

# Install the plateshapez dataset generator
cd PlateShapeCreator
uv sync
uv pip install -e .
cd ..
```

---

## Workflow

### 1. Prepare Input Images

Place vehicle background images (JPG) in a `backgrounds/` folder and cropped license plate overlays (transparent PNG) in an `overlays/` folder.

```
project/
├── backgrounds/     # Vehicle images
├── overlays/        # License plate images (PNG with alpha)
└── ...
```

### 2. Resize Overlays

Standardize overlay dimensions to match the scale of your background images:

```bash
python target.py
```

Adjust `TARGET_WIDTH` and `TARGET_HEIGHT` inside `target.py` as needed.

### 3. Generate Adversarial Dataset

```bash
cd PlateShapeCreator
uv run advplate generate
```

Or use the Python API with specific perturbation parameters:

```python
from plateshapez import DatasetGenerator

gen = DatasetGenerator(
    bg_dir="backgrounds",
    overlay_dir="overlays",
    out_dir="dataset",
    perturbations=[
        {"name": "shapes", "params": {"num_shapes": 47, "min_size": 1, "max_size": 5}},
        {"name": "noise", "params": {"intensity": 25}},
        {"name": "warp", "params": {"intensity": 1.5, "frequency": 5.0}},
        {"name": "texture", "params": {"type": "grain", "intensity": 0.27}},
    ],
    random_seed=1337,
)
gen.run(n_variants=500)
```

### 4. Evaluate Adversarial Effectiveness

Run the batch evaluator on your generated dataset:

```bash
python ALPRGbatch.py
```

Select the folder containing generated images. The script classifies each image:

- **Class A** — Plate detected and read correctly
- **Class B** — Plate detected but misread
- **Class C** — Plate not detected at all

Results are saved to `alpr_results.csv` with confidence scores, bounding boxes, and per-image classifications. Images are sorted into `Class A/`, `Class B/`, and `Class C/` subdirectories for inspection.

---

## Perturbation Types

| Perturbation | Description | Key Parameters |
|---|---|---|
| **Shapes** | Random geometric shapes (rectangles, ellipses, triangles) simulating occlusion | `num_shapes` (5–50), `min_size`, `max_size` (1–25 px) |
| **Noise** | Additive Gaussian noise simulating sensor noise or compression artifacts | `intensity` (σ = 5–50) |
| **Warp** | Sinusoidal displacement simulating perspective distortion or motion | `intensity` (0.5–20.0), `frequency` (5.0–50.0) |
| **Texture** | Surface overlays (grain, scratches, dirt) simulating environmental wear | `type`, `intensity` (0.0–1.0) |

All perturbations support a `scope` parameter: `region` (default, plate area only) or `global` (entire image).

---

## Dataset Preparation (UFPR-ALPR)

To convert the UFPR-ALPR dataset into YOLO format:

```bash
python File_Organizer.py
```

This creates a YOLO-compatible directory with `images/{train,val,test}` and `labels/{train,val,test}` subdirectories, copies images from the UFPR track structure, and rewrites annotation files into YOLO bounding box format.

**Requires:** the `UFPR-ALPR dataset/` directory at the repo root, and `Pillow` installed.

---

## Example Result
!lil_fork[lilfork.png]
## Classification Definitions

| Class | Detection | Recognition | Interpretation |
|---|---|---|---|
| **Class A** | ✓ Detected | ✓ Correct | Perturbation had no effect |
| **Class B** | ✓ Detected | ✗ Wrong | OCR disrupted but plate still localized |
| **Class C** | ✗ Not detected | — | Detection stage failed entirely |

Class C is the strongest adversarial outcome, as it prevents any downstream recognition. Class B indicates partial effectiveness where the perturbation disrupted character features but not plate-level features used by the detector.

---

## Ethics & Legal

- Only use images you own or have explicit, documented permission to use
- Do not test adversarial patterns on real vehicles or in public spaces without authorization
- Keep datasets and adversarial artifacts private; avoid publishing raw patterns that could enable misuse
- Seek institutional or legal review (IRB) where appropriate
- Follow responsible disclosure practices
- **Adversarial patterns are plate-specific and non-transferable** — findings from this research do not constitute a general-purpose evasion method

This repository is maintained for **academic research only**.

---

## References

- Laroca, R. et al. (2018). UFPR-ALPR dataset.
- Jocher, G. (2023). Ultralytics YOLO.
- Stanley, K. O. & Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies.
- JaidedAI (2026). EasyOCR.
