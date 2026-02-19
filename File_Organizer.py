import os
import shutil
from pathlib import Path

def find_corners(file_path):
    with open(file_path, "r") as f:
        t = f.read()
        corners = []
        for line in t.splitlines():
            if line.startswith("corners:"):
                points = line.replace("corners:", "").strip().split()
                for p in points:
                    x, y = map(int, p.split(","))
                    corners.append((x, y))
    return corners

def corners_to_yolo_bbox(corners, img_w, img_h):
    xs = [x for x, y in corners]
    ys = [y for x, y in corners]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h

    return f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n"


#This creates a folder "yolo_fuck" which is in the yolo format.

# Root directory name
root_dir = "yolo_fuck"

# Subdirectory structure
subdirs = [
    "images/train",
    "images/val",
    "images/test",
    "labels/train",
    "labels/val",
    "labels/test",
]

# Create directories
for subdir in subdirs:
    path = os.path.join(root_dir, subdir)
    os.makedirs(path, exist_ok=True)

print("YOLO folder structure created successfully.")

ufpr_root = Path("UFPR-ALPR dataset/training")
yolo_root = Path("yolo_fuck")

images_out = yolo_root / "images" / "train"
labels_out = yolo_root / "labels" / "train"

images_out.mkdir(parents=True, exist_ok=True)
labels_out.mkdir(parents=True, exist_ok=True)

for i in range(1, 61):  # track0001 to track0060
    track_folder = f"track{i:04d}"
    track_path = ufpr_root / track_folder

    for j in range(1, 31):  # [01] to [30]
        file_stem = f"{track_folder}[{j:02d}]"

        png_path = track_path / f"{file_stem}.png"
        txt_path = track_path / f"{file_stem}.txt"

        # Move or copy PNG
        if png_path.exists():
            shutil.copy2(png_path, images_out / f"{file_stem}.png")

        # Move or copy TXT
        if txt_path.exists():
            shutil.copy2(txt_path, labels_out / f"{file_stem}.txt")

print("Done with Training.")


import os
import shutil
from pathlib import Path

ufpr_root = Path("UFPR-ALPR dataset/validation")
yolo_root = Path("yolo_fuck")

images_out = yolo_root / "images" / "val"
labels_out = yolo_root / "labels" / "val"

images_out.mkdir(parents=True, exist_ok=True)
labels_out.mkdir(parents=True, exist_ok=True)

for i in range(61, 91):  # track0061 to track0090
    track_folder = f"track{i:04d}"
    track_path = ufpr_root / track_folder

    for j in range(1, 31):  # [01] to [30]
        file_stem = f"{track_folder}[{j:02d}]"

        png_path = track_path / f"{file_stem}.png"
        txt_path = track_path / f"{file_stem}.txt"

        # Move or copy PNG
        if png_path.exists():
            shutil.copy2(png_path, images_out / f"{file_stem}.png")

        # Move or copy TXT
        if txt_path.exists():
            shutil.copy2(txt_path, labels_out / f"{file_stem}.txt")

print("Done with Validation Data.")

import os
import shutil
from pathlib import Path

ufpr_root = Path("UFPR-ALPR dataset/testing")
yolo_root = Path("yolo_fuck")

images_out = yolo_root / "images" / "test"
labels_out = yolo_root / "labels" / "test"

images_out.mkdir(parents=True, exist_ok=True)
labels_out.mkdir(parents=True, exist_ok=True)

for i in range(91, 151):  # track0091 to track0150
    track_folder = f"track{i:04d}"
    track_path = ufpr_root / track_folder

    for j in range(1, 31):  # [01] to [30]
        file_stem = f"{track_folder}[{j:02d}]"

        png_path = track_path / f"{file_stem}.png"
        txt_path = track_path / f"{file_stem}.txt"

        # Move or copy PNG
        if png_path.exists():
            shutil.copy2(png_path, images_out / f"{file_stem}.png")

        # Move or copy TXT
        if txt_path.exists():
            shutil.copy2(txt_path, labels_out / f"{file_stem}.txt")

print("Done with Testing Data.")

from pathlib import Path
from PIL import Image

ROOT = Path("yolo_fuck")
splits = ["train", "val", "test"]


for split in splits:
    labels_dir = ROOT / "labels" / split
    images_dir = ROOT / "images" / split

    if not labels_dir.exists():
        continue

    for txt_path in labels_dir.glob("*.txt"):

        # --- deterministic name match ---
        img_path = images_dir / f"{txt_path.stem}.png"

        if not img_path.exists():
            print(f"[ERR] Missing image: {img_path}")
            continue

        corners = find_corners(txt_path)
        if len(corners) < 4:
            print(f"[ERR] No valid corners in: {txt_path.name}")
            continue

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        yolo_line = corners_to_yolo_bbox(corners, img_w, img_h)

        # overwrite label file
        txt_path.write_text(yolo_line)

print("Finished rewriting labels.")
