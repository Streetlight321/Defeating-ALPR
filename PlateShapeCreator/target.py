from PIL import Image
import os

# Target size - adjust to what looks right for your backgrounds
TARGET_WIDTH = 400
TARGET_HEIGHT = 200

input_folder = "overlays_raw"
output_folder = "overlays"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        img = Image.open(f"{input_folder}/{filename}")
        img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
        img.save(f"{output_folder}/{filename}")
        print(f"Resized {filename} to {TARGET_WIDTH}x{TARGET_HEIGHT}")