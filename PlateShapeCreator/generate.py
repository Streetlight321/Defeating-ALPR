from plateshapez import DatasetGenerator

gen = DatasetGenerator(
    bg_dir="backgrounds",
    overlay_dir="overlays",
    out_dir="dataset",
    perturbations=[{"name": "noise", "params": {"intensity": 25}}],
    random_seed=42
)
gen.run(n_variants=10)