import os
import shutil
import yaml
from pathlib import Path

# Root directory where your runs are stored
root_dir = Path("/home/charlie/Downloads/newdecoder/new_decoder")

# Output base directory
output_base = Path("/home/charlie/Downloads/newdecoder/new_decoder/out")
output_base.mkdir(exist_ok=True)

# Iterate over each subdirectory matching run-*/
for run_dir in root_dir.glob("run-*/files"):
    config_path = run_dir / "hydra_config.yaml"
    model_path = run_dir / "trained_5000.pt"

    if not config_path.exists() or not model_path.exists():
        print(f"Missing files in {run_dir}, skipping.")
        continue

    # Load YAML config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    try:
        model_type = config["model"]["type"]
        weighted_loss = config["loss"]["weighted_loss"]  # spelling as in your example
        latent_size = config["model"]["latent_size"]
    except KeyError as e:
        print(f"Missing key {e} in {config_path}, skipping.")
        continue

    # Build destination path
    dest_dir = output_base / f"{model_type}/W{weighted_loss}_L{latent_size}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    shutil.copy2(config_path, dest_dir / "config.yaml")
    shutil.copy2(model_path, dest_dir / "trained_5000.pt")

    print(f"Copied files from {run_dir} to {dest_dir}")
