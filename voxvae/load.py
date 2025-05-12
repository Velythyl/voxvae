import os
import re
from pathlib import Path
from typing import Union

import torch

from voxvae.utils.wandb_hydra import load_wandbconfig_as_hydraconfig


def get_checkpoint_path(path, checkpoint_idx):
    if checkpoint_idx == -1:
        # Match filenames like trained_0.pt, trained_199.pt
        pattern = re.compile(r"trained_(\d+)\.pt")
        checkpoints = []

        for fname in os.listdir(path):
            match = pattern.fullmatch(fname)
            if match:
                idx = int(match.group(1))
                checkpoints.append((idx, fname))

        if not checkpoints:
            raise FileNotFoundError("No checkpoint files found in directory.")

        # Get the filename with the highest index
        latest_checkpoint = max(checkpoints, key=lambda x: x[0])[1]
        checkpoint_path = os.path.join(path, latest_checkpoint)
    else:
        checkpoint_path = os.path.join(path, f"trained_{checkpoint_idx}.pt")

    return checkpoint_path


def load_vma(path: Union[str, Path], checkpoint=-1, cfg=None):
    if isinstance(path, str):
        path = Path(path)
    if not str(path).endswith('files'):
        path = path / 'files'
    assert os.path.exists(path)

    if cfg is None:
        cfg = load_wandbconfig_as_hydraconfig(path)

    from voxvae.training.models.autoencoder import build_model
    model = build_model(cfg.model)

    checkpoint_path = get_checkpoint_path(path, checkpoint)
    model.load_state_dict(torch.load(checkpoint_path))
    return model

if __name__ == "__main__":
    load_vma("./wandb/run-20250506_132651-caqvr0m0")