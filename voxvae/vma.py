import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import functools
import json
import os

import numpy as np
from tqdm import tqdm
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Union
from functools import lru_cache

@dataclass
class VMA:
    vma_path: Path

    latentdir: Path

    vma_checkpoint: int = -1
    inference_device: str = "cuda:0"

    def __hash__(self):
        STRING = f"{self.vma_path} {self.latentdir} {self.vma_checkpoint} {self.inference_device}"
        return hash(STRING)

    @property
    def cfg(self):
        from voxvae.utils.wandb_hydra import load_wandbconfig_as_hydraconfig
        return load_wandbconfig_as_hydraconfig(self.vma_path)

    @property
    def vma(self):

        from voxvae.load import load_vma
        return load_vma(self.vma_path, self.vma_checkpoint, self.cfg).to(self.inference_device)

    @property
    def vma_check_path(self):
        return f"{self.vma_path}{self.vma_checkpoint}"

    @property
    def vma_check_hash(self):
        return hashlib.sha256(str(self.vma_check_path).encode()).hexdigest()[:16]

    def to_latent_path(self, jsonpath):
        jsonpath = Path(jsonpath).resolve()
        rel_json = jsonpath.relative_to(jsonpath.anchor)
        latentpath = Path(self.latentdir) / self.vma_check_hash / rel_json.with_suffix(".json")
        latentpath = Path(str(latentpath).replace("-parsed.json", "-latent.json"))
        latentpath.parent.mkdir(parents=True, exist_ok=True)
        return latentpath

    def to_json_path(self, latentpath):
        latentpath = Path(latentpath).resolve()
        rel_path = latentpath.relative_to(Path(self.latentdir) / self.vma_check_hash)
        jsonpath = Path("/") / rel_path.with_suffix(".json")
        return jsonpath


    @lru_cache(maxsize=8192)
    def get_latents_for_robot(self, robot_path):
        latentpath = self.to_latent_path(robot_path)

        if os.path.exists(latentpath):
            pass
        else:
            possible_paths = []
            for root, _, files in os.walk(self.latentdir):
                for file in files:

                    if file.replace("-latent.json", "") == robot_path.replace(".xml", "").replace("-parsed.json", "").replace("-latent.json", ""):
                        possible_path = os.path.join(root, file)
                        possible_paths.append(possible_path)
            assert len(possible_paths) == 1
            latentpath = possible_paths[0]

        with open(latentpath, "r") as f:
            latents = json.load(f)
        ret = {}
        for k, v in latents.items():
            if isinstance(v, list):
                v = np.array(v)
            ret[k] = v
        return ret

    def write_latents(self, robot_dataset_path: Union[Path, str], normalization="zscore"):
        import torch

        from voxvae.dataloading.dataloader import load_for_inference


        device = torch.device(self.inference_device)
        vma = self.vma.to(device)

        json_paths = []
        for root, _, files in os.walk(robot_dataset_path):
            for file in files:
                json_path = os.path.join(root, file)
                if json_path.endswith(".json") and "/metadata/" not in json_path:
                    json_paths.append(json_path)
        json_paths = json_paths[:40]

        CHUNKSIZE = 30
        json_paths = [json_paths[i:i + CHUNKSIZE] for i in range(0, len(json_paths), CHUNKSIZE)]

        JPATHS, RCS, LATENTS = [], [], []
        for jbatch in tqdm(json_paths):
            jsonpaths, robotcomponents, batch = load_for_inference(jbatch, self.cfg)
            batch = batch[None]
            latents = vma.get_latent(batch.to(device)).squeeze().tolist()

            JPATHS.extend(jsonpaths)
            RCS.extend(robotcomponents)
            LATENTS.append(latents)

        # normalize
        LATENTS = np.vstack([np.array(x) for x in LATENTS])
        normalization_func = get_normalization_func(normalization)
        LATENTS, normalization_vals = normalization_func(LATENTS)
        LATENTS = LATENTS.tolist()

        os.makedirs(self.latentdir, exist_ok=True)
        with open(f"{self.latentdir}/normalization_config.json", "w") as f:
            normalization_vals = [x.tolist() for x in normalization_vals]
            json.dump({"normalization_name": normalization, "normalization_vals": normalization_vals}, f)

        CACHE = {}
        for jpath, rc, l in tqdm(zip(JPATHS, RCS, LATENTS), total=len(JPATHS)):
            if jpath not in CACHE:
                CACHE[jpath] = {}
            CACHE[jpath][rc] = l

        for k, j in CACHE.items():
            with open(self.to_latent_path(k), "w") as f:
                json.dump(j, f)

    def get_latent_for_robotcomponent(self, robot_path: str, robot_component: str):
        latents = self.get_latents_for_robot(robot_path)
        assert robot_component in latents
        return latents[robot_component]

    def visualize_robot_component(self, jsonpath, robotcomponent):
        from voxvae.dataloading.dataloader import load_for_inference
        jsonpaths, robotcomponents, batch = load_for_inference([jsonpath], self.cfg)

        for jpath, rc, b in zip(jsonpaths, robotcomponents, batch):
            if rc == robotcomponent:
                break

        import torch
        device = torch.device(self.inference_device)
        vma = self.vma.to(device)
        b = b[None]
        pred = vma(b.to(device)).squeeze().argmax(dim=0)

        from voxvae.pcd.pcd_vis import visualize_voxgrid
        visualize_voxgrid(b.cpu().numpy())
        visualize_voxgrid(pred.cpu().numpy())


def get_normalization_func(name, vals=None):
    if vals is not None:
        vals = [np.array(v) for v in vals]

    def normalization_zscore(vals, matrix):
        # Assuming latent_vectors is a (2500, d) numpy array
        if vals is not None:
            mean, std = vals
        else:
            mean = np.mean(matrix, axis=0)
            std = np.std(matrix, axis=0)
            vals = mean, std

        normalized_vectors = (matrix - mean) / (std + 1e-8)  # Small epsilon to avoid division by zero
        return normalized_vectors, vals

    def normalization_perdim_minmax(vals, matrix):
        # Assuming latent_vectors is a (2500, d) numpy array
        if vals is not None:
            min_vals, max_vals = vals
        else:
            min_vals = np.min(matrix, axis=0)  # Per-dimension min
            max_vals = np.max(matrix, axis=0)  # Per-dimension max
            vals = min_vals, max_vals

        # Avoid division by zero (if max == min, set normalized value to 0.5 or handle separately)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Prevent division by zero

        normalized_vectors = (matrix - min_vals) / range_vals
        return normalized_vectors, vals

    def normalization_global_minmax(vals, matrix):
        if vals is not None:
            global_min, global_max = vals
        else:
            global_min = np.min(matrix)
            global_max = np.max(matrix)
            vals = global_min, global_max

        global_range = global_max - global_min

        normalized_vectors = (matrix - global_min) / global_range
        return normalized_vectors, vals

    if name == "zscore":
        func = normalization_zscore
    elif name == "perdim_minmax":
        func = normalization_perdim_minmax
    elif name == "global_minmax":
        func = normalization_global_minmax

    return functools.partial(func, vals)

if __name__ == "__main__":
    vma = VMA("./saved_runs/upsample_resnet/WTrue_L32", "./latentdir", inference_device="cpu")
    #vma.write_latents("/home/charlie/Desktop/MJCFConvert/mjcf2o3d/unimals_100/")
    #x = vma.get_latents_for_robot("floor-5506-10-6-01-15-48-35_damping_3-latent.json")
    #y = vma.get_latent_for_robotcomponent("floor-5506-10-6-01-15-48-35_damping_3-latent.json", "limby/6")
    vma.visualize_robot_component("/home/charlie/Desktop/MJCFConvert/mjcf2o3d/unimals_100/dynamics/xml/floor-5506-10-6-01-15-48-35_damping_3-parsed.json", "limby/6")