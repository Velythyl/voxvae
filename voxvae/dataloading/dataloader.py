import dataclasses
import gzip
import os
import json
from pathlib import Path
from typing import Dict, Union, List

import jax
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm, trange

from voxvae.dataloading.splits import split_counts
from voxvae.pcd.pcd_utils import pc_marshall
from voxvae.utils.jaxutils import bool_ifelse, map_ternary
import torch2jax

import multiprocessing as mp

def load_json_to_npmjcf(inpath, array_backend=np):
    try:
        with gzip.open(inpath, 'rt', encoding='UTF-8') as zipfile:
            x = json.load(zipfile)
    except gzip.BadGzipFile:
        return inpath, None

    # loads as arrays
    ret = {}
    for partname, subdict in x.items():
        subret = {}
        for k, v in subdict.items():
            if isinstance(v, list) or isinstance(v, tuple):
                v = array_backend.asarray(v)
            subret[k] = v
        ret[partname] = subret

    p, c = ret[">FULL<"]["pcd_points"], ret[">FULL<"]["pcd_colors"]
    p, c = pc_marshall(p, c, 4096)

    unique_colors = []
    component_names = []
    for partname, subdict in ret.items():
        if partname in [">FULL<", ">ORIGINAL XML<"]:#, ">REST<"]:
            continue
        unique_colors.append(subdict["color"])
        component_names.append(partname)
        assert (unique_colors[-1] == c).all(axis=1).any()



    return inpath, {"points": p, "colors": c, "unique_colors": np.array(unique_colors), "component_names": np.array(component_names)}

def pcd_get_mask(colors, unique_color):
    mask = np.all(colors == unique_color, axis=1)
    return mask



class PointCloudDataset(Dataset):
    def __init__(self, json_paths):
        """
        Args:
            json_dir (str): Directory with all the JSON files.
            pcd_to_vox (callable): Function to convert point clouds to voxel grids.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        # Load all JSON files and store the point clouds and colors
        num_workers = max(mp.cpu_count() - 4, 1)
        print(f"Loading jsons using <{num_workers}> workers...")
        with mp.Pool(num_workers) as pool:
            jsons = list(
                tqdm(
                    pool.imap_unordered(load_json_to_npmjcf, json_paths,
                                        chunksize=max(len(json_paths) // (num_workers * 100), 1)),
                    total=len(json_paths),
                    desc="Loading PCDs from JSON"
                )
            )
        print("...done!")
        self.json_paths = [x for (x,_) in jsons]
        self.json_paths_to_components = {x:dico["component_names"] for (x,dico) in jsons}

        self.point_clouds = []
        self.pcd_colors = []
        self.unique_colors = []
        self.num_pcds = 0
        self.flat_idx_map = []
        for file_idx, (inpath, data) in enumerate(jsons):

            self.point_clouds.append(np.array(data['points']))
            self.pcd_colors.append(np.array(data['colors']))
            self.unique_colors.append(data["unique_colors"])

            num_pcds = len(data["unique_colors"])
            self.num_pcds += num_pcds

            indices = list(np.arange(num_pcds))
            # indices_map.append(indices)

            # Populate the flat_map with (file index, local index) pairs
            for local_idx in indices:
                #_, m = np_mjcf[local_idx]
                #if m.sum() < min_num_points:
                #    continue

                self.flat_idx_map.append((file_idx, local_idx))

    def __len__(self):
        return self.num_pcds

    def __getitem__(self, idx):
        file_idx, local_idx = self.flat_idx_map[idx]

        # Get the point cloud and colors
        pcd_points = self.point_clouds[file_idx]
        pcd_colors = self.pcd_colors[file_idx]
        unique_colors = self.unique_colors[file_idx]

        mask = pcd_get_mask(pcd_colors, unique_colors[local_idx])

        return pcd_points, mask

    def idx_to_jsonpath_and_robotcomponent(self, idx):
        fileidx, localidx = self.flat_idx_map[idx]
        jsonpath = self.json_paths[fileidx]
        robotcomponent = self.json_paths_to_components[jsonpath][localidx]
        return jsonpath, robotcomponent


@dataclasses.dataclass
class SplitLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader

    class_counts: Dict

    @property
    def class_weights(self):
        ret = []
        for i in range(len(self.class_counts)):
            ret.append(self.class_counts[i])
        ret = np.array(ret)

        weights = 1.0 / ret
        weights = weights / weights.sum() * len(self.class_counts)  # optional normalization
        return torch.tensor(weights, dtype=torch.float32)

def get_dataloaders(root, grid_size, batch_size, fewer_files, splits=(80,10,10), pcd_is=1.0, pcd_isnotis=2.0, pcd_isnot=3.0):
    # Create the dataset

    try:
        mp.set_start_method('spawn')
    except:
        pass
    json_paths = []
    for root, _, files in os.walk(root):
        for file in files:
            json_path = os.path.join(root, file)
            if json_path.endswith(".json") and "/metadata/" not in json_path:
                json_paths.append(json_path)

    json_paths = np.array(json_paths[:fewer_files])
    import omegaconf
    val_dataset = None
    if isinstance(splits, tuple) or isinstance(splits, list) or isinstance(splits, omegaconf.ListConfig):
        assert len(splits) in [2,3]

        splits = split_counts(len(json_paths), splits)

        train_dataset = json_paths[:splits[0]]
        test_dataset = json_paths[splits[0]:splits[1]+splits[0]]
        if len(splits) == 3:
            val_dataset = json_paths[splits[1]+splits[0]:]
    elif isinstance(splits, str) and "-" in splits:
        names = splits.split("-")
        train_dataset = [x for x in json_paths if names[0] in x]
        test_dataset = [x for x in json_paths if names[1] in x]
        if len(names) == 3:
            val_dataset = [x for x in json_paths if names[2] in x]

    train_dataset, test_dataset = PointCloudDataset(train_dataset), PointCloudDataset(test_dataset)
    if val_dataset:
        val_dataset = PointCloudDataset(val_dataset)

    from voxvae.dataloading.collator import get_collation_fn

    def create_dl(dataset):
        # Create the DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=get_collation_fn(voxgrid_size=grid_size, pcd_is=pcd_is, pcd_isnotis=pcd_isnotis, pcd_isnot=pcd_isnot))
        return dataloader


    train_dataset, test_dataset = create_dl(train_dataset), create_dl(test_dataset)
    if val_dataset:
        val_dataset = create_dl(val_dataset)

    valid_dls = [x for x in [train_dataset, val_dataset, test_dataset] if x is not None]

    class_count = {}
    for dl in valid_dls:
        for batch in dl:
            unique, uniquecounts = batch.unique(return_counts=True)

            unique = unique.cpu().tolist()
            uniquecounts = uniquecounts.cpu().tolist()

            for u, c in zip(unique, uniquecounts):
                if u not in class_count:
                    class_count[u] = 0
                class_count[u] += c

    return SplitLoaders(
        train=train_dataset,
        val=val_dataset,
        test=test_dataset,
        class_counts=class_count
    )

def load_for_inference(json_paths: Union[str, Path, List], cfg):
    if not isinstance(json_paths, list):
        json_paths = [json_paths]

    cleaned_json_paths = []
    for path in json_paths:
        if path.endswith(".xml"):
            newpath = path.replace(".xml", "-parsed.json")
            assert os.path.exists(newpath)
            cleaned_json_paths.append(newpath)
        else:
            cleaned_json_paths.append(path)
    del json_paths

    dataset = PointCloudDataset(cleaned_json_paths)

    from voxvae.dataloading.collator import get_collation_fn
    transformation_fn = get_collation_fn(voxgrid_size=cfg.dataloader.grid_size, pcd_is=cfg.datarep.pcd_is,
        pcd_isnotis=cfg.datarep.pcd_isnotis,
        pcd_isnot=cfg.datarep.pcd_isnot, disable_random_3d_rot=None, handle_singular_only=True)

    jsonpaths = []
    robotcomponents = []
    pcd_points = []
    masks = []
    for i in trange(len(dataset)):
        pcd, mask = dataset[i]
        jsonpath, robotcomponent = dataset.idx_to_jsonpath_and_robotcomponent(i)
        jsonpaths.append(jsonpath)
        robotcomponents.append(robotcomponent)
        pcd_points.append(pcd)
        masks.append(mask)

    import jax.numpy as jnp
    pcds = jnp.stack(pcd_points)
    masks = jnp.stack(masks)
    voxgrids = jax.vmap(transformation_fn)(jnp.ones((pcds.shape[0],)), pcds, masks)
    voxgrids = jnp.expand_dims(voxgrids, 1)
    voxgrids = torch2jax.j2t(voxgrids)

    return jsonpaths, robotcomponents, voxgrids

if __name__ == "__main__":
    splitloaders = get_dataloader("/home/charlie/Desktop/MJCFConvert/mjcf2o3d/unimals_100/dynamics/xml", 32, 32, 10)