import dataclasses
import gzip
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from voxvae.dataloading.splits import split_counts
from voxvae.pcd.pcd_utils import pc_marshall
from voxvae.utils.jaxutils import bool_ifelse, map_ternary

import multiprocessing as mp

def load_json_to_npmjcf(inpath, array_backend=np):
    try:
        with gzip.open(inpath, 'rt', encoding='UTF-8') as zipfile:
            x = json.load(zipfile)
    except gzip.BadGzipFile:
        return inpath, None


    ret = {}
    for partname, subdict in x.items():
        subret = {}
        for k, v in subdict.items():
            if isinstance(v, list) or isinstance(v, tuple):
                v = array_backend.asarray(v)
            subret[k] = v
        ret[partname] = subret


    unique_colors = []
    for partname, subdict in ret.items():
        if partname in [">FULL<", ">ORIGINAL XML<"]:#, ">REST<"]:
            continue
        unique_colors.append(subdict["color"])

    p,c = ret[">FULL<"]["pcd_points"], ret[">FULL<"]["pcd_colors"]
    p,c = pc_marshall(p, c, 4096)

    return inpath, {"points": p, "colors": c, "unique_colors": np.array(unique_colors)}

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
        self.json_files = json_paths

        # Load all JSON files and store the point clouds and colors
        num_workers = max(mp.cpu_count() - 4, 1)
        with mp.Pool(num_workers) as pool:
            jsons = list(
                tqdm(
                    pool.imap_unordered(load_json_to_npmjcf, json_paths,
                                        chunksize=max(len(json_paths) // (num_workers * 100), 1)),
                    total=len(json_paths),
                    desc="Loading PCDs from JSON"
                )
            )

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

@dataclasses.dataclass
class SplitLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader

    prop_empty: float
    prop_is: float
    prop_isnotis: float
    prop_isnot: float

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

    splits = split_counts(len(json_paths), splits)

    train = json_paths[:splits[0]]
    test = json_paths[splits[0]:splits[1]+splits[0]]
    val = json_paths[splits[1]+splits[0]:]

    train, test, val = PointCloudDataset(train), PointCloudDataset(test), PointCloudDataset(val)

    from voxvae.dataloading.collator import get_collation_fn

    def create_dl(dataset):
        # Create the DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=get_collation_fn(voxgrid_size=grid_size, pcd_is=pcd_is, pcd_isnotis=pcd_isnotis, pcd_isnot=pcd_isnot))
        return dataloader


    train, val, test = create_dl(train), create_dl(val), create_dl(test)


    num_empty = 0
    num_is = 0
    num_isnotis = 0
    num_isnot = 0

    for dl in [train, val, test]:
        for batch in dl:
            num_empty += (batch == 0).sum().item()
            num_is += (batch == pcd_is).sum().item()
            num_isnotis += (batch == pcd_isnotis).sum().item()
            num_isnot += (batch == pcd_isnot).sum().item()

    total = num_empty + num_is + num_isnotis + num_isnot

    return SplitLoaders(
        train=train,
        val=val,
        test=test,
        prop_empty=float(num_empty / total),
        prop_is=float(num_is / total),
        prop_isnotis=float(num_isnotis / total),
        prop_isnot=float(num_isnot / total)
    )


if __name__ == "__main__":
    splitloaders = get_dataloader("/home/charlie/Desktop/MJCFConvert/mjcf2o3d/unimals_100/dynamics/xml", 32, 32, 10)