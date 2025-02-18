import functools
import gzip
import json
import os

from typing_extensions import Self
from typing import List, Callable

import jax
import jaxvox
from jax._src.tree_util import Partial
from jaxvox import VoxGrid
from tqdm import tqdm
import numpy as np
from flax import struct
import jax.numpy as jnp

from voxvae.jaxutils import bool_ifelse
from voxvae.o3d_utils import pc_marshall, p_rescale_01, vis_pm

from voxvae.pcd_augmentations import random_3drot
from voxvae.remap_voxseg import map_tensor


@struct.dataclass
class NP_MJCF:
    pcd_points: np.ndarray
    pcd_colors: np.ndarray
    pcd_unique_colors: np.ndarray
    indicators: tuple = (-1, 1)

    def __len__(self):
        return len(self.pcd_unique_colors)

    def __getitem__(self, i):
        mask = np.all(self.pcd_colors == self.pcd_unique_colors[i], axis=1)
        _mask = mask.astype(float)[:, None]
        # no meaning here
        #_pcd_colors = (self.indicators[1] * np.ones_like(self.pcd_colors) * _mask +
        #               self.indicators[0] * np.ones_like(self.pcd_colors) * np.logical_not(_mask))
        return self.pcd_points, mask


@struct.dataclass
class DATASET:
    np_mjcfs: List[NP_MJCF]
    #indices_map: List[List[int]]
    flat_map: jnp.ndarray #List[Tuple[int, int]]
    min_num_points: int = 20

    @classmethod
    def create(cls, np_mjcfs: List[NP_MJCF], min_num_points: int = 2):
        #indices_map = []
        flat_map = []

        for file_idx, np_mjcf in enumerate(np_mjcfs):
            num_pcds = len(np_mjcf)
            indices = list(np.arange(num_pcds))
            #indices_map.append(indices)

            # Populate the flat_map with (file index, local index) pairs
            for local_idx in indices:

                _,m = np_mjcf[local_idx]
                if m.sum() < min_num_points:
                    continue

                flat_map.append((file_idx, local_idx))

        return cls(np_mjcfs, jnp.array(flat_map), min_num_points=min_num_points)

    def __getitem__(self, i):
        file_idx, local_idx = self.flat_map[i]
        return self.np_mjcfs[file_idx][local_idx]

    def __len__(self):
        return len(self.flat_map)


@struct.dataclass
class DL_SHUF:
    shuf: jnp.ndarray
    idx: jnp.ndarray
    batch_size: int

    def next_(self, key):
        oob = self.idx + self.batch_size > self.shuf.shape[0]

        new_idx = bool_ifelse(oob, 0, self.idx + self.batch_size)
        new_shuf = bool_ifelse(oob, jax.random.choice(key, jnp.arange(self.shuf.shape[0]), replace=False), self.shuf)
        return DL_SHUF(new_shuf, new_idx)

    def get_batch_indices(self):
        return self.shuf[self.idx:self.idx+self.batch_size]

@struct.dataclass
class DL:
    dataset: DATASET

    augment_data: Partial
    empty_voxgrid: VoxGrid

    shuf: jnp.ndarray
    shuf_idx: int = struct.field(pytree_node=False)

    batch_size: int = struct.field(pytree_node=False)
    grid_size: int = struct.field(pytree_node=False)
    dataset_len: int = struct.field(pytree_node=False)

    pcd_is: float = 0.33
    pcd_isnotis: float = 0.66
    pcd_isnot: float = 0.99

    @property
    def voxel_size(self):
        return self.empty_voxgrid.voxel_size

    @classmethod
    def create(cls, dataset, batch_size, grid_size, augment_data=random_3drot, pcd_is: float = 0.33, pcd_isnotis: float = 0.66, pcd_isnot: float = 0.99) -> (Self, DL_SHUF):
        if batch_size > len(dataset):
            batch_size = len(dataset)

        temp = cls(
            dataset=dataset,
            batch_size=batch_size,
            grid_size=grid_size,
            augment_data=Partial(augment_data),
            empty_voxgrid=jaxvox.VoxGrid.build_from_bounds(jnp.ones(3) * 0, jnp.ones(3) * 1, voxel_size=1/grid_size),
            shuf=jnp.arange(len(dataset)),
            dataset_len=len(dataset),
            shuf_idx=0, pcd_is=pcd_is, pcd_isnotis=pcd_isnotis, pcd_isnot=pcd_isnot)

        shuf = temp._shuf(jax.random.PRNGKey(0))
        return temp.replace(
            shuf=shuf,
        )

    @property
    def num_batch_per_epoch(self):
        return self.shuf.shape[0]

    def pm_to_voxgrid(self, p, m):
        v = self.empty_voxgrid.point_to_voxel(p)

        v_is = bool_ifelse(m, v, self.empty_voxgrid.padded_error_index_array)
        v_isnot = bool_ifelse(jnp.logical_not(m), v, self.empty_voxgrid.padded_error_index_array)

        grid_is = self.empty_voxgrid.set_voxel(v_is).grid
        grid_isnot = self.empty_voxgrid.set_voxel(v_isnot).grid

        combined_voxgrid = (grid_is * 1 + grid_isnot * 2)  # {"is": 0.5, "isnotis": 1.5, "isnot: 1}

        combined_voxgrid = map_tensor(combined_voxgrid, (self.pcd_is, self.pcd_isnotis, self.pcd_isnot))

        return combined_voxgrid

    #@jax.jit
    def vox_augment(self, key, p, m):
        p = self.augment_data(key, p)
        p = p_rescale_01(p)
        return self.pm_to_voxgrid(p, m)

    @jax.jit
    def _shuf(self, key):
        shuf = jax.random.choice(key, jnp.arange(self.dataset_len, dtype=jnp.int32), replace=False, shape=(self.dataset_len,))

        num_batches = self.dataset_len // self.batch_size
        shuf = shuf[:int(num_batches * self.batch_size)]
        shuf = shuf.reshape((num_batches, self.batch_size))

        return shuf

    def get_batch_(self, key):
        oob = self.shuf_idx >= self.shuf.shape[0]

        new_idx = 0 if oob else self.shuf_idx
        key, rng = jax.random.split(key)
        new_shuf = self._shuf(rng) if oob else self.shuf

        indices = new_shuf[new_idx]

        ps, ms = list(zip(*[self.dataset[i] for i in indices]))
        ps = jnp.array(ps)
        ms = jnp.array(ms)


        #x = self.vox_augment(key, ps[0], ms[0])

        return self.replace(shuf=new_shuf, shuf_idx=new_idx), jax.vmap(functools.partial(self.vox_augment, key))( ps, ms)


def load_json_to_npmjcf(inpath, array_backend=np):
    with gzip.open(inpath, 'rt', encoding='UTF-8') as zipfile:
        x = json.load(zipfile)

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

    return NP_MJCF(p,c, np.array(unique_colors))




def split_counts(total: int, percentages: List[int]) -> List[int]:
    # Ensure percentages sum to 100
    if sum(percentages) == 1:
        percentages = [p*100 for p in percentages]
    assert sum(percentages) == 100

    # Initial allocation based on percentage
    raw_counts = [max(1, round(total * p / 100)) for p in percentages]

    # Adjust to match total exactly
    difference = total - sum(raw_counts)

    # Distribute the remaining items to the largest percentage groups
    for _ in range(abs(difference)):
        if difference > 0:
            # Add 1 to the split with the highest percentage
            idx = max(range(len(percentages)), key=lambda i: (percentages[i], -raw_counts[i]))
            raw_counts[idx] += 1
        elif difference < 0:
            # Remove 1 from the split with the highest allocated count (while ensuring min 1)
            idx = max((i for i in range(len(raw_counts)) if raw_counts[i] > 1),
                      key=lambda i: raw_counts[i])
            raw_counts[idx] -= 1

    return raw_counts

import multiprocessing as mp
def get_dataloaders(root, grid_size, batch_size, splits=(80,10,10), num_workers=mp.cpu_count()-2):

    # Collect and filter XML file paths
    json_paths = []
    for root, _, files in os.walk(root):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                json_paths.append(json_path)

    json_paths = np.array(json_paths[:10])
    np.random.shuffle(json_paths)

    with mp.Pool(num_workers) as pool:
        jsons = list(
            tqdm(
                    pool.imap_unordered(load_json_to_npmjcf, json_paths, chunksize=max(len(json_paths) // (num_workers * 2), 1)),
                total=len(json_paths),
                desc="Loading PCDs from JSON"
            )
        )

    splits = split_counts(len(json_paths), splits)

    train = jsons[:splits[0]]
    test = jsons[splits[0]:splits[1]+splits[0]]
    val = jsons[splits[1]+splits[0]:]

    def get_DL(jsons):
        return DL.create(DATASET.create(jsons), grid_size=grid_size, batch_size=batch_size)

    return get_DL(train), get_DL(test), get_DL(val)

if __name__ == "__main__":
    train, test, val = get_dataloaders("/home/charlie/Desktop/MJCFConvert/mjcf2o3d/unimals_100/dynamics/xml", 32, 10)

    p,m = train.dataset[1]

    vis_pm(p,m)
    vis_pm(random_3drot(jax.random.PRNGKey(4), p), m)

    #visualize_pcd(pc_to_pcd(x[0], bool_ifelse(x[1], jnp.array([1,0,0]), jnp.array([0,0,1]))))
    exit()

    x = train.get_batch_(jax.random.PRNGKey(0))
    visualize_voxgrid(x[0])