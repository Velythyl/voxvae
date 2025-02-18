import functools
import gzip
import json
import os

import random
from typing_extensions import Self
from typing import List, Callable, Tuple

import jax
import jaxvox
from jax._src.tree_util import Partial
from jaxvox import VoxGrid
from tqdm import tqdm
import numpy as np
from flax import struct
import jax.numpy as jnp

from voxvae.jaxutils import bool_ifelse
from voxvae.o3d_utils import pc_marshall

from voxvae.o3d_utils import visualize_voxgrid


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

    @classmethod
    def create(cls, np_mjcfs: List[NP_MJCF]):
        #indices_map = []
        flat_map = []

        for file_idx, np_mjcf in enumerate(np_mjcfs):
            num_pcds = len(np_mjcf)
            indices = list(np.arange(num_pcds))
            #indices_map.append(indices)

            # Populate the flat_map with (file index, local index) pairs
            for local_idx in indices:
                flat_map.append((file_idx, local_idx))

        return cls(np_mjcfs, jnp.array(flat_map))

    def __getitem__(self, i):
        file_idx, local_idx = self.flat_map[i]
        return self.np_mjcfs[file_idx][local_idx]

    def __len__(self):
        return len(self.flat_map)


from jax.scipy.spatial.transform import Rotation as R
def random_3drot(key, pcd_points: jnp.ndarray):
    def random_rotation_matrix(key):
        """Generates a random 3D rotation matrix using JAX."""
        angles = jax.random.uniform(key, shape=(3,), minval=0.0, maxval=2 * jnp.pi)
        rot_x = R.from_euler('x', angles[0]).as_matrix()
        rot_y = R.from_euler('y', angles[1]).as_matrix()
        rot_z = R.from_euler('z', angles[2]).as_matrix()
        return rot_z @ rot_y @ rot_x  # Combined rotation matrix

    rot_matrix = random_rotation_matrix(key)
    return jnp.dot(pcd_points, rot_matrix.T)

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


    def pm_to_voxgrid(self, p, m):
        is_c = 1 * jnp.ones(p.shape[0]) * m
        is_not_c = 2 * jnp.ones(p.shape[0]) * jnp.logical_not(m)

        def set_p(p, val):


        is_c_voxgrid = self.empty_voxgrid.set_point(p, is_c)
        is_not_c_voxgrid = self.empty_voxgrid.set_point(p, is_not_c)

        combined_voxgrid = (is_c_voxgrid + is_not_c_voxgrid) / 2  # {"is": 0.5, "isnotis": 1.5, "isnot: 1}

        combined_voxgrid = combined_voxgrid.at[combined_voxgrid == 0.5].set(self.pcd_is)
        combined_voxgrid = combined_voxgrid.at[combined_voxgrid == 1.5].set(self.pcd_isnotis)
        combined_voxgrid = combined_voxgrid.at[combined_voxgrid == 1.0].set(self.pcd_isnot)

        return combined_voxgrid.grid

    #@jax.jit
    def vox_augment(self, key, p, m):
        p = self.augment_data(key, p)
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
        new_shuf = self._shuf(key) if oob else self.shuf

        indices = new_shuf[new_idx]

        ps, ms = list(zip(*[self.dataset[i] for i in indices]))
        ps = jnp.array(ps)
        ms = jnp.array(ms)


        x = self.vox_augment(key, ps[0], ms[0])

        return jax.vmap(functools.partial(self.vox_augment, key))( ps, ms)


class DATALOADER:
    def __init__(self, dataset, batch_size, shuf=True, grid_size=64, augment_data: Callable = random_3drot):
        self.dataset = dataset
        self.shuf = shuf
        self.jax_key = jax.random.PRNGKey(np.random.randint(99999))
        self.augment_data = augment_data

        self.shuf_idx = 0

        self.grid_size = grid_size
        self.voxel_size = 1 / grid_size

        self.batch_size = batch_size

        empty_voxgrid = jaxvox.VoxGrid.build_from_bounds(jnp.ones(3) * 0, jnp.ones(3) * 1, voxel_size=self.voxel_size)

        def pm_to_voxgrid(p, m):
            is_c = 1 * jnp.ones(p.shape[0]) * m
            is_not_c = 2 * jnp.ones(p.shape[0]) * m

            is_c_voxgrid = empty_voxgrid.set_point(p, is_c)
            is_not_c_voxgrid = empty_voxgrid.set_point(p, is_not_c)

            combined_voxgrid = (is_c_voxgrid + is_not_c_voxgrid) / 2    # 1: is, 1.5: is and is not, 2: is not
            return combined_voxgrid.grid

        self.pm_to_voxgrid = Partial(jax.jit(pm_to_voxgrid))


    def get_batch(self):
        if self.shuf_idx + self.batch_size > len(self.dataset):
            self.shuf_idx = 0  # Reset index if out of range
            np.random.shuffle(self.shuf)

        batch_indices = self.shuf[self.shuf_idx:self.shuf_idx + self.batch_size]
        self.shuf_idx += self.batch_size

        ps, ms = list(zip(*[self.dataset[i] for i in batch_indices]))
        ps = jnp.array(ps)
        ms = jnp.array(ms)

        ps = jax.vmap(self.augment_data)(ps)
        batch_data = jax.vmap(self.pm_to_voxgrid)(ps, ms)

        return batch_data


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
        if partname in [">FULL<", ">ORIGINAL XML<"]:
            continue
        unique_colors.append(subdict["color"])

    p,c = ret[">FULL<"]["pcd_points"], ret[">FULL<"]["pcd_colors"]
    p,c = pc_marshall(p, c, 4096)

    return NP_MJCF(p,c, np.array(unique_colors))

def load_dataset(root):
    """
    Traverse a directory tree, find all XML files, and call the `main` function for each XML file.

    Args:
        mjcf_tree (str): The root directory of the XML file tree.
        do_visualize (bool): Whether to visualize the point cloud.
        isolate_actuators (bool): Whether to isolate actuators in the point cloud.
    """
    # Collect and filter XML file paths
    json_paths = []
    for root, _, files in os.walk(root):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                json_paths.append(json_path)

    # Process files with tqdm progress bar
    npmjcfs = []

    pbar = tqdm(json_paths, desc="Processing JSON files")
    for json_file in pbar:
        pbar.set_postfix_str(json_file)
        npmjcf = load_json_to_npmjcf(json_file)
        npmjcfs.append(npmjcf)

    return DATASET.create(npmjcfs)

def get_dataloader(root, jax_key):
    return DATALOADER.create(load_dataset(root), shuffle=True, key=jax_key)


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
def get_dataloaders(root, grid_size, batch_size, splits=(80,10,10)):

    # Collect and filter XML file paths
    json_paths = []
    for root, _, files in os.walk(root):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                json_paths.append(json_path)

    json_paths = np.array(json_paths[:10])
    np.random.shuffle(json_paths)

    with mp.Pool(6) as pool:
        pool.imap(load_json_to_npmjcf, json_paths)

    with mp.Pool(mp.cpu_count()-2) as pool:
        jsons = list(tqdm(pool.imap_unordered(load_json_to_npmjcf, json_paths),
                          total=len(json_paths),
                          desc="Loading PCDs from JSON"))

    splits = split_counts(len(json_paths), splits)

    train = jsons[:splits[0]]
    test = jsons[splits[0]:splits[1]+splits[0]]
    val = jsons[splits[1]+splits[0]:]

    def get_DL(jsons):
        return DL.create(DATASET.create(jsons), grid_size=grid_size, batch_size=batch_size)

    return get_DL(train), get_DL(test), get_DL(val)

if __name__ == "__main__":
    train, test, val = get_dataloaders("/home/charlie/Desktop/MJCFConvert/mjcf2o3d/unimals_100/dynamics/xml", 64, 10)

    x = train.get_batch_(jax.random.PRNGKey(0))
    visualize_voxgrid(x[0])