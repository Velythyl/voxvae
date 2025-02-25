import functools

import jax
import jax.numpy as jnp
import jaxvox
import torch
import torch2jax

from voxvaetorch.pcd.pcd_utils import random_3drot, p_rescale_01
from voxvaetorch.utils.jaxutils import bool_ifelse, map_ternary, split_key


def get_collation_fn(voxgrid_size, pcd_is, pcd_isnotis, pcd_isnot):
    empty_voxgrid = jaxvox.VoxGrid.build_from_bounds(jnp.ones(3) * 0, jnp.ones(3) * 1, voxel_size=1 / voxgrid_size)

    def handle_singular(key, points, masks):
        points = random_3drot(key, points)
        points = p_rescale_01(points)

        v = empty_voxgrid.point_to_voxel(points)

        v_is = bool_ifelse(masks, v, empty_voxgrid.padded_error_index_array)
        v_isnot = bool_ifelse(jnp.logical_not(masks), v, empty_voxgrid.padded_error_index_array)

        grid_is = empty_voxgrid.set_voxel(v_is).grid
        grid_isnot = empty_voxgrid.set_voxel(v_isnot).grid

        combined_voxgrid = (grid_is * 1 + grid_isnot * 2)  # {"is": 0.5, "isnotis": 1.5, "isnot: 1}

        combined_voxgrid = map_ternary(combined_voxgrid, (pcd_is, pcd_isnotis, pcd_isnot))

        return combined_voxgrid

    def collate_fn(key, points, mask):
        _, keys = split_key(key, points.shape[0])
        voxgrid = jax.vmap(handle_singular)(keys, points, mask)
        return voxgrid[:,None,:,:,:]

    collate_fn = jax.jit(collate_fn)


    def torch_collate(batch):
        points, masks = zip(*batch)
        points = jnp.array(points)
        masks = jnp.array(masks)

        key = jax.random.PRNGKey(torch.randint(0, 10000, (1,)).item())  # gross
        batch = collate_fn(key, points, masks)
        batch = torch2jax.j2t(batch)
        return batch

    return torch_collate
