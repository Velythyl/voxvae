import functools
import random
import time

import jax
import jax.numpy as jnp
import jaxvox
import torch
import torch2jax

from voxvae.pcd.pcd_utils import  p_rescale_01
from voxvae.utils.jaxutils import bool_ifelse, map_ternary, split_key




from voxvae.dataloading.data_aug import random_3drot, patch_shuf, hybridize


def get_collation_fn(voxgrid_size, pcd_is, pcd_isnotis, pcd_isnot, disable_random_3d_rot=False, handle_singular_only=False, data_aug=None):
    empty_voxgrid = jaxvox.VoxGrid.build_from_bounds(jnp.ones(3) * 0, jnp.ones(3) * 1, voxel_size=1 / voxgrid_size)


    def handle_singular(key, points, masks):
        og_points = points


        if not disable_random_3d_rot:
            key, rng = jax.random.split(key)
            points = random_3drot(rng, points)
        points = p_rescale_01(points)


        if data_aug is not None and data_aug.mutate:
            # this takes rotated robot, adds it on top of non rotate robot, and then rotates the whole thing
            # happens with 50% chance

            augmented_points = jnp.concatenate((points, p_rescale_01(og_points)))
            key, rng = jax.random.split(key)
            augmented_points = random_3drot(rng, augmented_points)
            augmented_points = p_rescale_01(augmented_points)

            key, rng = jax.random.split(key)
            selected = jax.random.randint(rng, (1,), 0, 2)
            masks = jnp.concatenate([masks, masks])
            points = selected * (jnp.concatenate([points, points])) + (1-selected) * augmented_points

        v = empty_voxgrid.point_to_voxel(points)

        v_is = bool_ifelse(masks, v, empty_voxgrid.padded_error_index_array)
        v_isnot = bool_ifelse(jnp.logical_not(masks), v, empty_voxgrid.padded_error_index_array)

        grid_is = empty_voxgrid.set_voxel(v_is).grid
        grid_isnot = empty_voxgrid.set_voxel(v_isnot).grid

        combined_voxgrid = (grid_is * 1 + grid_isnot * 2)  # {"is": 0.5, "isnotis": 1.5, "isnot: 1}

        CUR_IS = 1
        CUR_ISNOT = 2
        CUR_ISNOTIS = 3

        rebuild_grid = empty_voxgrid.grid
        rebuild_grid = jax.numpy.where(combined_voxgrid == CUR_IS, pcd_is, rebuild_grid)
        rebuild_grid = jax.numpy.where(combined_voxgrid == CUR_ISNOTIS, pcd_isnotis, rebuild_grid)
        rebuild_grid = jax.numpy.where(combined_voxgrid == CUR_ISNOT, pcd_isnot, rebuild_grid)

        if data_aug is not None and data_aug.patch_shuf:
            key, rng = jax.random.split(key)
            selected = jax.random.randint(rng, (1,), 0, 2)
            rebuild_grid = rebuild_grid * selected + (1-selected) * patch_shuf(key, rebuild_grid, 8)



        return rebuild_grid

    if handle_singular_only:
        return handle_singular

    def collate_fn(key, points, mask):
        key, rngs = split_key(key, points.shape[0])
        voxgrid = jax.vmap(handle_singular)(rngs, points, mask)

        if data_aug is not None and data_aug.hybridize:
            key, rng = jax.random.split(key)
            selected = jax.random.randint(rng, (1,), 0, 2)
            voxgrid = voxgrid * selected + (1-selected) * hybridize(key, voxgrid)

        return voxgrid[:,None,:,:,:]

    collate_fn = jax.jit(collate_fn)


    def torch_collate(batch):
        points, masks = zip(*batch)
        points = jnp.array(points)
        masks = jnp.array(masks)

        key = jax.random.PRNGKey(time.time_ns() - random.randint(0,100))  # gross
        batch = collate_fn(key, points, masks)
        batch = torch2jax.j2t(batch)
        return batch

    return torch_collate
