import functools
import random

import jax
import numpy as np
from jax import numpy as jnp
from jax._src.scipy.spatial.transform import Rotation as R


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


import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple


def split_into_patches(volume, patch_size):
    """Split a 3D volume into smaller 3D patches."""
    d, h, w = volume.shape
    #assert d % patch_size == 0 and h % patch_size == 0 and w % patch_size == 0, \
    #    "Volume dimensions must be divisible by patch_size"

    n_d = d // patch_size
    n_h = h // patch_size
    n_w = w // patch_size

    patches = jnp.reshape(volume,
                          (n_d, patch_size,
                           n_h, patch_size,
                           n_w, patch_size))
    patches = jnp.transpose(patches, (0, 2, 4, 1, 3, 5))
    return patches


def merge_patches(patches):
    """Merge 3D patches back into a complete volume."""
    n_d, n_h, n_w, patch_size, _, _ = patches.shape
    patches = jnp.transpose(patches, (0, 3, 1, 4, 2, 5))
    volume = jnp.reshape(patches,
                         (n_d * patch_size,
                          n_h * patch_size,
                          n_w * patch_size))
    return volume


def shuffle_patches2(key, patches):
    """Shuffle 3D patches along the first 3 dimensions."""
    n_d, n_h, n_w = patches.shape[:3]

    # Flatten the patch grid indices
    patch_indices = jnp.arange(n_d * n_h * n_w)

    # Shuffle the indices
    shuffled_indices = jax.random.permutation(key, patch_indices)

    # Reshape back to 3D grid
    shuffled_indices_3d = jnp.reshape(shuffled_indices, (n_d, n_h, n_w))

    # Use shuffled indices to reorder patches
    shuffled_patches = jnp.zeros_like(patches)
    for i in range(n_d):
        for j in range(n_h):
            for k in range(n_w):
                src_i, src_j, src_k = jnp.unravel_index(
                    shuffled_indices_3d[i, j, k], (n_d, n_h, n_w))
                shuffled_patches = shuffled_patches.at[i, j, k].set(
                    patches[src_i, src_j, src_k])

    return shuffled_patches


def shuffle_patches(key, patches):
    """Shuffle 3D patches along the first 3 dimensions - FAST VERSION."""
    #n_d, n_h, n_w = patches.shape[:3]

    # Flatten patches and shuffle
    patches_flat = patches.reshape(-1, *patches.shape[3:])
    shuffled_flat = jax.random.permutation(key, patches_flat)

    # Reshape back to original structure
    return shuffled_flat.reshape(patches.shape)

def patch_shuf(key, volume, patch_size: int):
    patches = split_into_patches(volume, patch_size)
    patches = shuffle_patches(key, patches)
    return merge_patches(patches)

#@functools.partial(jax.jit, static_argnums=2)
def _hybridize_voxel_grids(grid1: jnp.ndarray, grid2: jnp.ndarray) -> jnp.ndarray:
    """
    Create a hybrid voxel grid by combining half of grid1 with half of grid2.

    Args:
        grid1: First voxel grid (32×32×32)
        grid2: Second voxel grid (32×32×32)
        axis: Axis along which to split and combine (0, 1, or 2)

    Returns:
        Hybrid voxel grid (32×32×32)
    """
    axis = 0
    # Split each grid into two halves along the specified axis
    split_idx = grid1.shape[0] // 2
    half1 = jnp.take(grid1, jnp.arange(split_idx), axis=axis)
    half2 = jnp.take(grid2, jnp.arange(split_idx, grid2.shape[0]), axis=axis)

    # Concatenate the halves
    hybrid_grid = jnp.concatenate([half1, half2], axis=axis)
    return hybrid_grid

def hybridize(key: jax.random.PRNGKey, batch: jnp.ndarray) -> jnp.ndarray:
    """
    Create a batch of hybrid grids by randomly pairing samples.

    Args:
        key: JAX random key
        batch: Input batch of shape (N, 32, 32, 32)

    Returns:
        Hybrid batch of same shape
    """
    # Shuffle the batch to create random pairs
    shuffled = jax.random.permutation(key, batch)

    # Randomly choose axis for each pair (0, 1, or 2)
    #key, rng = jax.random.split(key)
    #axes = jax.random.randint(rng, (batch.shape[0],), 0, 4)

    # Create hybrids
    hybrid_batch = jax.vmap(_hybridize_voxel_grids)(batch, shuffled)
    return hybrid_batch