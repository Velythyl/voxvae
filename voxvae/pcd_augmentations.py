import jax
from jax import numpy as jnp
from jax._src.scipy.spatial.transform import Rotation as R

from voxvae.jaxutils import bool_ifelse


def horizontal_symmetry(pcd_points: jnp.ndarray):
    """
    Flips the point cloud horizontally (along the X-axis).
    """
    # Flip the X-coordinates
    return pcd_points * jnp.array([-1, 1, 1])

def vertical_symmetry(pcd_points: jnp.ndarray):
    """
    Flips the point cloud vertically (along the Y-axis).
    """
    # Flip the Y-coordinates
    return pcd_points * jnp.array([1, -1, 1])

def depthwise_symmetry(pcd_points: jnp.ndarray):
    """
    Flips the point cloud depthwise (along the Z-axis).
    """
    # Flip the Z-coordinates
    return pcd_points * jnp.array([1, 1, -1])

def symmetries(key, pcd_points: jnp.ndarray):
    do_hor, do_ver, do_dep = jax.random.uniform(key, shape=(3,), minval=0, maxval=1) > 0.5


    horflip = bool_ifelse(do_hor, horizontal_symmetry(pcd_points), pcd_points)
    verflip = bool_ifelse(do_hor, vertical_symmetry(horflip), horflip)
    depflip = bool_ifelse(do_hor, depthwise_symmetry(verflip), verflip)

    return depflip

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
