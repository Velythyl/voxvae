import jax
import numpy as np
from jax import numpy as jnp
from jax._src.scipy.spatial.transform import Rotation as R



def pc_marshall(p, c, num_points):
    num_original_points = p.shape[0]

    if num_original_points > num_points:
        p,c = pc_downsample(p,c, num_points)

    elif num_original_points < num_points:
        p,c = pc_upsample(p,c, num_points)

    assert p.shape[0] == num_points

    return p,c


def pc_upsample(p, c, num_final):
    assert isinstance(num_final, int)

    num_create = num_final - p.shape[0]

    indices = np.arange(p.shape[0])

    def is_single_color(c):
        return (c[1:] == c[:-1]).all()

    if is_single_color(c):
        # in this case, we have a single-ID point cloud and we can directly resample
        selected_indices = np.random.choice(indices, size=num_create, replace=False, p=None)

    else:
        unique_colors, num_each_color = np.unique(c, axis=0, return_counts=True)
        color_probabilities = num_each_color / num_each_color.sum()  # Normalize to get probabilities

        # Determine how many points to keep per color
        num_keep_per_color = np.round(color_probabilities * num_create).astype(int)

        # Ensure the total number of selected points is exactly num_keep
        num_keep_per_color[np.argmax(num_keep_per_color)] += num_create - num_keep_per_color.sum()

        selected_indices = []
        for color, keep_count in zip(unique_colors, num_keep_per_color):
            color_indices = indices[(c == color).all(axis=1)]
            selected_indices.extend(np.random.choice(color_indices, size=keep_count, replace=False))

        selected_indices = np.array(selected_indices)

    new_ps, new_cs = p[selected_indices], c[selected_indices]
    return np.concatenate((p, new_ps)), np.concatenate((c, new_cs))


def pc_downsample(p, c, num_keep):
    assert isinstance(num_keep, int)

    indices = np.arange(p.shape[0])

    def is_single_color(c):
        return (c[1:] == c[:-1]).all()

    if is_single_color(c):
        # in this case, we have a single-ID point cloud and we can directly resample
        selected_indices = np.random.choice(indices, size=num_keep, replace=False, p=None)

    else:
        unique_colors, num_each_color = np.unique(c, axis=0, return_counts=True)
        color_probabilities = num_each_color / num_each_color.sum()  # Normalize to get probabilities

        # Determine how many points to keep per color
        num_keep_per_color = np.round(color_probabilities * num_keep).astype(int)

        # Ensure the total number of selected points is exactly num_keep
        num_keep_per_color[np.argmax(num_keep_per_color)] += num_keep - num_keep_per_color.sum()

        selected_indices = []
        for color, keep_count in zip(unique_colors, num_keep_per_color):
            color_indices = indices[(c == color).all(axis=1)]
            selected_indices.extend(np.random.choice(color_indices, size=keep_count, replace=False))

        selected_indices = np.array(selected_indices)

    return p[selected_indices], c[selected_indices]


EPSILON = 0.0001


def p_rescale_01(p):
    translate = 0 - p.min(axis=0)
    translated_p = p + translate + EPSILON
#    assert (translated_p.min(axis=0) >= 0).all()

    downscale_factor = 1 / (translated_p.max() + EPSILON)

    return translated_p * downscale_factor # (lambda p,c: ((p + translate + EPSILON) * downscale_factor,c))


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
