from typing import Union

import numpy as np
import plotly
from jaxvox import VoxCol, VoxGrid
import jax.numpy as jnp
from matplotlib import pyplot as plt

from voxvae.jaxutils import bool_ifelse


def pc_to_pcd(p, c):
    p = np.asarray(p, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    # todo merge with pointcloud.py
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p)
    pcd.colors = o3d.utility.Vector3dVector(c)
    return pcd

def pc_marshall(p,c, num_points):
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


def visualize_voxgrid(voxgrid: Union[jnp.ndarray, VoxGrid]):
    if isinstance(voxgrid, np.ndarray) or isinstance(voxgrid, jnp.ndarray):
        temp = VoxGrid.build_from_bounds(jnp.zeros(3), jnp.ones(3), 1 / voxgrid.shape[0])
        voxgrid = temp.replace(grid=voxgrid)

    voxgrid.display_as_o3d(attrmanager=plt.get_cmap("gist_rainbow"))


import plotly.graph_objects as go
def plotly_v(v, show=False):
    N = v.shape[0]

    v = np.asarray(v).flatten()

    X, Y, Z = np.mgrid[0:1:N*1j, 0:1:N*1j, 0:1:N*1j]

    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=v,
        isomin=0.1,
        isomax=1.0,
        opacity=0.1,  # needs to be small to see through all surfaces
        surface_count=17,  # needs to be a large number for good volume rendering
    ))

    if show:
        fig.show()
    return plotly.io.to_html(fig)



def vis_v(v):
    return visualize_voxgrid(v)
    m = p == 0.33
    return vis_pm(p,m)


def vis_pm(p,m):
    return visualize_pcd(pc_to_pcd(p, bool_ifelse(m, jnp.array([1,0,0]), jnp.array([0,0,1]))))

import open3d as o3d
def visualize_pcd(pcd):

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(pcd)
    visualizer.poll_events()
    visualizer.update_renderer()

    try:
        visualizer.run()
    except KeyboardInterrupt:
        pass