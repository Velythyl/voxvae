from typing import Union

import numpy as np
import plotly
from jaxvox import VoxGrid
import jax.numpy as jnp
from matplotlib import pyplot as plt

from voxvae.utils.jaxutils import bool_ifelse


def pc_to_pcd(p, c):
    p = np.asarray(p, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    # todo merge with pointcloud.py
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p)
    pcd.colors = o3d.utility.Vector3dVector(c)
    return pcd


def visualize_voxgrid(voxgrid: Union[jnp.ndarray, VoxGrid]):
    import torch
    if isinstance(voxgrid, torch.Tensor):
        voxgrid = voxgrid.cpu().numpy()

    if isinstance(voxgrid, np.ndarray) or isinstance(voxgrid, jnp.ndarray):
        voxgrid = voxgrid.squeeze()
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