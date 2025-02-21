import jax
import numpy as np
import wandb

from voxvae.pcd.pcd_vis import plotly_v
from voxvae.training.models.prepare_batch import prepare_batch

import jax.numpy as jnp
import equinox as eqx


@eqx.filter_jit
def accuracy_nonzero_voxels(model, batch):
    batch = prepare_batch(batch)
    pred_batch = jax.vmap(model.call_shunt)(batch)

    batch = batch.flatten()
    pred_batch = pred_batch.flatten()

    nonzero_mask = jnp.logical_not(batch == 0)
    num_nonzero = nonzero_mask.sum()

    return ((batch == pred_batch) * nonzero_mask).sum() / num_nonzero

@eqx.filter_jit
def accuracy(model, batch):
    batch = prepare_batch(batch)
    pred_batch = jax.vmap(model.call_shunt)(batch)

    batch = batch.flatten()
    pred_batch = pred_batch.flatten()

    return (batch == pred_batch).mean()


def get_vis(model, x, prefix, midfix=""):
    return {
        f"{prefix}/{midfix}_gt": wandb.Html(plotly_v(x)),
        f"{prefix}/{midfix}_pred": wandb.Html(plotly_v(model.call_shunt(x[None, :, :, :])))
    }

def vis(key, model, loader, prefix):
    for i in range(loader.num_batch_per_epoch):
        loader, batch = loader.get_batch_(key)
        if i == 0:
            first_element = batch[0]
    last_element = batch[-1]

    dico = get_vis(model, first_element, prefix, "A")
    dico.update(get_vis(model, last_element, prefix, "B"))
    return dico

def metrics(key, model, loader, prefix, loss_fn):
    accs_nonzero = []
    accs_full = []
    losses = []
    for _ in range(loader.num_batch_per_epoch):
        loader, batch = loader.get_batch_(key)
        accs_nonzero.append(accuracy_nonzero_voxels(model, batch))
        accs_full.append(accuracy(model, batch))
        losses.append(loss_fn(model, prepare_batch(batch)))

    dico = {}
    dico[f"{prefix}/acc_nonzero"] = np.mean(np.array(accs_nonzero))
    dico[f"{prefix}/acc_full"] = np.mean(np.array(accs_full))
    dico[f"{prefix}/losses"] = np.mean(np.array(losses))

    return dico